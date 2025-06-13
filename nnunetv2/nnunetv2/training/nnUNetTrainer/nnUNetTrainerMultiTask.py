from cProfile import label
import torch
from torch import nn
from torch import distributed as dist
import os
import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.architectures.MultiTaskResEncUNet import MultiTaskResEncUNet, MultiTaskChannelAttentionResEncUNet, MultiTaskEfficientAttentionResEncUNet
from nnunetv2.training.loss.multitask_losses import MultiTaskLoss
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.collate_outputs import collate_outputs

DEBUG = os.environ.get("DEBUG", False)

class nnUNetTrainerMultiTask(nnUNetTrainerNoDeepSupervision):
    """Multi-task trainer for segmentation + classification"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Multi-task specific parameters
        self.num_classification_classes = 3
        self.seg_weight = 1.0
        self.cls_weight = 0.25
        self.running_seg_loss = 1.0
        self.running_cls_loss = 1.0
        self.running_alpha = 0.98
        self.loss_type = 'dice_ce'  # Options: 'dice_ce', 'focal', 'tversky'
        self.num_epochs = 200

    def get_tr_and_val_datasets(self):
        """Override to use dataset class with classification labels"""
        tr_keys, val_keys = self.do_split()
        dataset_name = self.plans_manager.dataset_name

        # Infer the appropriate dataset class (numpy or blosc2)
        dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # Use dataset class with classification labels enabled
        dataset_tr = dataset_class(
            self.preprocessed_dataset_folder, tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            load_subtype_labels=True,
            label_path=os.path.join(os.environ['nnUNet_raw'], dataset_name, "labels.csv")
        )
        dataset_val = dataset_class(
            self.preprocessed_dataset_folder, val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            load_subtype_labels=True,
            label_path=os.path.join(os.environ['nnUNet_raw'], dataset_name, "labels.csv")
        )

        return dataset_tr, dataset_val

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build the multi-task network architecture.
        This method follows the nnUNetv2 trainer interface but builds our custom multi-task network.
        """

        # Handle the import requirements for architecture kwargs
        import pydoc
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        # Map architecture class names to our custom classes
        architecture_mapping = {
            'nnunetv2.architectures.MultiTaskResEncUNet.MultiTaskResEncUNet': MultiTaskResEncUNet,
            'nnunetv2.architectures.MultiTaskResEncUNet.MultiTaskChannelAttentionResEncUNet': MultiTaskChannelAttentionResEncUNet,
            'nnunetv2.architectures.MultiTaskResEncUNet.MultiTaskEfficientAttentionResEncUNet': MultiTaskEfficientAttentionResEncUNet,
            # Add fallback for just the class name
            'MultiTaskResEncUNet': MultiTaskResEncUNet,
            'MultiTaskChannelAttentionResEncUNet': MultiTaskChannelAttentionResEncUNet,
            'MultiTaskEfficientAttentionResEncUNet': MultiTaskEfficientAttentionResEncUNet,
        }

        # Get the network class
        if architecture_class_name in architecture_mapping:
            network_class = architecture_mapping[architecture_class_name]
        else:
            # Fallback to default nnUNet behavior
            raise ValueError(f"Unknown architecture_class_name: {architecture_class_name}")

        # Create the network - note the different parameter names for multi-task networks
        network = network_class(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            # num_classification_classes=3,  # Update based on your classification classes
            **architecture_kwargs
        )

        return network

    # def _build_loss(self):
    #     """Override to use multi-task loss"""
    #     return MultiTaskLoss(
    #         seg_weight=self.seg_weight,
    #         cls_weight=self.cls_weight,
    #         loss_type=self.loss_type
    #     )

    # def _build_loss(self):
    #     if self.label_manager.has_regions:
    #         loss = DC_and_BCE_loss({},
    #                                {'batch_dice': self.configuration_manager.batch_dice,
    #                                 'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
    #                                use_ignore_label=self.label_manager.ignore_label is not None,
    #                                dice_class=MemoryEfficientSoftDiceLoss)
    #     else:
    #         loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
    #                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
    #                               ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

    #     if self._do_i_compile():
    #         loss.dc = torch.compile(loss.dc)

    #     # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    #     # this gives higher resolution outputs more weight in the loss

    #     if self.enable_deep_supervision:
    #         deep_supervision_scales = self._get_deep_supervision_scales()
    #         weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
    #         if self.is_ddp and not self._do_i_compile():
    #             # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
    #             # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
    #             # Anywho, the simple fix is to set a very low weight to this.
    #             weights[-1] = 1e-6
    #         else:
    #             weights[-1] = 0

    #         # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
    #         weights = weights / weights.sum()
    #         # now wrap the loss
    #         loss = DeepSupervisionWrapper(loss, weights)

    #     return loss


    def train_step(self, batch: dict) -> dict:
        """
        Custom training step for multi-task learning.
        """
        data = batch['data'].to(self.device)
        target_seg = batch['target'].to(self.device)  # Segmentation targets
        target_cls = batch['class_target'].to(self.device)  # Classification targets

        if DEBUG:
            print(f"[DEBUG] === Starting training step ===")
            print("Input data is blank? ", torch.all(data == 0))


        self.optimizer.zero_grad()

        # Forward pass
        output = self.network(data)


        # Multi-task output: segmentation and classification
        if isinstance(output, dict) and len(output) == 2:
            seg_output, cls_output = (output['segmentation'], output['classification'])
        else:
            # Fallback if network returns only segmentation
            seg_output = output
            cls_output = None
        if DEBUG:
            print("Segmentation output is blank? ", torch.all(seg_output == 0))
            print("Classification output is blank? ", torch.all(cls_output == 0))

        # Calculate loss
        loss_dict = self.loss(seg_output, target_seg, cls_output, target_cls)

        # Update running means
        seg_loss_val = loss_dict['segmentation_loss'].item()
        cls_loss_val = loss_dict['classification_loss'].item()
        self.running_seg_loss = self.running_alpha * self.running_seg_loss + (1 - self.running_alpha) * seg_loss_val
        self.running_cls_loss = self.running_alpha * self.running_cls_loss + (1 - self.running_alpha) * cls_loss_val

        # Normalize losses
        norm_seg_loss = loss_dict['segmentation_loss'] / (self.running_seg_loss + 1e-8)
        norm_cls_loss = loss_dict['classification_loss'] / (self.running_cls_loss + 1e-8)
        total_loss = self.seg_weight * norm_seg_loss + self.cls_weight * norm_cls_loss

        loss_dict['loss'] = total_loss

        if DEBUG:
            print(f"Training step loss: {loss_dict['loss'].item()}")
            print(f"Segmentation loss: {loss_dict['segmentation_loss'].item()}")
            print(f"Classification loss: {loss_dict['classification_loss'].item()}")

        # Backward pass
        loss_dict['loss'].backward()

        if DEBUG:
            # Check gradients on specific parameter groups
            encoder_grads = []
            cls_grads = []
            seg_grads = []

            for name, param in self.network.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()

                    if 'encoder' in name or 'stages' in name:
                        encoder_grads.append(grad_norm)
                    elif 'classification_head' in name:
                        cls_grads.append(grad_norm)
                    elif 'seg_layers' in name or 'decoder' in name:
                        seg_grads.append(grad_norm)

            print(f"Encoder gradient mean: {np.mean(encoder_grads) if encoder_grads else 0:.6f}")
            print(f"Classification gradient mean: {np.mean(cls_grads) if cls_grads else 0:.6f}")
            print(f"Segmentation gradient mean: {np.mean(seg_grads) if seg_grads else 0:.6f}")
            print(f"Total parameters with gradients: {len(encoder_grads) + len(cls_grads) + len(seg_grads)}")

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        return loss_dict

    def validation_step(self, batch: dict) -> dict:
        """
        Custom validation step for multi-task learning.
        Calculates both segmentation and classification metrics.
        """
        if DEBUG:
            print(f"[DEBUG] === Starting validation step ===")
            print(f"[DEBUG] Batch keys: {list(batch.keys())}")
            print(f"[DEBUG] Data shape: {batch['data'].shape}")
            print(f"[DEBUG] Target shape: {batch['target'].shape}")
            if 'class_target' in batch:
                print(f"[DEBUG] Class target shape: {batch['class_target'].shape}")
                print(f"[DEBUG] Class target values: {batch['class_target']}")
            else:
                print(f"[DEBUG] WARNING: No class_target in batch!")

        data = batch['data'].to(self.device)
        target_seg = batch['target'].to(self.device)  # Segmentation targets
        target_cls = batch['class_target'].to(self.device) if 'class_target' in batch else None

        if DEBUG:
            print(f"[DEBUG] Data device: {data.device}")
            print(f"[DEBUG] Target seg device: {target_seg.device}")
            print(f"[DEBUG] Target cls device: {target_cls.device if target_cls is not None else 'None'}")
            print(f"[DEBUG] Network device: {next(self.network.parameters()).device}")

        with torch.no_grad():
            if DEBUG:
                print(f"[DEBUG] Running forward pass...")
                print(f"[DEBUG] Network in training mode: {self.network.training}")

            output = self.network(data)

            if DEBUG:
                print(f"[DEBUG] Network output type: {type(output)}")
                if isinstance(output, dict):
                    print(f"[DEBUG] Output keys: {list(output.keys())}")
                    for key, value in output.items():
                        if isinstance(value, torch.Tensor):
                            print(f"[DEBUG] {key} shape: {value.shape}")
                            print(f"[DEBUG] {key} device: {value.device}")
                            print(f"[DEBUG] {key} dtype: {value.dtype}")
                            print(f"[DEBUG] {key} range: [{value.min():.4f}, {value.max():.4f}]")
                            print(f"[DEBUG] {key} mean: {value.mean():.4f}")

            # Multi-task output: segmentation and classification
            if isinstance(output, dict) and len(output) == 2:
                seg_output, cls_output = (output['segmentation'], output['classification'])
            else:
                seg_output = output
                cls_output = None

            if DEBUG:
                print(f"[DEBUG] Seg output is list: {isinstance(seg_output, list)}")
                if isinstance(seg_output, list):
                    print(f"[DEBUG] Seg output length: {len(seg_output)}")
                    for i, seg in enumerate(seg_output):
                        print(f"[DEBUG] Seg output {i} shape: {seg.shape}")
                print(f"[DEBUG] Cls output shape: {cls_output.shape if cls_output is not None else 'None'}")

            # Calculate loss
            if DEBUG:
                print(f"[DEBUG] Calculating loss...")
                print(f"[DEBUG] Loss function: {type(self.loss)}")

            loss_dict = self.loss(seg_output, target_seg, cls_output, target_cls)

            if DEBUG:
                print(f"[DEBUG] Loss dict keys: {list(loss_dict.keys())}")
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        print(f"[DEBUG] {key}: {value.item():.6f}")
                    else:
                        print(f"[DEBUG] {key}: {value}")

            # === SEGMENTATION METRICS ===
            if DEBUG:
                print(f"[DEBUG] Processing segmentation metrics...")

            # Handle deep supervision if enabled
            if self.enable_deep_supervision:
                seg_output_for_metrics = seg_output[0]  # Use highest resolution
                target_seg_for_metrics = target_seg[0] if isinstance(target_seg, list) else target_seg
            else:
                seg_output_for_metrics = seg_output
                target_seg_for_metrics = target_seg

            if DEBUG:
                print(f"[DEBUG] Seg output for metrics shape: {seg_output_for_metrics.shape}")
                print(f"[DEBUG] Target seg for metrics shape: {target_seg_for_metrics.shape}")
                print(f"[DEBUG] Has regions: {self.label_manager.has_regions}")
                print(f"[DEBUG] Has ignore label: {self.label_manager.has_ignore_label}")

            # Generate segmentation predictions
            axes = [0] + list(range(2, seg_output_for_metrics.ndim))

            if self.label_manager.has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(seg_output_for_metrics) > 0.5).long()
            else:
                # Standard multi-class segmentation
                output_seg = seg_output_for_metrics.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(
                    seg_output_for_metrics.shape,
                    device=seg_output_for_metrics.device,
                    dtype=torch.float32
                )
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg

            if DEBUG:
                print(f"[DEBUG] Predicted segmentation onehot shape: {predicted_segmentation_onehot.shape}")
                print(f"[DEBUG] Predicted segmentation onehot sum: {predicted_segmentation_onehot.sum()}")

            # Handle ignore labels if present
            if self.label_manager.has_ignore_label:
                if not self.label_manager.has_regions:
                    mask = (target_seg_for_metrics != self.label_manager.ignore_label).float()
                    target_seg_for_metrics = target_seg_for_metrics.clone()
                    target_seg_for_metrics[target_seg_for_metrics == self.label_manager.ignore_label] = 0
                else:
                    if target_seg_for_metrics.dtype == torch.bool:
                        mask = ~target_seg_for_metrics[:, -1:]
                    else:
                        mask = 1 - target_seg_for_metrics[:, -1:]
                    target_seg_for_metrics = target_seg_for_metrics[:, :-1]
            else:
                mask = None

            if DEBUG:
                print(f"[DEBUG] Mask shape: {mask.shape if mask is not None else 'None'}")
                print(f"[DEBUG] Final target seg shape: {target_seg_for_metrics.shape}")

            # Calculate TP, FP, FN for segmentation
            tp, fp, fn, _ = get_tp_fp_fn_tn(
                predicted_segmentation_onehot,
                target_seg_for_metrics,
                axes=axes,
                mask=mask
            )

            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()

            if DEBUG:
                print(f"[DEBUG] TP shape: {tp_hard.shape}, values: {tp_hard}")
                print(f"[DEBUG] FP shape: {fp_hard.shape}, values: {fp_hard}")
                print(f"[DEBUG] FN shape: {fn_hard.shape}, values: {fn_hard}")

            # Remove background class for standard segmentation
            if not self.label_manager.has_regions:
                tp_hard = tp_hard[1:]  # Remove background
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]

            if DEBUG:
                print(f"[DEBUG] After removing background - TP: {tp_hard}, FP: {fp_hard}, FN: {fn_hard}")

            # === CLASSIFICATION METRICS ===
            cls_metrics = {}
            if cls_output is not None and target_cls is not None:
                if DEBUG:
                    print(f"[DEBUG] Processing classification metrics...")
                    print(f"[DEBUG] Cls output shape: {cls_output.shape}")
                    print(f"[DEBUG] Target cls shape: {target_cls.shape}")
                    print(f"[DEBUG] Cls output range: [{cls_output.min():.4f}, {cls_output.max():.4f}]")

                # Get predicted classes
                cls_pred = torch.argmax(cls_output, dim=1)  # Shape: [batch_size]
                cls_target = target_cls.long()  # Ensure target is long type

                if DEBUG:
                    print(f"[DEBUG] Cls pred: {cls_pred}")
                    print(f"[DEBUG] Cls target: {cls_target}")

                # Calculate per-class metrics
                num_classes = cls_output.shape[1]
                cls_tp = torch.zeros(num_classes, dtype=torch.long)
                cls_fp = torch.zeros(num_classes, dtype=torch.long)
                cls_fn = torch.zeros(num_classes, dtype=torch.long)
                cls_tn = torch.zeros(num_classes, dtype=torch.long)

                for class_idx in range(num_classes):
                    # Binary classification metrics for each class
                    pred_positive = (cls_pred == class_idx)
                    target_positive = (cls_target == class_idx)

                    cls_tp[class_idx] = (pred_positive & target_positive).sum()
                    cls_fp[class_idx] = (pred_positive & ~target_positive).sum()
                    cls_fn[class_idx] = (~pred_positive & target_positive).sum()
                    cls_tn[class_idx] = (~pred_positive & ~target_positive).sum()

                cls_metrics = {
                    'cls_tp': cls_tp.cpu().numpy(),
                    'cls_fp': cls_fp.cpu().numpy(),
                    'cls_fn': cls_fn.cpu().numpy(),
                    'cls_tn': cls_tn.cpu().numpy(),
                    'cls_correct': (cls_pred == cls_target).sum().cpu().numpy(),
                    'cls_total': cls_target.numel()
                }

                if DEBUG:
                    print(f"[DEBUG] Classification metrics: {cls_metrics}")

            # Combine all metrics
            result = {
                'loss': loss_dict.get('loss', 0.0).detach().cpu().numpy() if isinstance(loss_dict.get('loss', 0.0), torch.Tensor) else loss_dict.get('loss', 0.0),
                'seg_loss': loss_dict.get('segmentation_loss', 0.0).detach().cpu().numpy() if isinstance(loss_dict.get('segmentation_loss', 0.0), torch.Tensor) else loss_dict.get('segmentation_loss', 0.0),
                'cls_loss': loss_dict.get('classification_loss', 0.0).detach().cpu().numpy() if isinstance(loss_dict.get('classification_loss', 0.0), torch.Tensor) else loss_dict.get('classification_loss', 0.0),
                'tp_hard': tp_hard,
                'fp_hard': fp_hard,
                'fn_hard': fn_hard,
                **cls_metrics
            }

            if DEBUG:
                print(f"[DEBUG] Final validation result:")
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        print(f"[DEBUG] {key}: shape={value.shape}, values={value}")
                    else:
                        print(f"[DEBUG] {key}: {value}")
                print(f"[DEBUG] === End validation step ===\n")

            return result

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """
        Custom validation epoch end for multi-task learning.
        Calculates and logs both segmentation and classification metrics.
        """
        if DEBUG:
            print(f"[DEBUG] === Starting validation epoch end ===")
            print(f"[DEBUG] Number of validation outputs: {len(val_outputs)}")
            if val_outputs:
                print(f"[DEBUG] First output keys: {list(val_outputs[0].keys())}")
                print(f"[DEBUG] Sample loss values from first few outputs:")
                for i, output in enumerate(val_outputs[:3]):
                    print(f"[DEBUG] Output {i}: loss={output.get('loss', 'missing')}, seg_loss={output.get('seg_loss', 'missing')}, cls_loss={output.get('cls_loss', 'missing')}")

        outputs_collated = collate_outputs(val_outputs)

        if DEBUG:
            print(f"[DEBUG] Collated outputs keys: {list(outputs_collated.keys())}")
            for key, value in outputs_collated.items():
                if isinstance(value, np.ndarray):
                    print(f"[DEBUG] Collated {key}: shape={value.shape}, mean={np.mean(value):.6f}")
                else:
                    print(f"[DEBUG] Collated {key}: {type(value)}")

        # === SEGMENTATION METRICS ===
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if DEBUG:
            print(f"[DEBUG] Aggregated TP: {tp}")
            print(f"[DEBUG] Aggregated FP: {fp}")
            print(f"[DEBUG] Aggregated FN: {fn}")

        # Handle distributed training for segmentation metrics
        if self.is_ddp:
            if DEBUG:
                print(f"[DEBUG] Handling distributed training...")

            world_size = dist.get_world_size()

            # Gather segmentation metrics
            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            # Gather losses
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            total_loss = np.vstack(losses_val).mean()

            seg_losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(seg_losses_val, outputs_collated['seg_loss'])
            seg_loss = np.vstack(seg_losses_val).mean()

            cls_losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(cls_losses_val, outputs_collated['cls_loss'])
            cls_loss = np.vstack(cls_losses_val).mean()
        else:
            total_loss = np.mean(outputs_collated['loss'])
            seg_loss = np.mean(outputs_collated['seg_loss'])
            cls_loss = np.mean(outputs_collated['cls_loss'])

        if DEBUG:
            print(f"[DEBUG] Final aggregated losses:")
            print(f"[DEBUG] Total loss: {total_loss}")
            print(f"[DEBUG] Seg loss: {seg_loss}")
            print(f"[DEBUG] Cls loss: {cls_loss}")
            print(f"[DEBUG] Total loss is zero: {total_loss == 0.0}")
            print(f"[DEBUG] Seg loss is zero: {seg_loss == 0.0}")
            print(f"[DEBUG] Cls loss is zero: {cls_loss == 0.0}")

        # Calculate segmentation Dice scores
        global_dc_per_class = [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        if DEBUG:
            print(f"[DEBUG] Dice per class: {global_dc_per_class}")
            print(f"[DEBUG] Mean FG Dice: {mean_fg_dice}")

        # === CLASSIFICATION METRICS ===
        cls_metrics_summary = {}
        if 'cls_tp' in outputs_collated and outputs_collated['cls_tp'].size > 0:
            # Aggregate classification metrics
            cls_tp = np.sum(outputs_collated['cls_tp'], 0)
            cls_fp = np.sum(outputs_collated['cls_fp'], 0)
            cls_fn = np.sum(outputs_collated['cls_fn'], 0)
            cls_tn = np.sum(outputs_collated['cls_tn'], 0)
            cls_correct = np.sum(outputs_collated['cls_correct'])
            cls_total = np.sum(outputs_collated['cls_total'])

            # Handle distributed training for classification metrics
            if self.is_ddp:
                # Gather classification metrics
                cls_tps = [None for _ in range(world_size)]
                dist.all_gather_object(cls_tps, cls_tp)
                cls_tp = np.vstack([i[None] for i in cls_tps]).sum(0)

                cls_fps = [None for _ in range(world_size)]
                dist.all_gather_object(cls_fps, cls_fp)
                cls_fp = np.vstack([i[None] for i in cls_fps]).sum(0)

                cls_fns = [None for _ in range(world_size)]
                dist.all_gather_object(cls_fns, cls_fn)
                cls_fn = np.vstack([i[None] for i in cls_fns]).sum(0)

                cls_corrects = [None for _ in range(world_size)]
                dist.all_gather_object(cls_corrects, cls_correct)
                cls_correct = np.sum(cls_corrects)

                cls_totals = [None for _ in range(world_size)]
                dist.all_gather_object(cls_totals, cls_total)
                cls_total = np.sum(cls_totals)

            # Calculate classification metrics
            cls_accuracy = cls_correct / cls_total if cls_total > 0 else 0.0

            # Per-class precision, recall, F1
            cls_precision = np.divide(cls_tp, cls_tp + cls_fp, out=np.zeros_like(cls_tp, dtype=float), where=(cls_tp + cls_fp) != 0)
            cls_recall = np.divide(cls_tp, cls_tp + cls_fn, out=np.zeros_like(cls_tp, dtype=float), where=(cls_tp + cls_fn) != 0)
            cls_f1 = np.divide(2 * cls_precision * cls_recall, cls_precision + cls_recall,
                            out=np.zeros_like(cls_precision), where=(cls_precision + cls_recall) != 0)

            # Macro averages
            macro_precision = np.nanmean(cls_precision)
            macro_recall = np.nanmean(cls_recall)
            macro_f1 = np.nanmean(cls_f1)

            cls_metrics_summary = {
                'cls_accuracy': cls_accuracy,
                'cls_precision_per_class': cls_precision,
                'cls_recall_per_class': cls_recall,
                'cls_f1_per_class': cls_f1,
                'cls_macro_precision': macro_precision,
                'cls_macro_recall': macro_recall,
                'cls_macro_f1': macro_f1
            }

        # === LOGGING ===
        if DEBUG:
            print(f"[DEBUG] Logging metrics...")

        # Log segmentation metrics
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', total_loss, self.current_epoch)

        # Log losses
        self.logger.log('val_total_loss', total_loss, self.current_epoch)
        self.logger.log('val_seg_loss', seg_loss, self.current_epoch)
        self.logger.log('val_cls_loss', cls_loss, self.current_epoch)

        # Log classification metrics
        if cls_metrics_summary:
            self.logger.log('cls_accuracy', cls_metrics_summary['cls_accuracy'], self.current_epoch)
            self.logger.log('cls_macro_f1', cls_metrics_summary['cls_macro_f1'], self.current_epoch)
            self.logger.log('cls_macro_precision', cls_metrics_summary['cls_macro_precision'], self.current_epoch)
            self.logger.log('cls_macro_recall', cls_metrics_summary['cls_macro_recall'], self.current_epoch)
            self.logger.log('cls_f1_per_class', cls_metrics_summary['cls_f1_per_class'], self.current_epoch)
            self.logger.log('cls_precision_per_class', cls_metrics_summary['cls_precision_per_class'], self.current_epoch)
            self.logger.log('cls_recall_per_class', cls_metrics_summary['cls_recall_per_class'], self.current_epoch)

        # === CONSOLE OUTPUT ===
        print(f"\n=== Validation Epoch {self.current_epoch} Results ===")
        print(f"Total Loss: {total_loss:.4f}")
        print(f"Segmentation Loss: {seg_loss:.4f}")
        print(f"Classification Loss: {cls_loss:.4f}")
        print(f"Mean Foreground Dice: {mean_fg_dice:.4f}")

        if len(global_dc_per_class) >= 2:
            print(f"Pancreas Dice: {global_dc_per_class[0]:.4f}")
            print(f"Lesion Dice: {global_dc_per_class[1]:.4f}")

        if cls_metrics_summary:
            print(f"Classification Accuracy: {cls_metrics_summary['cls_accuracy']:.4f}")
            print(f"Classification Macro F1: {cls_metrics_summary['cls_macro_f1']:.4f}")
            print(f"Classification Macro Precision: {cls_metrics_summary['cls_macro_precision']:.4f}")
            print(f"Classification Macro Recall: {cls_metrics_summary['cls_macro_recall']:.4f}")

        print("=" * 50)

        return {
            'mean_fg_dice': mean_fg_dice,
            'dice_per_class': global_dc_per_class,
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'cls_loss': cls_loss,
            **cls_metrics_summary
        }


    def on_train_epoch_start(self):
        """
        Hook called at the start of each training epoch.
        Can be used for custom logic like dynamic loss weighting.
        """
        super().on_train_epoch_start()

        # if hasattr(self, 'current_epoch'):
        #     epoch_ratio = min(self.current_epoch / self.num_epochs, 1.0)
        #     self.loss.cls_weight = self.cls_weight * epoch_ratio

    def run_training(self):
        """
        Override the main training loop if needed for multi-task specific logic.
        """
        # You can add custom training logic here
        # For now, use the parent implementation
        super().run_training()
