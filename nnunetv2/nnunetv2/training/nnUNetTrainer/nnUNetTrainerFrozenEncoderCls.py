import torch
from torch import nn
import numpy as np
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask
from nnunetv2.architectures.ResEncUnetWithCls import ResEncUnetWithCls
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import PolynomialLR
import pydoc
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

class nnUNetTrainerFrozenEncoderCls(nnUNetTrainerMultiTask):
    """
    Trainer for classification head only: uses pretrained ResEnc U-Net backbone, frozen, and trains only cls head.
    Reports segmentation metrics but uses crossentropy loss for classification training.
    Requires 'pretrained_checkpoint' in plans dict.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.batch_size = 32

    def initialize(self):
        """
        Custom initialize: skip default network build and instantiate frozen backbone + cls head model
        """
        if self.was_initialized:
            raise RuntimeError("Trainer already initialized")
        # adjust batch size (DDP)
        self._set_batch_size_and_oversample()
        # determine input channels
        self.num_input_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json)
        # load pretrained checkpoint path
        ckpt = self.plans_manager.plans.get('pretrained_checkpoint')
        if ckpt is None:
            raise RuntimeError("Please set 'pretrained_checkpoint' in plans to fold0 checkpoint.")
        # instantiate model
        unet_kwargs = self.configuration_manager.network_arch_init_kwargs
        # Convert string references to actual classes (like nnU-Net does)
        processed_kwargs = dict(**unet_kwargs)
        for ri in self.configuration_manager.network_arch_init_kwargs_req_import:
            if processed_kwargs[ri] is not None:
                processed_kwargs[ri] = pydoc.locate(processed_kwargs[ri])

        model = ResEncUnetWithCls(
            pretrained_checkpoint=ckpt,
            input_channels=self.num_input_channels,
            num_classes=self.label_manager.num_segmentation_heads,
            num_cls_classes=self.num_classification_classes,
            **processed_kwargs)
        self.network = model.to(self.device)
        # optimizer & scheduler (only cls head params)
        cls_params = list(self.network.cls_head.parameters())
        # train only classification head
        self.optimizer = SGD(
            cls_params,
            lr=self.initial_lr * 0.1,
            weight_decay=self.weight_decay,
            momentum=0.9,
            nesterov=True
        )
        self.lr_scheduler = PolynomialLR(
            optimizer=self.optimizer,
            total_iters=self.num_epochs,
            power=0.9
        )
        # segmentation loss for reporting
        self.loss = self._build_loss()
        if isinstance(self.loss, nn.Module):
            self.loss = self.loss.to(self.device)
        # dataset class for dataloaders
        self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        self.was_initialized = True

    def configure_optimizers(self):
        # train only classification head
        cls_params = list(self.network.cls_head.parameters())
        optimizer = SGD(cls_params, self.initial_lr * 0.1,
                                    weight_decay=self.weight_decay, momentum=0.9, nesterov=True)
        scheduler = PolynomialLR(optimizer, self.num_epochs, power=0.9)
        return optimizer, scheduler

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device)
        target_seg = batch['target'].to(self.device)
        target_cls = batch['class_target'].to(self.device)

        self.optimizer.zero_grad()
        output = self.network(data)

        # segmentation loss for reporting only
        seg_loss = self.loss(output['segmentation'], target_seg)  # original seg loss
        # classification loss for training
        cls_loss = self.cls_criterion(output['classification'], target_cls)

        cls_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.cls_head.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Compute training classification accuracy
        with torch.no_grad():
            cls_pred = torch.argmax(output['classification'], dim=1)
            cls_accuracy = (cls_pred == target_cls).float().mean().item()

        return {
            'loss': cls_loss,
            'seg_loss': seg_loss.item(),
            'cls_loss': cls_loss.item(),
            'cls_accuracy': cls_accuracy
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device)
        target_seg = batch['target'].to(self.device)
        target_cls = batch['class_target'].to(self.device)

        with torch.no_grad():
            output = self.network(data)

            # segmentation loss for reporting only
            seg_loss = self.loss(output['segmentation'], target_seg)
            # classification loss for metrics
            cls_loss = self.cls_criterion(output['classification'], target_cls)

            # Compute classification metrics
            cls_pred = torch.argmax(output['classification'], dim=1)
            cls_true = target_cls

            # Convert to numpy for metrics calculation
            cls_pred_np = cls_pred.detach().cpu().numpy()
            cls_true_np = cls_true.detach().cpu().numpy()

            # Compute segmentation metrics (required by parent class)
            from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
            seg_output = output['segmentation']
            if isinstance(seg_output, list):
                seg_output = seg_output[0]
            if isinstance(target_seg, list):
                target_seg = target_seg[0]

            axes = [0] + list(range(2, seg_output.ndim))

            if self.label_manager.has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).long()
            else:
                output_seg = seg_output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(seg_output.shape, device=seg_output.device, dtype=torch.float32)
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)

            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_seg, axes=axes, mask=None)
            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()

            if not self.label_manager.has_regions:
                tp_hard = tp_hard[1:]  # Remove background
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]

        return {
            'loss': cls_loss.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
            'seg_loss': seg_loss.item(),
            'cls_loss': cls_loss.item(),
            'cls_pred': cls_pred_np,
            'cls_true': cls_true_np
        }

    def on_validation_epoch_end(self, val_outputs):
        """Override to compute detailed classification metrics while still computing segmentation metrics for compatibility"""
        # Let parent handle segmentation metrics computation
        super().on_validation_epoch_end(val_outputs)

        # Compute detailed classification metrics
        outputs_collated = collate_outputs(val_outputs)
        avg_cls_loss = np.mean(outputs_collated['cls_loss'])

        # Aggregate all predictions and true labels
        all_cls_pred = np.concatenate(outputs_collated['cls_pred'])
        all_cls_true = np.concatenate(outputs_collated['cls_true'])

        # Compute classification metrics
        cls_f1_macro = f1_score(all_cls_true, all_cls_pred, average='macro', zero_division=0)
        cls_f1_micro = f1_score(all_cls_true, all_cls_pred, average='micro', zero_division=0)
        cls_precision = precision_score(all_cls_true, all_cls_pred, average='macro', zero_division=0)
        cls_recall = recall_score(all_cls_true, all_cls_pred, average='macro', zero_division=0)
        cls_accuracy = np.mean(all_cls_pred == all_cls_true)

        # Log detailed classification metrics
        self.print_to_log_file(f"Validation classification loss: {avg_cls_loss:.4f}")
        self.print_to_log_file(f"Classification accuracy: {cls_accuracy:.4f}")
        self.print_to_log_file(f"Classification F1 (macro): {cls_f1_macro:.4f}")
        self.print_to_log_file(f"Classification F1 (micro): {cls_f1_micro:.4f}")
        self.print_to_log_file(f"Classification precision (macro): {cls_precision:.4f}")
        self.print_to_log_file(f"Classification recall (macro): {cls_recall:.4f}")

        # Log to the trainer's logger for plotting
        self.logger.log('val_cls_loss', avg_cls_loss, self.current_epoch)
        self.logger.log('val_cls_accuracy', cls_accuracy, self.current_epoch)
        self.logger.log('val_cls_f1_macro', cls_f1_macro, self.current_epoch)
        self.logger.log('val_cls_f1_micro', cls_f1_micro, self.current_epoch)
        self.logger.log('val_cls_precision', cls_precision, self.current_epoch)
        self.logger.log('val_cls_recall', cls_recall, self.current_epoch)

        # Print detailed classification report every 10 epochs
        if self.current_epoch % 10 == 0:
            self.print_to_log_file("Detailed classification report:")
            self.print_to_log_file(classification_report(all_cls_true, all_cls_pred, zero_division=0))

    def on_train_epoch_start(self):
        """Override to fix scheduler warning"""
        self.network.train()
        self.lr_scheduler.step()  # Remove epoch parameter to fix warning
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def on_train_epoch_end(self, train_outputs):
        # Use base trainer's logging for basic metrics
        super().on_train_epoch_end(train_outputs)

        # Log additional classification training metrics
        outputs_collated = collate_outputs(train_outputs)
        avg_cls_accuracy = np.mean(outputs_collated['cls_accuracy'])
        avg_cls_loss = np.mean(outputs_collated['cls_loss'])

        self.print_to_log_file(f"Training classification loss: {avg_cls_loss:.4f}")
        self.print_to_log_file(f"Training classification accuracy: {avg_cls_accuracy:.4f}")

        # Log to the trainer's logger for plotting
        self.logger.log('train_cls_loss', avg_cls_loss, self.current_epoch)
        self.logger.log('train_cls_accuracy', avg_cls_accuracy, self.current_epoch)
