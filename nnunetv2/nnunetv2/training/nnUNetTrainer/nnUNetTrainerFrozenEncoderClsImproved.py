#!/usr/bin/env python3
"""
Improved nnUNet Trainer for Frozen Encoder Classification
Fixes multiple issues identified in the debugging analysis.
"""
import torch
from torch import nn
import numpy as np
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask
from nnunetv2.architectures.ResEncUnetWithClsImproved import ResEncUnetWithClsImproved
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import PolynomialLR
import pydoc
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import warnings

# def debug_features(features, epoch):
#     """Debug feature statistics"""
#     feat_mean = features.mean()
#     feat_std = features.std()
#     feat_max = features.max()
#     feat_min = features.min()

#     print(f"Features - Mean: {feat_mean:.4f}, Std: {feat_std:.4f}")
#     print(f"Features - Range: [{feat_min:.4f}, {feat_max:.4f}]")

#     if feat_std < 0.01:
#         print("⚠️ WARNING: Very low feature variance!")

class nnUNetTrainerFrozenEncoderClsImproved(nnUNetTrainerMultiTask):
    """
    IMPROVED trainer for classification head only with multiple fixes:
    1. Higher learning rate and better optimization
    2. Class-weighted loss for imbalanced data
    3. Better initialization of classification head
    4. Fallback classification target extraction
    5. Gradient accumulation for better training
    6. Configurable encoder fine-tuning strategies
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Use class-weighted CrossEntropy for imbalanced data
        self.cls_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 0.8, 1.2]).to(device))  # Adjust based on class frequency
        # self.batch_size = 32
        self.gradient_accumulation_steps = 4  # Effective batch size = 32 * 4 = 128
        self.enc_lr = 1e-4
        self.cls_lr = 1e-3

        # Configurable fine-tuning strategy
        # Options: 'partial' (stages 4,5), 'full' (all encoder), 'minimal' (cls only)
        self.fine_tuning_strategy = getattr(plans, 'fine_tuning_strategy', 'full')
        self.unfreeze_stages = getattr(plans, 'unfreeze_stages', [4, 5])  # For partial strategy
        self.auto_adjust_strategy = getattr(plans, 'auto_adjust_strategy', False)  # Enable auto-adjustment
        self.classification_mode = getattr(plans, 'classification_mode', "spatial_attention_multiscale")

    def initialize(self):
        """Custom initialize with improvements"""
        if self.was_initialized:
            raise RuntimeError("Trainer already initialized")

        self._set_batch_size_and_oversample()
        self.num_input_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json)

        ckpt = self.plans_manager.plans.get('pretrained_checkpoint')
        if ckpt is None:
            raise RuntimeError("Please set 'pretrained_checkpoint' in plans to fold0 checkpoint.")

        unet_kwargs = self.configuration_manager.network_arch_init_kwargs
        processed_kwargs = dict(**unet_kwargs)
        for ri in self.configuration_manager.network_arch_init_kwargs_req_import:
            if processed_kwargs[ri] is not None:
                processed_kwargs[ri] = pydoc.locate(processed_kwargs[ri])

        model = ResEncUnetWithClsImproved(
            pretrained_checkpoint=ckpt,
            input_channels=self.num_input_channels,
            num_classes=self.label_manager.num_segmentation_heads,
            num_cls_classes=self.num_classification_classes,
            **processed_kwargs)

        # IMPROVED: Better initialization of classification head
        self._initialize_classification_head(model.cls_head)

        self.network = model.to(self.device)

        # IMPROVED: Configurable fine-tuning strategy from benchmark
        param_groups = self._setup_fine_tuning_strategy(model)

        self.optimizer, self.lr_scheduler = self._configure_optimizer_from_plans(param_groups)

        # Log detailed parameter information (from benchmark)
        # param_info = self.get_trainable_params_info()
        # print(f"\n📊 Model Parameter Summary:")
        # print(f"  Strategy: {self.fine_tuning_strategy}")
        # print(f"  Total parameters: {param_info['total_params']:,}")
        # print(f"  Trainable parameters: {param_info['total_trainable']:,} ({param_info['trainable_percentage']:.2f}%)")
        # print(f"  Encoder: {param_info['encoder_trainable']:,} / {param_info['encoder_params']:,}")
        # print(f"  Decoder: {param_info['decoder_trainable']:,} / {param_info['decoder_params']:,}")
        # print(f"  Classification head: {param_info['cls_trainable']:,} / {param_info['cls_params']:,}")
        # print("🎯 Ready for training!\n")

        # segmentation loss for reporting
        self.loss = self._build_loss()
        if isinstance(self.loss, nn.Module):
            self.loss = self.loss.to(self.device)

        self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        self.was_initialized = True

        # Log complete configuration for reproducibility
        self.log_training_config()

    def _set_batch_size_and_oversample(self):
        """Set batch size and oversampling strategy based on configuration"""
        if 'batch_size' in self.plans_manager.plans and self.plans_manager.plans['batch_size'] > 0:
            self.batch_size = self.plans_manager.plans['batch_size']
        else:
            # Default to 32 if not specified
            self.batch_size = 2

        # Oversampling strategy
        self.oversample_foreground = self.plans_manager.plans.get('oversample_foreground', False)
        self.undersample_background = self.plans_manager.plans.get('undersample_background', False)

        if self.oversample_foreground and self.undersample_background:
            raise ValueError("Cannot both oversample foreground and undersample background")

        print(f"Batch size set to {self.batch_size}")
        if self.oversample_foreground:
            print("Foreground oversampling enabled")
        if self.undersample_background:
            print("Background undersampling enabled")

    def _setup_fine_tuning_strategy(self, model):
        """
        Setup fine-tuning strategy based on benchmark approaches.
        Returns parameter groups for optimizer.
        """
        print(f"Setting up fine-tuning strategy: {self.fine_tuning_strategy}")

        # Start with everything frozen
        for param in model.parameters():
            param.requires_grad = False

        # Always unfreeze classification head
        for param in model.cls_head.parameters():
            param.requires_grad = True
        cls_params = list(model.cls_head.parameters())

        param_groups = []

        if self.fine_tuning_strategy == 'minimal':
            # Only train classification head (baseline)
            print("  - Training only classification head")
            param_groups = [
                {'params': cls_params, 'lr': self.cls_lr, 'name': 'cls_head_only'}
            ]

        elif self.fine_tuning_strategy == 'partial':
            # Partial encoder fine-tuning (benchmark strategy)
            print(f"  - Partial encoder fine-tuning: stages {self.unfreeze_stages}")

            # Ensure decoder stays frozen
            for param in model.unet.decoder.parameters():
                param.requires_grad = False

            # Unfreeze specified encoder stages
            unfrozen_params = []
            for stage_idx in self.unfreeze_stages:
                if stage_idx < len(model.unet.encoder.stages):
                    stage = model.unet.encoder.stages[stage_idx]
                    for param in stage.parameters():
                        param.requires_grad = True
                    unfrozen_params.extend(list(stage.parameters()))
                    print(f"    - Unfrozen encoder stage {stage_idx}")
                else:
                    print(f"    - Warning: Stage {stage_idx} does not exist")

            enc_params = [p for p in unfrozen_params if p.requires_grad]
            param_groups = [
                {'params': enc_params, 'lr': self.enc_lr, 'name': f'encoder_stages_{self.unfreeze_stages}'},
                {'params': cls_params, 'lr': self.cls_lr, 'name': 'cls_head'}
            ]

        elif self.fine_tuning_strategy == 'full':
            # Full encoder fine-tuning (benchmark strategy)
            print("  - Full encoder fine-tuning")

            # Freeze only decoder-specific parameters (not embedded encoder parts)
            for name, param in model.unet.decoder.named_parameters():
                if not name.startswith('encoder.'):
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            # Unfreeze entire encoder
            for param in model.unet.encoder.parameters():
                param.requires_grad = True

            enc_params = list(model.unet.encoder.parameters())
            param_groups = [
                {'params': enc_params, 'lr': self.enc_lr, 'name': 'full_encoder'},
                {'params': cls_params, 'lr': self.cls_lr, 'name': 'cls_head'}
            ]

        else:
            raise ValueError(f"Unknown fine-tuning strategy: {self.fine_tuning_strategy}")

        # Filter out empty parameter groups
        param_groups = [group for group in param_groups if len(list(group['params'])) > 0]

        # Log parameter counts (from benchmark)
        for group in param_groups:
            group_params = sum(p.numel() for p in group['params'] if p.requires_grad)
            print(f"    - {group['name']}: {group_params:,} parameters")

        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  - Total trainable: {total_trainable:,} / {total_params:,} ({100*total_trainable/total_params:.2f}%)")

        return param_groups

    def get_trainable_params_info(self):
        """Get detailed information about trainable parameters (from benchmark)"""
        total_params = sum(p.numel() for p in self.network.parameters())
        total_trainable = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        # Breakdown by component
        encoder_params = sum(p.numel() for p in self.network.unet.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.network.unet.encoder.parameters() if p.requires_grad)

        decoder_params = sum(p.numel() for p in self.network.unet.decoder.parameters())
        decoder_trainable = sum(p.numel() for p in self.network.unet.decoder.parameters() if p.requires_grad)

        cls_params = sum(p.numel() for p in self.network.cls_head.parameters())
        cls_trainable = sum(p.numel() for p in self.network.cls_head.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'total_trainable': total_trainable,
            'trainable_percentage': 100 * total_trainable / total_params,
            'encoder_params': encoder_params,
            'encoder_trainable': encoder_trainable,
            'decoder_params': decoder_params,
            'decoder_trainable': decoder_trainable,
            'cls_params': cls_params,
            'cls_trainable': cls_trainable,
        }

    def switch_fine_tuning_strategy(self, new_strategy, new_unfreeze_stages=None):
        """
        Switch fine-tuning strategy during training (useful for curriculum learning)
        """
        print(f"Switching fine-tuning strategy from {self.fine_tuning_strategy} to {new_strategy}")

        old_strategy = self.fine_tuning_strategy
        self.fine_tuning_strategy = new_strategy

        if new_unfreeze_stages is not None:
            self.unfreeze_stages = new_unfreeze_stages

        # Setup new parameter groups
        param_groups = self._setup_fine_tuning_strategy(self.network)

        # Create new optimizer with new parameter groups
        # Use special handling for strategy switching - adjust remaining epochs
        remaining_epochs = self.num_epochs - self.current_epoch

        # Temporarily store original num_epochs for scheduler configuration
        original_num_epochs = self.num_epochs
        self.num_epochs = remaining_epochs

        self.optimizer, self.lr_scheduler = self._configure_optimizer_from_plans(param_groups)

        # Restore original num_epochs
        self.num_epochs = original_num_epochs

        print(f"Successfully switched from {old_strategy} to {new_strategy}")

        # Log new parameter info
        info = self.get_trainable_params_info()
        self.print_to_log_file(f"Strategy switch - New trainable params: {info['total_trainable']:,} ({info['trainable_percentage']:.2f}%)")

    def _initialize_classification_head(self, cls_head):
        """Better initialization for classification head"""
        for module in cls_head.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU networks
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # Small positive bias

    def _extract_classification_targets(self, batch):
        """
        Extract classification targets from batch with multiple fallback strategies
        """
        batch_size = batch['data'].shape[0]

        # Strategy 1: Try to get from batch directly
        if 'class_target' in batch and batch['class_target'] is not None:
            cls_targets = batch['class_target']
            if not isinstance(cls_targets, torch.Tensor):
                cls_targets = torch.tensor(cls_targets, dtype=torch.long, device=self.device)
            return cls_targets.to(self.device)

        # Strategy 2: Extract from keys/filenames
        if 'keys' in batch:
            cls_targets = []
            for key in batch['keys']:
                # Extract classification based on filename patterns
                if 'subtype0' in key or any(x in key for x in ['type_0', 'class_0', '_0_']):
                    cls_targets.append(0)
                elif 'subtype1' in key or any(x in key for x in ['type_1', 'class_1', '_1_']):
                    cls_targets.append(1)
                elif 'subtype2' in key or any(x in key for x in ['type_2', 'class_2', '_2_']):
                    cls_targets.append(2)
                else:
                    # Fallback: try to extract number from filename
                    import re
                    numbers = re.findall(r'(\d+)', key)
                    if numbers:
                        cls_targets.append(int(numbers[0]) % 3)  # Mod 3 to ensure valid class
                    else:
                        cls_targets.append(0)  # Default class

            return torch.tensor(cls_targets, dtype=torch.long, device=self.device)

        # Strategy 3: Random targets as last resort (for debugging)
        warnings.warn("Could not extract classification targets, using random targets")
        return torch.randint(0, 3, (batch_size,), device=self.device)

    def debug_logits(self, logits, labels, phase='train'):
        """Debug classification logits to catch collapse early"""
        if self.current_epoch % 5 == 0:  # Every 5 epochs
            logits_mean = logits.mean(dim=0)
            logits_std = logits.std(dim=0)
            probs = torch.softmax(logits, dim=1).mean(dim=0)

            self.print_to_log_file(f"{phase} Logits mean: {logits_mean.cpu().numpy()}")
            self.print_to_log_file(f"{phase} Logits std: {logits_std.cpu().numpy()}")
            self.print_to_log_file(f"{phase} Avg probs: {probs.cpu().numpy()}")

            # Check for collapse
            if logits_std.max() < 0.1:
                self.print_to_log_file("⚠️ WARNING: Logits collapse detected!")

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device)
        target_seg = batch['target'].to(self.device)
        target_cls = self._extract_classification_targets(batch)

        # Gradient accumulation for larger effective batch size
        accumulation_step = hasattr(self, '_accumulation_step')
        if not accumulation_step:
            self._accumulation_step = 0
            self.optimizer.zero_grad()

        output = self.network(data)

        # segmentation loss for reporting only
        seg_loss = self.loss(output['segmentation'], target_seg)
        # classification loss for training with class weighting
        cls_loss = self.cls_criterion(output['classification'], target_cls)

        # Total loss with uncertainty weighting as a backup
        total_loss = cls_loss + seg_loss



        # Scale loss by accumulation steps
        scaled_cls_loss = cls_loss / self.gradient_accumulation_steps
        scaled_cls_loss.backward()

        self._accumulation_step += 1

        # Update weights every gradient_accumulation_steps
        if self._accumulation_step >= self.gradient_accumulation_steps:
            # IMPROVED: Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.cls_head.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._accumulation_step = 0

        # Compute training classification accuracy
        with torch.no_grad():
            cls_pred = torch.argmax(output['classification'], dim=1)
            cls_accuracy = (cls_pred == target_cls).float().mean().item()

        # IMPROVED: Add segmentation monitoring like benchmark (but don't train on it)
        seg_loss_value = seg_loss.item() if torch.is_tensor(seg_loss) else seg_loss

        if self.current_epoch % 5 == 0:
            self.debug_logits(output['classification'], target_cls, 'train')

        return {
            'loss': cls_loss,  # Return unscaled loss for logging
            'seg_loss': seg_loss_value,
            'cls_loss': cls_loss.item(),
            'cls_accuracy': cls_accuracy,
            'seg_preserved': seg_loss_value < 10.0  # Flag for monitoring seg preservation
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device)
        target_seg = batch['target'].to(self.device)
        target_cls = self._extract_classification_targets(batch)

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

        if self.current_epoch % 5 == 0:
            self.debug_logits(output['classification'], target_cls, 'val')

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
        """Override to compute detailed classification metrics"""
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

        # IMPROVED: More detailed logging
        unique_true, counts_true = np.unique(all_cls_true, return_counts=True)
        unique_pred, counts_pred = np.unique(all_cls_pred, return_counts=True)

        self.print_to_log_file(f"Validation classification loss: {avg_cls_loss:.4f}")
        self.print_to_log_file(f"Classification accuracy: {cls_accuracy:.4f}")
        self.print_to_log_file(f"Classification F1 (macro): {cls_f1_macro:.4f}")
        self.print_to_log_file(f"Classification F1 (micro): {cls_f1_micro:.4f}")
        self.print_to_log_file(f"Classification precision (macro): {cls_precision:.4f}")
        self.print_to_log_file(f"Classification recall (macro): {cls_recall:.4f}")
        self.print_to_log_file(f"True class distribution: {dict(zip(unique_true, counts_true))}")
        self.print_to_log_file(f"Predicted class distribution: {dict(zip(unique_pred, counts_pred))}")

        # IMPROVED: Add segmentation preservation monitoring (from benchmark)
        avg_seg_loss = np.mean(outputs_collated['seg_loss'])
        self.print_to_log_file(f"Segmentation loss (monitoring): {avg_seg_loss:.4f}")

        if hasattr(self, '_initial_val_seg_loss'):
            seg_degradation = avg_seg_loss - self._initial_val_seg_loss
            if seg_degradation > 0.05:  # 5% degradation threshold for validation
                self.print_to_log_file(f"⚠️  Validation segmentation degrading: Δ{seg_degradation:+.4f}")
            elif abs(seg_degradation) < 0.01:
                self.print_to_log_file(f"✅ Segmentation performance preserved")
        else:
            self._initial_val_seg_loss = avg_seg_loss

        # Log to the trainer's logger for plotting
        self.logger.log('val_cls_loss', avg_cls_loss, self.current_epoch)
        self.logger.log('val_cls_accuracy', cls_accuracy, self.current_epoch)
        self.logger.log('val_cls_f1_macro', cls_f1_macro, self.current_epoch)
        self.logger.log('val_cls_f1_micro', cls_f1_micro, self.current_epoch)
        self.logger.log('val_cls_precision', cls_precision, self.current_epoch)
        self.logger.log('val_cls_recall', cls_recall, self.current_epoch)
        # self.logger.log('val_seg_loss_monitor', avg_seg_loss, self.current_epoch)  # For preservation tracking

        # IMPROVED: Track performance for auto-adjustment
        self.track_performance_for_auto_adjustment(cls_accuracy, avg_seg_loss)

        # Auto-adjust strategy if enabled and needed
        if getattr(self, 'auto_adjust_strategy', False):
            self.auto_adjust_strategy_if_needed()

        # Print detailed classification report every 5 epochs
        if self.current_epoch % 5 == 0:
            self.print_to_log_file("Detailed classification report:")
            self.print_to_log_file(classification_report(all_cls_true, all_cls_pred, zero_division=0))

    def on_train_epoch_start(self):
        """Override to handle gradient accumulation properly"""
        self.network.train()
        self.lr_scheduler.step()
        self._accumulation_step = 0  # Reset accumulation counter
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def on_train_epoch_end(self, train_outputs):
        # Handle any remaining gradients from accumulation
        if hasattr(self, '_accumulation_step') and self._accumulation_step > 0:
            torch.nn.utils.clip_grad_norm_(self.network.cls_head.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._accumulation_step = 0

        # Use base trainer's logging for basic metrics
        super().on_train_epoch_end(train_outputs)

        # Log additional classification training metrics
        outputs_collated = collate_outputs(train_outputs)
        avg_cls_accuracy = np.mean(outputs_collated['cls_accuracy'])
        avg_cls_loss = np.mean(outputs_collated['cls_loss'])
        avg_seg_loss = np.mean(outputs_collated['seg_loss'])  # For monitoring preservation

        self.print_to_log_file(f"Training classification loss: {avg_cls_loss:.4f}")
        self.print_to_log_file(f"Training classification accuracy: {avg_cls_accuracy:.4f}")
        self.print_to_log_file(f"Training segmentation loss (monitoring): {avg_seg_loss:.4f}")

        # Log to the trainer's logger for plotting
        self.logger.log('train_cls_loss', avg_cls_loss, self.current_epoch)
        self.logger.log('train_cls_accuracy', avg_cls_accuracy, self.current_epoch)
        # self.logger.log('train_seg_loss_monitor', avg_seg_loss, self.current_epoch)  # For preservation tracking

        # IMPROVED: Check for segmentation preservation (from benchmark concept)
        if hasattr(self, '_initial_seg_loss'):
            seg_degradation = avg_seg_loss - self._initial_seg_loss
            if seg_degradation > 0.1:  # 10% degradation threshold
                self.print_to_log_file(f"⚠️  Segmentation performance degrading: Δ{seg_degradation:+.4f}")
        else:
            self._initial_seg_loss = avg_seg_loss  # Store initial for comparison

    def get_training_config_summary(self):
        """
        Get a comprehensive summary of the current training configuration
        Useful for logging and reproducibility
        """
        param_info = self.get_trainable_params_info()

        config = {
            'fine_tuning_strategy': self.fine_tuning_strategy,
            'unfreeze_stages': self.unfreeze_stages if hasattr(self, 'unfreeze_stages') else None,
            'learning_rates': {
                'encoder_lr': self.enc_lr,
                'cls_lr': self.cls_lr
            },
            'batch_config': {
                'batch_size': self.batch_size,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'effective_batch_size': self.batch_size * self.gradient_accumulation_steps
            },
            'parameter_counts': param_info,
            'loss_config': {
                'cls_criterion': str(self.cls_criterion),
                'seg_loss': str(type(self.loss).__name__) if hasattr(self, 'loss') else 'Unknown'
            },
            'optimizer': str(type(self.optimizer).__name__) if hasattr(self, 'optimizer') else 'Not initialized',
            'scheduler': str(type(self.lr_scheduler).__name__) if hasattr(self, 'lr_scheduler') else 'Not initialized',
            'plans_config': {
                'optimizer_class': self.plans_manager.plans.get('optimizer', {}).get('class', 'Default'),
                'scheduler_class': self.plans_manager.plans.get('scheduler', {}).get('class', 'Default'),
                'optimizer_kwargs': self.plans_manager.plans.get('optimizer', {}).get('kwargs', {}),
                'scheduler_kwargs': self.plans_manager.plans.get('scheduler', {}).get('kwargs', {})
            }
        }

        return config

    def log_training_config(self):
        """Log the complete training configuration"""
        config = self.get_training_config_summary()

        self.print_to_log_file("\n" + "="*80)
        self.print_to_log_file("🔧 TRAINING CONFIGURATION SUMMARY")
        self.print_to_log_file("="*80)

        self.print_to_log_file(f"Strategy: {config['fine_tuning_strategy']}")
        if config['unfreeze_stages']:
            self.print_to_log_file(f"Unfreeze stages: {config['unfreeze_stages']}")

        self.print_to_log_file(f"Learning rates: Encoder={config['learning_rates']['encoder_lr']}, Cls={config['learning_rates']['cls_lr']}")
        self.print_to_log_file(f"Batch config: {config['batch_config']['batch_size']} x {config['batch_config']['gradient_accumulation_steps']} = {config['batch_config']['effective_batch_size']} effective")

        param_info = config['parameter_counts']
        self.print_to_log_file(f"Parameters: {param_info['total_trainable']:,}/{param_info['total_params']:,} ({param_info['trainable_percentage']:.2f}%) trainable")

        self.print_to_log_file(f"Optimizer: {config['optimizer']}")
        self.print_to_log_file(f"Scheduler: {config['scheduler']}")
        self.print_to_log_file("="*80 + "\n")

    def auto_adjust_strategy_if_needed(self):
        """
        Automatically adjust fine-tuning strategy if classification isn't improving
        Inspired by benchmark comparison logic
        """
        if not hasattr(self, '_performance_history'):
            self._performance_history = []
            return

        # Get recent performance
        if len(self._performance_history) < 5:
            return  # Need at least 5 epochs of history

        recent_accuracy = [epoch['cls_accuracy'] for epoch in self._performance_history[-5:]]
        improvement = recent_accuracy[-1] - recent_accuracy[0]

        # If no improvement in last 5 epochs and using minimal strategy, upgrade
        if improvement < 0.01 and self.fine_tuning_strategy == 'minimal':
            self.print_to_log_file("🔄 Auto-adjusting: Switching from minimal to partial fine-tuning")
            self.switch_fine_tuning_strategy('partial', [4, 5])

        # If still no improvement with partial, try full (but be conservative)
        elif improvement < 0.005 and self.fine_tuning_strategy == 'partial' and self.current_epoch > 10:
            self.print_to_log_file("🔄 Auto-adjusting: Switching from partial to full fine-tuning")
            self.switch_fine_tuning_strategy('full')

    def track_performance_for_auto_adjustment(self, cls_accuracy, seg_loss):
        """Track performance metrics for auto-adjustment decisions"""
        if not hasattr(self, '_performance_history'):
            self._performance_history = []

        self._performance_history.append({
            'epoch': self.current_epoch,
            'cls_accuracy': cls_accuracy,
            'seg_loss': seg_loss,
            'strategy': self.fine_tuning_strategy
        })

        # Keep only last 20 epochs
        if len(self._performance_history) > 20:
            self._performance_history = self._performance_history[-20:]

    def _configure_optimizer_from_plans(self, param_groups):
        """
        Enhanced optimizer and scheduler configuration supporting multiple optimizers
        and parameter groups based on the new plans.json structure.
        """
        plans = self.plans_manager.plans

        # Check for new multi-optimizer configuration structure
        if 'optimizer_configs' in plans and 'active_optimizer' in plans:
            return self._configure_multi_optimizer_from_plans(param_groups)

        # Fallback to legacy single optimizer configuration
        return self._configure_legacy_optimizer_from_plans(param_groups)

    def _configure_multi_optimizer_from_plans(self, param_groups):
        """
        Configure optimizer using the new multi-optimizer configuration system.
        """
        plans = self.plans_manager.plans
        optimizer_configs = plans.get('optimizer_configs', {})
        active_optimizer = plans.get('active_optimizer', 'sgd')
        parameter_group_config = plans.get('parameter_group_config', {})

        print("🔧 Using multi-optimizer configuration system")
        print(f"🎯 Active optimizer: {active_optimizer}")

        # Get the active optimizer configuration
        if active_optimizer not in optimizer_configs:
            print(f"⚠️ Active optimizer '{active_optimizer}' not found in optimizer_configs!")
            print(f"📋 Available optimizers: {list(optimizer_configs.keys())}")
            # Use first available optimizer as fallback
            active_optimizer = list(optimizer_configs.keys())[0] if optimizer_configs else 'sgd'
            print(f"🔄 Falling back to: {active_optimizer}")

        optimizer_config = optimizer_configs.get(active_optimizer, {})
        optimizer_class_name = optimizer_config.get('class', 'torch.optim.SGD')
        optimizer_kwargs = optimizer_config.get('kwargs', {}).copy()

        print(f"📦 Optimizer class: {optimizer_class_name}")
        print(f"⚙️ Optimizer kwargs: {optimizer_kwargs}")

        # Apply parameter group configurations if enabled
        use_different_lr = parameter_group_config.get('use_different_lr_for_components', False)
        if use_different_lr and param_groups:
            print("🎛️ Applying parameter group learning rate multipliers")
            param_groups = self._apply_parameter_group_config(param_groups, parameter_group_config, optimizer_kwargs)

        # Import and create optimizer
        optimizer = self._create_optimizer_from_config(optimizer_class_name, param_groups, optimizer_kwargs)

        # Configure scheduler
        scheduler = self._configure_scheduler_from_plans(optimizer)

        return optimizer, scheduler

    def _configure_legacy_optimizer_from_plans(self, param_groups):
        """
        Enhanced legacy optimizer configuration for backward compatibility.
        Now properly supports parameter groups with different learning rates.
        """
        print("🔧 Using legacy optimizer configuration (enhanced)")

        plans = self.plans_manager.plans

        # Get optimizer configuration with better defaults
        optimizer_config = plans.get('optimizer', {})
        optimizer_class_name = optimizer_config.get('class', 'torch.optim.SGD')
        optimizer_kwargs = optimizer_config.get('kwargs', {}).copy()

        print(f"📦 Legacy optimizer class: {optimizer_class_name}")
        print(f"⚙️ Legacy optimizer kwargs: {optimizer_kwargs}")

        # Enhanced parameter group handling for legacy format
        if param_groups and len(param_groups) > 1:
            print(f"🎛️ Applying enhanced parameter group handling for {len(param_groups)} groups")

            # Apply any parameter group multipliers if available in plans
            param_group_config = plans.get('parameter_group_config', {})
            if param_group_config.get('use_different_lr_for_components', False):
                param_groups = self._apply_parameter_group_config(param_groups, param_group_config, optimizer_kwargs)
            else:
                # Ensure each parameter group has proper learning rate from base config
                base_lr = optimizer_kwargs.get('lr', self.initial_lr)
                for group in param_groups:
                    if 'lr' not in group:
                        group['lr'] = base_lr

            # Log parameter group details
            for group in param_groups:
                group_name = group.get('name', 'unnamed')
                group_lr = group.get('lr', 'inherited')
                param_count = sum(p.numel() for p in group['params'] if p.requires_grad)
                print(f"    - {group_name}: {param_count:,} parameters, lr: {group_lr}")

        # Create optimizer with parameter groups or single parameter list
        if param_groups:
            # Use parameter groups (supports different learning rates)
            optimizer_params = param_groups
        else:
            # Fallback to all model parameters
            optimizer_params = [p for p in self.network.parameters() if p.requires_grad]

        # Create optimizer
        optimizer = self._create_optimizer_from_config(optimizer_class_name, optimizer_params, optimizer_kwargs)

        # Configure scheduler
        scheduler = self._configure_scheduler_from_plans(optimizer)

        return optimizer, scheduler

    def _apply_parameter_group_config(self, param_groups, parameter_group_config, base_optimizer_kwargs):
        """
        Apply parameter group-specific learning rate multipliers.
        """
        base_lr = base_optimizer_kwargs.get('lr', self.initial_lr)

        # Get multipliers from config
        encoder_multiplier = parameter_group_config.get('encoder_lr_multiplier', 1.0)
        decoder_multiplier = parameter_group_config.get('decoder_lr_multiplier', 1.0)
        cls_head_multiplier = parameter_group_config.get('cls_head_lr_multiplier', 1.0)

        print(f"📊 LR multipliers - Encoder: {encoder_multiplier}, Decoder: {decoder_multiplier}, Cls: {cls_head_multiplier}")

        # Apply multipliers to parameter groups
        updated_param_groups = []
        for group in param_groups:
            group_copy = group.copy()
            group_name = group.get('name', 'unknown')

            # Determine multiplier based on component name
            if 'encoder' in group_name.lower():
                multiplier = encoder_multiplier
            elif 'decoder' in group_name.lower():
                multiplier = decoder_multiplier
            elif 'cls' in group_name.lower() or 'classification' in group_name.lower():
                multiplier = cls_head_multiplier
            else:
                multiplier = 1.0  # Default multiplier

            # Apply multiplier to learning rate
            if 'lr' in group_copy:
                original_lr = group_copy['lr']
                group_copy['lr'] = original_lr * multiplier
                print(f"🎯 {group_name}: {original_lr:.6f} → {group_copy['lr']:.6f} (×{multiplier})")
            else:
                group_copy['lr'] = base_lr * multiplier
                print(f"🎯 {group_name}: {base_lr:.6f} → {group_copy['lr']:.6f} (×{multiplier})")

            updated_param_groups.append(group_copy)

        return updated_param_groups

    def _create_optimizer_from_config(self, optimizer_class_name, param_groups, optimizer_kwargs):
        """
        Create optimizer instance from plans.json configuration.
        Since plans.json specification is robust, no fallback strategies are needed.
        """
        import pydoc

        try:
            if 'optim' not in optimizer_class_name:
                raise ImportError(f"Could not locate optimizer class: {optimizer_class_name}")
            optimizer_class = eval(optimizer_class_name)
            optimizer = optimizer_class(param_groups, **optimizer_kwargs)
            print(f"✅ Successfully created optimizer: {optimizer_class_name}")
            return optimizer

        except Exception as e:
            error_msg = (
                f"Failed to create optimizer '{optimizer_class_name}' with kwargs {optimizer_kwargs}. "
                f"Error: {e}. Please check your plans.json optimizer configuration."
            )
            raise RuntimeError(error_msg) from e

    def _configure_scheduler_from_plans(self, optimizer):
        """
        Configure learning rate scheduler with support for multiple scheduler configs.
        """
        plans = self.plans_manager.plans

        # Check for multi-scheduler configuration
        if 'scheduler_configs' in plans and 'active_scheduler' in plans:
            scheduler_configs = plans.get('scheduler_configs', {})
            active_scheduler = plans.get('active_scheduler', 'poly')

            print("🔧 Using multi-scheduler configuration")
            print(f"🎯 Active scheduler: {active_scheduler}")

            if active_scheduler not in scheduler_configs:
                print(f"⚠️ Active scheduler '{active_scheduler}' not found!")
                print(f"📋 Available schedulers: {list(scheduler_configs.keys())}")
                active_scheduler = list(scheduler_configs.keys())[0] if scheduler_configs else 'poly'
                print(f"🔄 Falling back to: {active_scheduler}")

            scheduler_config = scheduler_configs.get(active_scheduler, {})
        else:
            # Legacy single scheduler configuration
            print("🔧 Using legacy scheduler configuration")
            scheduler_config = plans.get('scheduler', {})

        scheduler_class_name = scheduler_config.get('class', 'nnunetv2.training.lr_scheduler.polylr.PolyLRScheduler')
        scheduler_kwargs = scheduler_config.get('kwargs', {}).copy()

        print(f"📦 Scheduler class: {scheduler_class_name}")
        print(f"⚙️ Scheduler kwargs: {scheduler_kwargs}")

        # Create scheduler with fallback handling
        return self._create_scheduler_from_config(scheduler_class_name, optimizer, scheduler_kwargs)

    def _create_scheduler_from_config(self, scheduler_class_name, optimizer, scheduler_kwargs):
        """
        Create scheduler instance with robust error handling.
        """
        import pydoc

        # Handle scheduler-specific parameter injection
        scheduler_kwargs = self._prepare_scheduler_kwargs(scheduler_class_name, scheduler_kwargs)

        # Primary attempt: use specified scheduler
        try:
            scheduler_class = pydoc.locate(scheduler_class_name)
            if scheduler_class is None:
                raise ImportError(f"Could not locate {scheduler_class_name}")

            scheduler = scheduler_class(optimizer, **scheduler_kwargs)
            print(f"✅ Successfully created {scheduler_class_name}")
            return scheduler

        except Exception as e:
            print(f"❌ Failed to create {scheduler_class_name}: {e}")

        # Fallback 1: PolyLRScheduler (most common in nnUNet)
        try:
            from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
            poly_kwargs = {
                'initial_lr': scheduler_kwargs.get('initial_lr', self.initial_lr),
                'max_steps': scheduler_kwargs.get('max_steps', self.num_epochs)
            }
            scheduler = PolyLRScheduler(optimizer, **poly_kwargs)
            print(f"✅ Fallback successful: Using PolyLRScheduler")
            return scheduler

        except Exception as e:
            print(f"❌ PolyLRScheduler fallback failed: {e}")

        # Fallback 2: PyTorch StepLR (most robust)
        try:
            from torch.optim.lr_scheduler import StepLR
            step_kwargs = {
                'step_size': scheduler_kwargs.get('step_size', max(1, self.num_epochs // 4)),
                'gamma': scheduler_kwargs.get('gamma', 0.5)
            }
            scheduler = StepLR(optimizer, **step_kwargs)
            print(f"✅ Final fallback: Using StepLR")
            return scheduler

        except Exception as e:
            raise RuntimeError(f"All scheduler fallbacks failed! Final error: {e}")

    def _prepare_scheduler_kwargs(self, scheduler_class_name, scheduler_kwargs):
        """
        Prepare scheduler kwargs based on scheduler type.
        """
        kwargs = scheduler_kwargs.copy()
        name_lower = scheduler_class_name.lower()

        try:
            if 'polylr' in name_lower:
                # PolyLRScheduler expects initial_lr and max_steps
                kwargs.setdefault('initial_lr', self.initial_lr)
                kwargs.setdefault('max_steps', self.num_epochs)
            elif 'polynomiallr' in name_lower:
                # PyTorch PolynomialLR expects total_iters
                kwargs.setdefault('total_iters', self.num_epochs)
                kwargs.setdefault('power', 0.9)
            elif 'cosineannealinglr' in name_lower:
                # CosineAnnealingLR expects T_max
                kwargs.setdefault('T_max', self.num_epochs)
                kwargs.setdefault('eta_min', kwargs.get('eta_min', 1e-6))
            elif 'steplr' in name_lower:
                # StepLR expects step_size
                kwargs.setdefault('step_size', max(1, self.num_epochs // 4))
                kwargs.setdefault('gamma', 0.5)
            # Add more scheduler types as needed

        except Exception as e:
            print(f"⚠️ Error preparing scheduler kwargs: {e}")
            # Safe fallback kwargs
            kwargs = {
                'initial_lr': self.initial_lr,
                'max_steps': self.num_epochs
            }

        return kwargs
