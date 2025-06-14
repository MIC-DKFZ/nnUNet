#!/usr/bin/env python3
"""
Enhanced Multi-Task End-to-End Trainer with Uncertainty Loss and Gradient Surgery
Implements both uncertainty weighting and PCGrad-style gradient surgery for optimal multi-task learning.
"""
import torch
from torch import nn
import numpy as np
import warnings
from typing import Dict, List, Optional, Union
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerFrozenEncoderClsImproved import nnUNetTrainerFrozenEncoderClsImproved
from nnunetv2.training.loss.multitask_losses import MultiTaskUncertaintyLoss
from nnunetv2.utilities.collate_outputs import collate_outputs
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import copy


class PCGrad:
    """
    Project Conflicting Gradients (PCGrad) implementation for multi-task learning.

    Based on "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020)
    https://arxiv.org/abs/2001.06782
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: Reduction method for gradient conflicts ('mean' or 'sum')
        """
        self.reduction = reduction

    def _project_conflicting_grad(self, grad_task: torch.Tensor, grad_ref: torch.Tensor) -> torch.Tensor:
        """
        Project grad_task onto the normal plane of grad_ref if they conflict (negative cosine similarity).

        Args:
            grad_task: Gradient of the task to be projected
            grad_ref: Reference gradient

        Returns:
            Projected gradient
        """
        # Flatten gradients
        grad_task_flat = grad_task.flatten()
        grad_ref_flat = grad_ref.flatten()

        # Compute cosine similarity
        cos_sim = torch.dot(grad_task_flat, grad_ref_flat) / (
            torch.norm(grad_task_flat) * torch.norm(grad_ref_flat) + 1e-8
        )

        # If gradients conflict (negative cosine similarity), project
        if cos_sim < 0:
            # Project grad_task onto the normal plane of grad_ref
            # proj = grad - (grad · ref / ||ref||²) * ref
            dot_product = torch.dot(grad_task_flat, grad_ref_flat)
            ref_norm_sq = torch.norm(grad_ref_flat) ** 2 + 1e-8
            projection = (dot_product / ref_norm_sq) * grad_ref_flat
            proj_grad = grad_task_flat - projection

            # Handle edge case where projection results in zero gradient
            if torch.norm(proj_grad) < 1e-6:
                # Return a small orthogonal vector instead of zero
                orthogonal = torch.randn_like(grad_task_flat) * 0.01
                orthogonal = orthogonal - (torch.dot(orthogonal, grad_ref_flat) / ref_norm_sq) * grad_ref_flat
                return orthogonal.view_as(grad_task)

            return proj_grad.view_as(grad_task)
        else:
            return grad_task

    def apply_gradient_surgery(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply gradient surgery to resolve conflicts between task gradients.
        Uses the official PCGrad algorithm with proper handling of edge cases.

        Args:
            gradients: List of gradients for each task [grad_seg, grad_cls]

        Returns:
            List of modified gradients after surgery
        """
        if len(gradients) != 2:
            warnings.warn("PCGrad currently supports exactly 2 tasks")
            return gradients

        grad_seg, grad_cls = gradients

        # Flatten gradients for computation
        grad_seg_flat = grad_seg.flatten()
        grad_cls_flat = grad_cls.flatten()

        # Compute cosine similarity
        cos_sim = torch.dot(grad_seg_flat, grad_cls_flat) / (
            torch.norm(grad_seg_flat) * torch.norm(grad_cls_flat) + 1e-8
        )

        # If no conflict, return original gradients
        if cos_sim >= 0:
            return [grad_seg, grad_cls]

        # Apply PCGrad: project each gradient onto the normal plane of the other
        # but only if they conflict

        # Project grad_seg onto normal plane of grad_cls
        dot_product_seg = torch.dot(grad_seg_flat, grad_cls_flat)
        cls_norm_sq = torch.norm(grad_cls_flat) ** 2 + 1e-8
        proj_seg = grad_seg_flat - (dot_product_seg / cls_norm_sq) * grad_cls_flat

        # Project grad_cls onto normal plane of grad_seg
        dot_product_cls = torch.dot(grad_cls_flat, grad_seg_flat)
        seg_norm_sq = torch.norm(grad_seg_flat) ** 2 + 1e-8
        proj_cls = grad_cls_flat - (dot_product_cls / seg_norm_sq) * grad_seg_flat

        # Handle edge case where projections are too small
        if torch.norm(proj_seg) < 1e-6:
            # Use average gradient direction
            proj_seg = 0.5 * (grad_seg_flat + grad_cls_flat)

        if torch.norm(proj_cls) < 1e-6:
            # Use average gradient direction
            proj_cls = 0.5 * (grad_seg_flat + grad_cls_flat)

        return [proj_seg.view_as(grad_seg), proj_cls.view_as(grad_cls)]


class GradientSurgery:
    """
    Enhanced gradient surgery with multiple strategies for multi-task optimization.
    """

    def __init__(self, method: str = 'pcgrad', alpha: float = 0.5):
        """
        Args:
            method: Surgery method ('pcgrad', 'graddrop', 'mgda')
            alpha: Balance parameter for some methods
        """
        self.method = method
        self.alpha = alpha
        self.pcgrad = PCGrad()

    def compute_task_gradients(self, losses: Dict[str, torch.Tensor],
                             parameters: List[torch.nn.Parameter]) -> Dict[str, List[torch.Tensor]]:
        """
        Compute gradients for each task separately.

        Args:
            losses: Dictionary of task losses
            parameters: Model parameters

        Returns:
            Dictionary mapping task names to their gradient lists
        """
        task_gradients = {}

        for task_name, loss in losses.items():
            # Clear gradients
            for param in parameters:
                if param.grad is not None:
                    param.grad.zero_()

            # Compute gradients for this task
            loss.backward(retain_graph=True)

            # Store gradients
            task_gradients[task_name] = []
            for param in parameters:
                if param.grad is not None:
                    task_gradients[task_name].append(param.grad.clone())
                else:
                    # Handle parameters without gradients
                    task_gradients[task_name].append(torch.zeros_like(param))

        return task_gradients

    def apply_surgery(self, task_gradients: Dict[str, List[torch.Tensor]],
                     parameters: List[torch.nn.Parameter]) -> None:
        """
        Apply gradient surgery and update parameter gradients.

        Args:
            task_gradients: Gradients for each task
            parameters: Model parameters to update
        """
        if self.method == 'pcgrad':
            self._apply_pcgrad(task_gradients, parameters)
        elif self.method == 'graddrop':
            self._apply_graddrop(task_gradients, parameters)
        elif self.method == 'mgda':
            self._apply_mgda(task_gradients, parameters)
        else:
            # Fallback: simple averaging
            self._apply_averaging(task_gradients, parameters)

    def _apply_pcgrad(self, task_gradients: Dict[str, List[torch.Tensor]],
                     parameters: List[torch.nn.Parameter]) -> None:
        """Apply PCGrad surgery."""
        task_names = list(task_gradients.keys())
        if len(task_names) != 2:
            warnings.warn("PCGrad works best with 2 tasks, falling back to averaging")
            self._apply_averaging(task_gradients, parameters)
            return

        # Get gradients for both tasks
        grad_lists = [task_gradients[task] for task in task_names]

        # Apply surgery parameter by parameter
        for i, param in enumerate(parameters):
            if i < len(grad_lists[0]) and i < len(grad_lists[1]):
                grads = [grad_lists[0][i], grad_lists[1][i]]
                if grads[0].numel() > 0 and grads[1].numel() > 0:
                    projected_grads = self.pcgrad.apply_gradient_surgery(grads)
                    # Average the projected gradients
                    param.grad = (projected_grads[0] + projected_grads[1]) / 2.0

    def _apply_graddrop(self, task_gradients: Dict[str, List[torch.Tensor]],
                       parameters: List[torch.nn.Parameter]) -> None:
        """Apply GradDrop: randomly drop conflicting gradients."""
        task_names = list(task_gradients.keys())

        for i, param in enumerate(parameters):
            valid_grads = []
            for task in task_names:
                if i < len(task_gradients[task]) and task_gradients[task][i].numel() > 0:
                    valid_grads.append(task_gradients[task][i])

            if len(valid_grads) > 1:
                # Compute pairwise cosine similarities
                similarities = []
                for j in range(len(valid_grads)):
                    for k in range(j + 1, len(valid_grads)):
                        grad_j = valid_grads[j].flatten()
                        grad_k = valid_grads[k].flatten()
                        cos_sim = torch.dot(grad_j, grad_k) / (
                            torch.norm(grad_j) * torch.norm(grad_k) + 1e-8
                        )
                        similarities.append(cos_sim.item())

                # If any conflicts (negative similarity), randomly drop one gradient
                if any(sim < 0 for sim in similarities):
                    keep_idx = np.random.randint(len(valid_grads))
                    param.grad = valid_grads[keep_idx]
                else:
                    # No conflicts, average all gradients
                    param.grad = torch.stack(valid_grads).mean(dim=0)
            elif len(valid_grads) == 1:
                param.grad = valid_grads[0]

    def _apply_mgda(self, task_gradients: Dict[str, List[torch.Tensor]],
                   parameters: List[torch.nn.Parameter]) -> None:
        """Apply MGDA-style weighting (simplified version)."""
        task_names = list(task_gradients.keys())
        num_tasks = len(task_names)

        # Compute gradient norms for weighting
        grad_norms = {}
        for task in task_names:
            total_norm = 0.0
            for grad in task_gradients[task]:
                if grad.numel() > 0:
                    total_norm += grad.norm().item() ** 2
            grad_norms[task] = np.sqrt(total_norm)

        # Compute weights (inverse of norms for balancing)
        total_norm = sum(grad_norms.values())
        weights = {task: (total_norm / (norm + 1e-8)) / num_tasks
                  for task, norm in grad_norms.items()}

        # Apply weighted combination
        for i, param in enumerate(parameters):
            weighted_grad = None
            for task in task_names:
                if i < len(task_gradients[task]) and task_gradients[task][i].numel() > 0:
                    if weighted_grad is None:
                        weighted_grad = weights[task] * task_gradients[task][i]
                    else:
                        weighted_grad += weights[task] * task_gradients[task][i]

            if weighted_grad is not None:
                param.grad = weighted_grad

    def _apply_averaging(self, task_gradients: Dict[str, List[torch.Tensor]],
                        parameters: List[torch.nn.Parameter]) -> None:
        """Simple gradient averaging fallback."""
        task_names = list(task_gradients.keys())

        for i, param in enumerate(parameters):
            valid_grads = []
            for task in task_names:
                if i < len(task_gradients[task]) and task_gradients[task][i].numel() > 0:
                    valid_grads.append(task_gradients[task][i])

            if valid_grads:
                param.grad = torch.stack(valid_grads).mean(dim=0)


class nnUNetTrainerMultiTaskEnd2End(nnUNetTrainerFrozenEncoderClsImproved):
    """
    Enhanced multi-task trainer with uncertainty loss weighting and gradient surgery.

    Features:
    1. Uncertainty-based automatic loss weighting
    2. PCGrad-style gradient surgery to resolve task conflicts
    3. Configurable gradient surgery methods
    4. End-to-end training of both segmentation and classification
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Override parent settings for end-to-end training
        self.fine_tuning_strategy = 'full'  # Train everything end-to-end
        self.gradient_accumulation_steps = 2  # Smaller for stability

        # Gradient surgery configuration
        self.gradient_surgery_method = getattr(plans, 'gradient_surgery_method', 'pcgrad')
        self.surgery_alpha = getattr(plans, 'surgery_alpha', 0.5)
        self.enable_gradient_surgery = getattr(plans, 'enable_gradient_surgery', True)

        # Initialize gradient surgery
        self.gradient_surgery = GradientSurgery(
            method=self.gradient_surgery_method,
            alpha=self.surgery_alpha
        )
        self.dec_lr = getattr(plans, 'decoder_learning_rate', self.cls_lr)
        # Loss tracking for adaptive weighting
        self.loss_history = {
            'seg_loss': [],
            'cls_loss': [],
            'total_loss': []
        }
        self.adaptive_weighting = getattr(plans, 'adaptive_weighting', True)
        # self.batch_size = getattr(plans, 'batch_size', 1)
        self.batch_size = 1
        self.unfreeze_stages = getattr(plans, 'unfreeze_stages', [3,4,5])

    def _build_loss(self):
        """Build uncertainty-weighted multi-task loss."""
        return MultiTaskUncertaintyLoss(
            loss_type='dice_ce',
            ddp=self.is_ddp,
            ignore_label=self.label_manager.ignore_label
        )

    def _setup_fine_tuning_strategy(self, model):
        """
        Override to ensure full end-to-end training with proper parameter groups.
        This creates parameter groups compatible with the enhanced optimizer configuration system.
        """
        print("🔧 Setting up end-to-end multi-task training")

        # Unfreeze everything for end-to-end training
        for param in model.parameters():
            param.requires_grad = True

        # Collect parameters for different components
        enc_params = []
        dec_params = []
        cls_params = []

        # Collect encoder parameters
        for stage_idx in self.unfreeze_stages:
            if stage_idx < len(model.unet.encoder.stages):
                stage = model.unet.encoder.stages[stage_idx]
                enc_params.extend(list(stage.parameters()))
                print(f"    - Unfrozen encoder stage {stage_idx}")
            else:
                print(f"    - Warning: Stage {stage_idx} does not exist")


        # Collect decoder parameters (excluding encoder parameters)
        for name, param in model.unet.decoder.named_parameters():
            if not name.startswith('encoder.'):
                dec_params.append(param)

        # Collect classification head parameters
        for name, param in model.cls_head.named_parameters():
            cls_params.append(param)

        # Create parameter groups with component names for the optimizer configuration system
        param_groups = [
            {
                'params': enc_params,
                'lr': self.enc_lr * 0.1,  # Lower LR for encoder in end-to-end training
                'name': 'encoder',
                'component': 'encoder'  # Used by parameter group config system
            },
            {
                'params': dec_params,
                'lr': self.dec_lr,
                'name': 'decoder',
                'component': 'decoder'  # Used by parameter group config system
            },
            {
                'params': cls_params,
                'lr': self.cls_lr,
                'name': 'cls_head',
                'component': 'cls_head'  # Used by parameter group config system
            }
        ]

        # Filter out empty parameter groups
        param_groups = [group for group in param_groups if len(list(group['params'])) > 0]

        # Log parameter counts
        total_trainable = 0
        for group in param_groups:
            group_params = sum(p.numel() for p in group['params'] if p.requires_grad)
            total_trainable += group_params
            print(f"    - {group['name']}: {group_params:,} parameters (lr: {group['lr']:.2e})")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  📊 Total trainable: {total_trainable:,} / {total_params:,} ({100*total_trainable/total_params:.2f}%)")
        print(f"  🎯 Training strategy: End-to-end multi-task learning")

        return param_groups

    def train_step(self, batch: dict) -> dict:
        """Enhanced training step with uncertainty weighting and gradient surgery."""
        data = batch['data'].to(self.device)
        target_seg = batch['target'].to(self.device)
        target_cls = self._extract_classification_targets(batch)

        # Gradient accumulation setup
        if not hasattr(self, '_accumulation_step'):
            self._accumulation_step = 0
            self.optimizer.zero_grad()

        # Forward pass
        output = self.network(data)

        # Compute losses using uncertainty weighting
        loss_dict = self.loss(output['segmentation'], target_seg, output['classification'], target_cls)

        total_loss = loss_dict['loss']
        seg_loss = loss_dict['segmentation_loss']
        cls_loss = loss_dict['classification_loss']

        # Scale for gradient accumulation
        scaled_loss = total_loss / self.gradient_accumulation_steps

        if self.enable_gradient_surgery and self._accumulation_step == 0:
            # Apply gradient surgery
            self._apply_gradient_surgery(loss_dict, scaled_loss)
        else:
            # Standard backward pass
            scaled_loss.backward()

        self._accumulation_step += 1

        # Update weights every gradient_accumulation_steps
        if self._accumulation_step >= self.gradient_accumulation_steps:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._accumulation_step = 0

        # Compute metrics
        with torch.no_grad():
            cls_pred = torch.argmax(output['classification'], dim=1)
            cls_accuracy = (cls_pred == target_cls).float().mean().item()

        # Track losses for adaptive weighting
        self.loss_history['seg_loss'].append(seg_loss.item())
        self.loss_history['cls_loss'].append(cls_loss.item())
        self.loss_history['total_loss'].append(total_loss.item())

        # Keep only recent history
        if len(self.loss_history['seg_loss']) > 100:
            for key in self.loss_history:
                self.loss_history[key] = self.loss_history[key][-100:]

        # Debug uncertainty weights
        if self.current_epoch % 5 == 0 and hasattr(loss_dict, 'log_sigma_seg'):
            seg_weight = torch.exp(-loss_dict['log_sigma_seg']).item()
            cls_weight = torch.exp(-loss_dict['log_sigma_cls']).item()
            self.print_to_log_file(f"Uncertainty weights - Seg: {seg_weight:.4f}, Cls: {cls_weight:.4f}")

        return {
            'loss': total_loss,
            'seg_loss': seg_loss.item(),
            'cls_loss': cls_loss.item(),
            'cls_accuracy': cls_accuracy,
            'total_loss': total_loss.item()
        }

    def _apply_gradient_surgery(self, loss_dict: dict, scaled_loss: torch.Tensor):
        """Apply gradient surgery to resolve task conflicts."""
        # Get individual task losses
        seg_loss = loss_dict['segmentation_loss'] / self.gradient_accumulation_steps
        cls_loss = loss_dict['classification_loss'] / self.gradient_accumulation_steps

        # Compute task-specific gradients
        task_losses = {
            'segmentation': seg_loss,
            'classification': cls_loss
        }

        # Get all trainable parameters
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]

        # Compute gradients for each task
        task_gradients = self.gradient_surgery.compute_task_gradients(task_losses, trainable_params)

        # Apply gradient surgery
        self.gradient_surgery.apply_surgery(task_gradients, trainable_params)

    def validation_step(self, batch: dict) -> dict:
        """Enhanced validation step with uncertainty metrics."""
        data = batch['data'].to(self.device)
        target_seg = batch['target'].to(self.device)
        target_cls = self._extract_classification_targets(batch)

        with torch.no_grad():
            output = self.network(data)

            # Compute losses
            loss_dict = self.loss(output['segmentation'], target_seg, output['classification'], target_cls)

            total_loss = loss_dict['loss']
            seg_loss = loss_dict['segmentation_loss']
            cls_loss = loss_dict['classification_loss']

            # Compute classification metrics
            cls_pred = torch.argmax(output['classification'], dim=1)
            cls_true = target_cls

            # Convert to numpy for metrics
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
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]

        result = {
            'loss': total_loss.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
            'seg_loss': seg_loss.item(),
            'cls_loss': cls_loss.item(),
            'total_loss': total_loss.item(),
            'cls_pred': cls_pred_np,
            'cls_true': cls_true_np
        }

        # Add uncertainty metrics if available
        if 'log_sigma_seg' in loss_dict:
            result['log_sigma_seg'] = loss_dict['log_sigma_seg'].item()
            result['log_sigma_cls'] = loss_dict['log_sigma_cls'].item()

        return result

    def on_validation_epoch_end(self, val_outputs):
        """Enhanced validation epoch end with uncertainty tracking."""
        super().on_validation_epoch_end(val_outputs)

        # Log uncertainty weights if available
        outputs_collated = collate_outputs(val_outputs)
        if 'log_sigma_seg' in outputs_collated:
            avg_log_sigma_seg = np.mean(outputs_collated['log_sigma_seg'])
            avg_log_sigma_cls = np.mean(outputs_collated['log_sigma_cls'])

            # Convert to actual weights
            seg_weight = np.exp(-avg_log_sigma_seg)
            cls_weight = np.exp(-avg_log_sigma_cls)

            self.print_to_log_file(f"Validation uncertainty weights - Seg: {seg_weight:.4f}, Cls: {cls_weight:.4f}")
            self.print_to_log_file(f"Log sigma values - Seg: {avg_log_sigma_seg:.4f}, Cls: {avg_log_sigma_cls:.4f}")

            # Log to trainer's logger
            self.logger.log('val_uncertainty_seg_weight', seg_weight, self.current_epoch)
            self.logger.log('val_uncertainty_cls_weight', cls_weight, self.current_epoch)

        # Adaptive gradient surgery method switching
        if self.adaptive_weighting and len(self.loss_history['total_loss']) > 20:
            self._adapt_gradient_surgery_method()

    def _adapt_gradient_surgery_method(self):
        """Adaptively switch gradient surgery methods based on performance."""
        recent_losses = self.loss_history['total_loss'][-20:]
        loss_std = np.std(recent_losses)
        loss_trend = recent_losses[-1] - recent_losses[0]

        # If loss is unstable or increasing, try a different method
        if loss_std > 0.1 or loss_trend > 0:
            current_method = self.gradient_surgery.method

            # Cycle through methods
            methods = ['pcgrad', 'graddrop', 'mgda']
            current_idx = methods.index(current_method) if current_method in methods else 0
            new_method = methods[(current_idx + 1) % len(methods)]

            if new_method != current_method:
                self.print_to_log_file(f"Switching gradient surgery method from {current_method} to {new_method}")
                self.gradient_surgery.method = new_method

    def get_training_config_summary(self):
        """Enhanced configuration summary including gradient surgery settings."""
        config = super().get_training_config_summary()

        # Add gradient surgery configuration
        config['gradient_surgery'] = {
            'enabled': self.enable_gradient_surgery,
            'method': self.gradient_surgery_method,
            'alpha': self.surgery_alpha,
            'adaptive_weighting': self.adaptive_weighting
        }

        config['loss_function'] = 'MultiTaskUncertaintyLoss'
        config['training_mode'] = 'end_to_end'

        return config

    def log_training_config(self):
        """Enhanced configuration logging."""
        super().log_training_config()

        self.print_to_log_file(f"🔧 ENHANCED MULTI-TASK CONFIGURATION")
        self.print_to_log_file("="*80)

        self.print_to_log_file(f"Loss function: MultiTaskUncertaintyLoss")
        self.print_to_log_file(f"Training mode: End-to-end")

        if self.enable_gradient_surgery:
            self.print_to_log_file(f"Gradient surgery: {self.gradient_surgery_method}")
            self.print_to_log_file(f"Surgery alpha: {self.surgery_alpha}")
            self.print_to_log_file(f"Adaptive weighting: {self.adaptive_weighting}")
        else:
            self.print_to_log_file("Gradient surgery: Disabled")

        self.print_to_log_file("="*80 + "\n")
