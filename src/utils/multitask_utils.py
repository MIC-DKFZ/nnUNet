"""
Utility functions for multi-task nnUNet implementation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def extract_classification_targets_from_batch(batch: Dict[str, Any],
                                            num_classes: int = 3) -> torch.Tensor:
    """
    Extract classification targets from batch metadata.

    This function extracts classification labels from filenames or metadata
    in the batch. The implementation depends on how classification labels
    are stored in your dataset.

    Args:
        batch: Batch dictionary containing 'keys' and other metadata
        num_classes: Number of classification classes

    Returns:
        torch.Tensor: Classification targets of shape (batch_size,)
    """
    batch_size = batch['data'].shape[0]
    cls_targets = []

    for i in range(batch_size):
        # Extract from keys or properties - adjust based on your data structure
        filename = batch['keys'][i] if 'keys' in batch else f"case_{i}"

        # Example classification logic based on filename patterns
        # Adjust this based on your actual data structure
        if 'subtype0' in filename or 'type_0' in filename:
            cls_targets.append(0)
        elif 'subtype1' in filename or 'type_1' in filename:
            cls_targets.append(1)
        elif 'subtype2' in filename or 'type_2' in filename:
            cls_targets.append(2)
        else:
            # Default to class 0 if no pattern matches
            cls_targets.append(0)

    return torch.tensor(cls_targets, dtype=torch.long)


def compute_multitask_metrics(seg_outputs: torch.Tensor,
                            seg_targets: torch.Tensor,
                            cls_outputs: torch.Tensor,
                            cls_targets: torch.Tensor,
                            label_manager: Any) -> Dict[str, float]:
    """
    Compute metrics for both segmentation and classification tasks.

    Args:
        seg_outputs: Segmentation predictions
        seg_targets: Segmentation ground truth
        cls_outputs: Classification predictions (logits)
        cls_targets: Classification ground truth
        label_manager: nnUNet label manager

    Returns:
        Dictionary containing computed metrics
    """
    metrics = {}

    # Classification metrics
    cls_probs = torch.softmax(cls_outputs, dim=1)
    cls_preds = torch.argmax(cls_probs, dim=1)

    # Convert to numpy for sklearn metrics
    cls_preds_np = cls_preds.detach().cpu().numpy()
    cls_targets_np = cls_targets.detach().cpu().numpy()

    # Classification accuracy
    cls_accuracy = accuracy_score(cls_targets_np, cls_preds_np)
    metrics['classification_accuracy'] = cls_accuracy

    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        cls_targets_np, cls_preds_np, average=None, zero_division=0
    )

    for i in range(len(precision)):
        metrics[f'classification_precision_class_{i}'] = precision[i]
        metrics[f'classification_recall_class_{i}'] = recall[i]
        metrics[f'classification_f1_class_{i}'] = f1[i]

    # Macro averages
    metrics['classification_precision_macro'] = np.mean(precision)
    metrics['classification_recall_macro'] = np.mean(recall)
    metrics['classification_f1_macro'] = np.mean(f1)

    # Segmentation metrics (basic Dice computation)
    if label_manager.has_regions:
        seg_probs = torch.sigmoid(seg_outputs)
        seg_preds = (seg_probs > 0.5).float()
    else:
        seg_probs = torch.softmax(seg_outputs, dim=1)
        seg_preds = torch.argmax(seg_probs, dim=1)
        # Convert to one-hot for Dice computation
        seg_preds_onehot = torch.zeros_like(seg_outputs)
        seg_preds_onehot.scatter_(1, seg_preds.unsqueeze(1), 1)
        seg_preds = seg_preds_onehot

    # Compute Dice scores
    axes = [0] + list(range(2, seg_outputs.ndim))

    # Handle ignore label if present
    if label_manager.has_ignore_label:
        if not label_manager.has_regions:
            mask = (seg_targets != label_manager.ignore_label).float()
            seg_targets_masked = seg_targets.clone()
            seg_targets_masked[seg_targets == label_manager.ignore_label] = 0
        else:
            if seg_targets.dtype == torch.bool:
                mask = ~seg_targets[:, -1:]
            else:
                mask = 1 - seg_targets[:, -1:]
            seg_targets_masked = seg_targets[:, :-1]
    else:
        mask = None
        seg_targets_masked = seg_targets

    # Convert targets to one-hot if needed
    if not label_manager.has_regions and seg_targets_masked.ndim == seg_outputs.ndim - 1:
        seg_targets_onehot = torch.zeros_like(seg_outputs)
        seg_targets_onehot.scatter_(1, seg_targets_masked.unsqueeze(1), 1)
        seg_targets_masked = seg_targets_onehot

    # Compute intersection and union
    if mask is not None:
        intersection = torch.sum(seg_preds * seg_targets_masked * mask, dim=axes)
        union = torch.sum((seg_preds + seg_targets_masked) * mask, dim=axes)
    else:
        intersection = torch.sum(seg_preds * seg_targets_masked, dim=axes)
        union = torch.sum(seg_preds + seg_targets_masked, dim=axes)

    # Dice coefficient
    dice_scores = (2.0 * intersection) / (union + 1e-8)

    # Average Dice (excluding background if not using regions)
    if not label_manager.has_regions:
        dice_scores = dice_scores[1:]  # Remove background

    mean_dice = torch.mean(dice_scores).item()
    metrics['segmentation_dice_mean'] = mean_dice

    # Per-class Dice scores
    for i, dice in enumerate(dice_scores):
        metrics[f'segmentation_dice_class_{i}'] = dice.item()

    return metrics


def log_multitask_metrics(logger: Any,
                         metrics: Dict[str, float],
                         epoch: int,
                         prefix: str = '') -> None:
    """
    Log multi-task metrics to the nnUNet logger.

    Args:
        logger: nnUNet logger instance
        metrics: Dictionary of computed metrics
        epoch: Current epoch number
        prefix: Prefix for metric names (e.g., 'train_', 'val_')
    """
    for metric_name, metric_value in metrics.items():
        full_metric_name = f"{prefix}{metric_name}"
        logger.log(full_metric_name, metric_value, epoch)


def create_multitask_summary(seg_metrics: Dict[str, float],
                           cls_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Create a summary of multi-task performance.

    Args:
        seg_metrics: Segmentation metrics
        cls_metrics: Classification metrics

    Returns:
        Dictionary containing performance summary
    """
    summary = {
        'segmentation': {
            'mean_dice': seg_metrics.get('segmentation_dice_mean', 0.0),
            'per_class_dice': {
                k: v for k, v in seg_metrics.items()
                if k.startswith('segmentation_dice_class_')
            }
        },
        'classification': {
            'accuracy': cls_metrics.get('classification_accuracy', 0.0),
            'macro_f1': cls_metrics.get('classification_f1_macro', 0.0),
            'macro_precision': cls_metrics.get('classification_precision_macro', 0.0),
            'macro_recall': cls_metrics.get('classification_recall_macro', 0.0),
            'per_class_f1': {
                k: v for k, v in cls_metrics.items()
                if k.startswith('classification_f1_class_')
            }
        }
    }

    # Combined score (weighted average)
    seg_weight = 0.7
    cls_weight = 0.3
    combined_score = (seg_weight * summary['segmentation']['mean_dice'] +
                     cls_weight * summary['classification']['macro_f1'])
    summary['combined_score'] = combined_score

    return summary


def format_multitask_metrics_for_logging(metrics: Dict[str, float]) -> str:
    """
    Format metrics for pretty printing in logs.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Formatted string for logging
    """
    lines = []

    # Segmentation metrics
    seg_metrics = {k: v for k, v in metrics.items() if k.startswith('segmentation_')}
    if seg_metrics:
        lines.append("Segmentation Metrics:")
        for k, v in seg_metrics.items():
            clean_name = k.replace('segmentation_', '').replace('_', ' ').title()
            lines.append(f"  {clean_name}: {v:.4f}")

    # Classification metrics
    cls_metrics = {k: v for k, v in metrics.items() if k.startswith('classification_')}
    if cls_metrics:
        lines.append("Classification Metrics:")
        for k, v in cls_metrics.items():
            clean_name = k.replace('classification_', '').replace('_', ' ').title()
            lines.append(f"  {clean_name}: {v:.4f}")

    return '\n'.join(lines)
