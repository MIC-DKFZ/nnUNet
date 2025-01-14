from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2,
        smooth: float = 0.0,
        weight: torch.Tensor | None = None,
        nonlinearity: Callable[[torch.Tensor], torch.Tensor] = nn.Softmax(dim=1),
        ignore_index: int = -100
    ) -> None:
        """
        Focal loss that supports label smoothing and multiclass tasks. Inspired by the
        implementation in
        https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/nnunet/training_docker/nnUNetTrainerV2_focalLoss.py

        Args:
            gamma (float): Exponent of the modulating factor (1 - p_t). Larger values
                increase the weighting of misclassified samples.
            smooth (float, optional): A float in [0.0, 1.0]. Specifies the amount of
                label smoothing to use when computiing the loss. 0.0 corresponds to no
                smoothing.
            weight (torch.Tensor | None, optional): A manual rescaling weight given to
                each class. If provided, must be a Tensor of size C and floating point
                type where C is the number of classes. Meant to replace the alpha term
                in focal loss in a more explicit manner.
            nonlinearity (nn.Module): Nonlinearity to apply to output layer of nnUNet.
                Must constrain logits to the range [0, 1]
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        assert smooth >= 0 and smooth <= 1, "Smooth must be in range [0, 1]"
        self.smooth = smooth
        self.nonlinearity = nonlinearity
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes Focal Loss on logits and targets with initialized gamma and alpha.

        Args:
            logits (torch.Tensor): A float tensor (logits) of arbitrary shape. The
                predictions for each example. Must have be one hot encoded whith shape
                (N, C, ...) where N is the number of samples and C is the number of
                classes.
            targets (torch.Tensor): A float tensor containing correct class labels. If
                the same shape as logits, pytorch assumes it has already been one hot
                encoded. Otherwise, pytorch assumes it contains class indices.

        Returns:
            (torch.tensor) Focal Loss with mean reduction.
        """
        # Nnunet weirdness, get rid of an empty dim
        if targets.ndim == logits.ndim:
            assert targets.shape[1] == 1  
            targets = targets[:, 0]
        
        # Ensure target is an int. In nnunet they are floats
        targets = targets.long()

        # logits shape (N, C, ...), targets shape either (N, ...) or (N, C, ...)
        num_classes = logits.size(1)  # C
        ce_loss = F.cross_entropy(
            logits, targets, self.weight, reduction="none", label_smoothing=self.smooth, ignore_index=self.ignore_index
        )  # Returns shape (N, ...)

        # Ensure targets are one hot encoded
        if logits.shape != targets.shape:
            targets = F.one_hot(targets, num_classes=num_classes).movedim(-1, 1)
            assert logits.shape == targets.shape, f"Logits must have Shape (N, C, ...). Logits shape: {logits.shape}, Targets Shape after OHE: {targets.shape}"

        # Get probs and smooth labels
        probs = self.nonlinearity(logits)  # Shape (N, C, ...)
        targets_smoothed = (1 - self.smooth) * targets + self.smooth / num_classes

        # p_t contains the predicted probability of the correct class
        p_t = (probs * targets_smoothed).sum(1)  # Shape (N, ...)

        loss = ((1 - p_t) ** self.gamma) * ce_loss

        return loss.mean()


class FocalLossAndCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2,
        smooth: float = 0.0,
        weight: torch.Tensor | None = None,
        nonlinearity: Callable[[torch.Tensor], torch.Tensor] = nn.Softmax(dim=1),
        focal_loss_weight: float = 0.5,
        ignore_index: int = -100
    ):
        """
        Combination of FocalLoss and CrossEntropy Loss. Inputs must be logits and not probabilities.

        Args:
            gamma (float): Exponent of the modulating factor (1 - p_t). Larger values
                increase the weighting of misclassified samples inf Focal Loss.
            smooth (float, optional): A float in [0.0, 1.0]. Specifies the amount of
                label smoothing to use when computiing the Focal loss. 0.0 corresponds
                to no smoothing.
            weight (torch.Tensor | None, optional): A manual rescaling weight given to
                each class. If provided, must be a Tensor of size C and floating point
                type where C is the number of classes. Meant to replace the alpha term
                in focal loss in a more explicit manner. Does not apply to the CE loss
                component.
            nonlinearity (nn.Module): Nonlinearity to apply to output layer of nnUNet.
                Must constrain logits to the range [0, 1]
            focal_loss_weight (float): Weight between 0 and 1 to apply to weight focal 
                loss. (1 - focal_loss_weight) applied to cross entropy loss.
            ignore_index (int): Class index to ignore
        """
        super().__init__()

        assert focal_loss_weight >= 0 and focal_loss_weight <= 1.0
        self.focal_loss_weight = focal_loss_weight

        self.focal_loss = FocalLoss(gamma, smooth, weight, nonlinearity, ignore_index)
        self.cross_entropy_loss = RobustCrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Computes combination Cross Entropy Loss and Focal Loss on logits
            and targets with initialized gamma and alpha.

        Args:
            logits (torch.Tensor): A float tensor (logits) of arbitrary shape.
                    The predictions for each example.
            targets (torch.Tensor): A float tensor with the same shape as inputs. Stores
                the binary classification label for each element in inputs (0 for the
                negative class and 1 for the positive class).

        Returns:
            (torch.tensor) Focal Loss with mean reduction.
        """
        floss = self.focal_loss(logits, targets)
        celoss = self.cross_entropy_loss(logits, targets)
        result = self.focal_loss_weight * floss + (1 - self.focal_loss_weight) * celoss
        return result