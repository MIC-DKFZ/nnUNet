#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, reduction='mean'):
        """
        Args:
            gamma (float): Focusing parameter (default: 2).
            alpha (float or list or None): Weighting factor for class imbalance (default: None).
                If None, alpha will be computed based on the class frequencies.
            reduction (str): Specifies the reduction to apply to the output ('none', 'mean', 'sum'). Default: 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the focal loss.

        Args:
            inputs (torch.Tensor): The predicted logits, shape (N, C, D, H, W) or (N, C, H, W).
            targets (torch.Tensor): The ground truth, shape (N, D, H, W) or (N, H, W), where N is batch size.
            
        Returns:
            torch.Tensor: The computed focal loss value.
        """
        # Ensure targets are in long type for indexing
        targets = targets.long()
        if targets.ndim == inputs.ndim:
            assert targets.shape[1] == 1
            targets = targets[:, 0]

        # If alpha is not provided, calculate it based on class frequencies
        if self.alpha is None:
            # Flatten the targets to compute class distribution
            target_flat = targets.view(-1)

            # Use torch.bincount() to get the number of occurrences for each class
            class_counts = torch.bincount(target_flat, minlength=inputs.shape[1]).float()

            total_count = target_flat.size(0)

            # Compute alpha: Inverse of class frequencies, normalized to sum to 1
            alpha = 1.0 / (class_counts + 1e-7)  # Add small epsilon to avoid division by zero
            alpha /= alpha.sum()  # Normalize the alpha values
            
            # Transfer to the same device as inputs
            alpha = alpha.to(inputs.device)
        else:
            # If alpha is provided as a constant or tensor, ensure it is on the correct device
            if isinstance(self.alpha, (list, torch.Tensor)):
                alpha = torch.tensor(self.alpha, dtype=torch.float32, device=inputs.device)
            else:
                alpha = torch.full((inputs.shape[1],), self.alpha, dtype=torch.float32, device=inputs.device)

        # Softmax over the channel dimension (class dimension)
        probs = F.softmax(inputs, dim=1)

        # Expand targets to match the shape of probs (batch_size, D, H, W) -> (batch_size, 1, D, H, W)
        targets_expanded = targets.unsqueeze(1)  # Shape: [N, 1, D, H, W]

        # Index the predicted probabilities for the correct class using torch.gather
        # torch.gather takes the dimension to gather along (1 - for classes), and the index tensor (targets_expanded)
        pt = probs.gather(1, targets_expanded)  # Shape: [N, 1, D, H, W]
        pt = pt.squeeze(1)  # Remove the class dimension: Shape: [N, D, H, W]

        # Clamp probabilities to avoid log(0)
        pt = torch.clamp(pt, min=1e-8, max=1.0 - 1e-8)

        # Focal loss computation
        loss = -alpha[targets] * (1 - pt) ** self.gamma * torch.log(pt)

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")