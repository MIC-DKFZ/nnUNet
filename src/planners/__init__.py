"""
Multi-task planners for nnUNet.

These planners extend the ResEncUNet planner to support multi-task architectures
that perform both segmentation and classification tasks.
"""

from .multitask_base_planner import MultiTaskResEncUNetPlanner
from .multitask_channel_attention_planner import MultiTaskChannelAttentionResEncUNetPlanner
from .multitask_efficient_attention_planner import MultiTaskEfficientAttentionResEncUNetPlanner

__all__ = [
    'MultiTaskResEncUNetPlanner',
    'MultiTaskChannelAttentionResEncUNetPlanner',
    'MultiTaskEfficientAttentionResEncUNetPlanner'
]
