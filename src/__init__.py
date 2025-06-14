"""
Registration module for custom nnUNetv2 components
"""
import os
import sys
from pathlib import Path

# Add src directory to Python path for module discovery
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def register_custom_components():
    """Register custom components with nnUNetv2"""
    try:
        # Register experiment planner
        from nnunetv2.experiment_planning import experiment_planners
        from src.experiment_planning.multitask_residual_encoder_planner import MultiTasknnUNetPlannerResEncM

        if not hasattr(experiment_planners, 'MultiTasknnUNetPlannerResEncM'):
            setattr(experiment_planners, 'MultiTasknnUNetPlannerResEncM', MultiTasknnUNetPlannerResEncM)

        # Register architecture
        import dynamic_network_architectures.architectures
        from src.architectures.multitask_resenc_unet import MultiTaskResEncUNet

        if not hasattr(dynamic_network_architectures.architectures, 'MultiTaskResEncUNet'):
            setattr(dynamic_network_architectures.architectures, 'MultiTaskResEncUNet', MultiTaskResEncUNet)

        # Register trainer
        from nnunetv2.training.nnUNetTrainer import nnUNetTrainer
        from src.training.multitask_trainer import nnUNetTrainerMultiTask

        # Make trainer discoverable via string name
        import nnunetv2.training.nnUNetTrainer
        if not hasattr(nnunetv2.training.nnUNetTrainer, 'nnUNetTrainerMultiTask'):
            setattr(nnunetv2.training.nnUNetTrainer, 'nnUNetTrainerMultiTask', nnUNetTrainerMultiTask)

        print("✓ Custom components registered: planner, architecture, trainer")

    except ImportError as e:
        print(f"⚠ Failed to register custom components: {e}")

# Auto-register when module is imported
register_custom_components()