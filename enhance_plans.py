#!/usr/bin/env python3
"""
Script to convert existing nnUNet plans to use the enhanced multi-task end-to-end trainer.
"""

import json
import argparse
import os
from pathlib import Path

def enhance_plans(input_plans, output_plans, config_overrides=None):
    """
    Convert existing plans to use enhanced multi-task trainer.

    Args:
        input_plans: Path to existing nnUNetPlans.json
        output_plans: Path for enhanced plans
        config_overrides: Dict of configuration overrides
    """

    # Load existing plans
    with open(input_plans, 'r') as f:
        plans = json.load(f)

    # Default enhanced configuration
    enhanced_config = {
        'trainer_class_name': 'nnUNetTrainerMultiTaskEnd2End',

        # Gradient surgery settings
        'gradient_surgery_method': 'pcgrad',
        'enable_gradient_surgery': True,

        # Uncertainty loss settings
        'adaptive_weighting': True,
        'loss_type': 'dice_ce',

        # Task balancing
        'task_balancing': {
            'method': 'uncertainty',
            'seg_weight': 1.0,
            'cls_weight': 1.0
        },

        # Classification settings
        'num_classification_classes': 3,  # Adjust based on dataset
        'classification_mode': 'spatial_attention_multiscale',

        # Fine-tuning strategy
        'fine_tuning_strategy': 'full',
        'auto_adjust_strategy': True,

        # Training enhancements
        'gradient_clipping': True,
        'gradient_clip_norm': 1.0,

        # Monitoring
        'log_gradient_stats': True,
        'log_task_weights': True,

        # Learning rates
        'learning_rates': {
            'encoder_lr': 1e-4,
            'cls_lr': 1e-3
        },

        # Batch configuration
        'batch_config': {
            'gradient_accumulation_steps': 2,
            'effective_batch_multiplier': 2
        }
    }

    # Apply user overrides
    if config_overrides:
        enhanced_config.update(config_overrides)

    # Merge with existing plans
    plans.update(enhanced_config)

    # Save enhanced plans
    with open(output_plans, 'w') as f:
        json.dump(plans, f, indent=2)

    print(f"✓ Enhanced plans saved to: {output_plans}")
    return plans

def main():
    parser = argparse.ArgumentParser(description='Enhance nnUNet plans for multi-task end-to-end training')
    parser.add_argument('input_plans', help='Path to existing nnUNetPlans.json')
    parser.add_argument('-o', '--output', help='Output path for enhanced plans', default=None)
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classification classes')
    parser.add_argument('--surgery-method', choices=['pcgrad', 'graddrop', 'mgda'], default='pcgrad',
                       help='Gradient surgery method')
    parser.add_argument('--fine-tuning', choices=['minimal', 'partial', 'full'], default='full',
                       help='Fine-tuning strategy')
    parser.add_argument('--no-surgery', action='store_true', help='Disable gradient surgery')
    parser.add_argument('--static-weights', nargs=2, type=float, metavar=('SEG', 'CLS'),
                       help='Use static task weights instead of uncertainty weighting')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input_plans):
        print(f"Error: Input plans file not found: {args.input_plans}")
        return 1

    # Determine output path
    if args.output is None:
        input_path = Path(args.input_plans)
        output_path = input_path.parent / 'nnUNetPlansEnhanced.json'
    else:
        output_path = Path(args.output)

    # Prepare configuration overrides
    config_overrides = {
        'num_classification_classes': args.num_classes,
        'gradient_surgery_method': args.surgery_method,
        'enable_gradient_surgery': not args.no_surgery,
        'fine_tuning_strategy': args.fine_tuning
    }

    # Handle static weights
    if args.static_weights:
        config_overrides['task_balancing'] = {
            'method': 'static',
            'seg_weight': args.static_weights[0],
            'cls_weight': args.static_weights[1]
        }
        config_overrides['adaptive_weighting'] = False

    # Enhance plans
    try:
        enhanced_plans = enhance_plans(args.input_plans, output_path, config_overrides)

        print("\n" + "="*60)
        print("ENHANCED PLANS SUMMARY")
        print("="*60)
        print(f"Input plans: {args.input_plans}")
        print(f"Output plans: {output_path}")
        print(f"Trainer: {enhanced_plans['trainer_class_name']}")
        print(f"Gradient surgery: {enhanced_plans['gradient_surgery_method']} ({'enabled' if enhanced_plans['enable_gradient_surgery'] else 'disabled'})")
        print(f"Task balancing: {enhanced_plans['task_balancing']['method']}")
        print(f"Fine-tuning: {enhanced_plans['fine_tuning_strategy']}")
        print(f"Classification classes: {enhanced_plans['num_classification_classes']}")
        print("="*60)

        print("\nNext steps:")
        print(f"1. Review the enhanced plans: {output_path}")
        print("2. Adjust num_classification_classes if needed")
        print("3. Run training with:")
        print(f"   nnUNetv2_train DATASET 3d_fullres FOLD --trainer nnUNetTrainerMultiTaskEnd2End --plans_name {output_path.stem}")

        return 0

    except Exception as e:
        print(f"Error enhancing plans: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
