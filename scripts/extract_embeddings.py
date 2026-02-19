import os
import argparse
from nnunetv2.inference.bottleneck_predictor import BottleneckPredictor
from batchgenerators.utilities.file_and_folder_operations import join
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Extract bottleneck embeddings from preprocessed nnUNet data')
    
    # Required arguments
    parser.add_argument('-d', '--dataset_id', type=str, required=True,
                       help='Dataset ID and name (e.g. Dataset001_BrainTumor)')
    parser.add_argument('-c', '--configuration', type=str, required=True,
                       help='Model configuration (e.g. 2d, 3d_fullres)')
    
    # Optional arguments
    parser.add_argument('-f', '--folds', nargs='+', type=int, default=[0],
                       help='Folds to use for prediction (e.g. 0 1 2 3 4)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth',
                       help='Checkpoint name to use (default: checkpoint_final.pth)')
    parser.add_argument('--no_save', action='store_false', dest='save_embeddings',
                       help='Do not save embeddings to disk')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Verify environment variables are set
    if not all(os.environ.get(var) for var in ['nnUNet_preprocessed', 'nnUNet_results']):
        raise RuntimeError(
            "Environment variables nnUNet_preprocessed and nnUNet_results must be set. "
            "Please see nnunetv2/documentation/setting_up_paths.md"
        )
    
    # Initialize predictor
    predictor = BottleneckPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(args.device if torch.cuda.is_available() else 'cpu'),
        verbose=False
    )
    
    # Setup paths
    model_folder = join(os.environ['nnUNet_results'], 
                       args.dataset_id,
                       f'nnUNetTrainer__nnUNetPlans__{args.configuration}')
    preprocessed_folder = join(os.environ['nnUNet_preprocessed'],
                             args.dataset_id,
                             f'nnUNetPlans_{args.configuration}')
    output_folder = join(os.environ['nnUNet_results'],
                        args.dataset_id,
                        'bottleneck_features')
    
    # Initialize from model folder
    print(f"Loading model from {model_folder}")
    print(f"Using folds: {args.folds}")
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=tuple(args.folds),
        checkpoint_name=args.checkpoint
    )
    
    # Extract embeddings
    print(f"Processing preprocessed data from {preprocessed_folder}")
    print(f"Saving results to {output_folder}")
    embeddings = predictor.predict_from_preprocessed_folder(
        preprocessed_folder,
        output_folder if args.save_embeddings else None,
        save_embeddings=args.save_embeddings
    )
    
    print("Extraction complete!")
    return embeddings

if __name__ == "__main__":
    main()
