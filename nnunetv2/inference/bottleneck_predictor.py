from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import numpy as np
import torch
import os
from typing import Union, Tuple
from batchgenerators.utilities.file_and_folder_operations import join

class BottleneckPredictor(nnUNetPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = None
        
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                           use_folds: Union[Tuple[Union[int, str]], None],
                                           checkpoint_name: str = "checkpoint_final.pth"):
        """
        Initialize model from the trained model folder, with fallback to best checkpoint
        """
        # First try the specified checkpoint
        checkpoint_exists = False
        if use_folds is not None:
            checkpoint_exists = all(
                os.path.exists(join(model_training_output_dir, f'fold_{i}', checkpoint_name))
                for i in use_folds if i != 'all'
            )
        
        # Fallback to checkpoint_best if final not found
        if not checkpoint_exists:
            print(f"Warning: {checkpoint_name} not found in all folds, falling back to checkpoint_best.pth")
            checkpoint_name = "checkpoint_best.pth"
            
        return super().initialize_from_trained_model_folder(
            model_training_output_dir,
            use_folds,
            checkpoint_name
        )

    def predict_from_preprocessed_data(self, preprocessed_data: Union[str, np.ndarray]) -> np.ndarray:
        """
        Predict bottleneck features from preprocessed data
        Args:
            preprocessed_data: Either path to .npy file or numpy array
        Returns:
            bottleneck features as numpy array
        """
        self.network.eval()
        
        # Load data if path provided
        if isinstance(preprocessed_data, str):
            data = np.load(preprocessed_data)
        else:
            data = preprocessed_data
            
        # Convert to torch tensor
        with torch.no_grad():
            data = torch.from_numpy(data).to(self.device)
            if len(data.shape) == 3:
                data = data.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            elif len(data.shape) == 4:
                data = data.unsqueeze(0)  # Add batch dim only
                
            # Get bottleneck features
            features = self.network.encoder(data)
            return features.cpu().numpy()

    def predict_from_preprocessed_folder(self, 
                                       input_folder: str,
                                       output_folder: str = None,
                                       save_embeddings: bool = True) -> dict:
        """
        Predict bottleneck features for all .npy files in a folder
        Args:
            input_folder: Folder containing preprocessed .npy files
            output_folder: Where to save embeddings (optional)
            save_embeddings: Whether to save embeddings to disk
        Returns:
            Dictionary mapping filenames to bottleneck features
        """
        if output_folder is not None and not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        results = {}
        
        # Process all .npy files
        for filename in os.listdir(input_folder):
            if filename.endswith('.npy'):
                filepath = join(input_folder, filename)
                print(f"Processing {filename}...")
                
                # Get embeddings
                embeddings = self.predict_from_preprocessed_data(filepath)
                results[filename] = embeddings
                
                # Save if requested
                if save_embeddings and output_folder is not None:
                    output_path = join(output_folder, f"{filename.split('.')[0]}_embeddings.npy")
                    np.save(output_path, embeddings)
                    
        return results

# Example usage:
if __name__ == "__main__":
    # Set environment variables
    os.environ['nnUNet_preprocessed'] = r"C:\Users\Eliot Behr\VS\Data\HST18\nnUNet_preprocessed"
    os.environ['nnUNet_results'] = r"C:\Users\Eliot Behr\VS\Data\HST18\nnUNet_results"
    
    # Initialize predictor
    predictor = BottleneckPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose=False
    )
    
    # Initialize from model folder
    model_folder = join(os.environ['nnUNet_results'], 
                       'Dataset001_BrainTumor/nnUNetTrainer__nnUNetPlans__2d')
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),  # Using fold 0
    )
    
    # Predict from preprocessed folder
    preprocessed_folder = join(os.environ['nnUNet_preprocessed'],
                             'Dataset001_BrainTumor/nnUNetPlans_2d')
    output_folder = join(os.environ['nnUNet_results'],
                        'Dataset001_BrainTumor/bottleneck_features')
    
    embeddings = predictor.predict_from_preprocessed_folder(
        preprocessed_folder,
        output_folder
    )
