from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import numpy as np
import torch
import os

class BottleneckEnsemblePredictor(nnUNetPredictor):
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                           use_folds: tuple[int, ...],
                                           checkpoint_name: str = "checkpoint_final.pth"):
        # Try checkpoint_final.pth first, fall back to checkpoint_best.pth if not found
        if not all(os.path.exists(os.path.join(model_training_output_dir, f'fold_{i}', checkpoint_name)) 
                  for i in use_folds):
            print(f"Warning: {checkpoint_name} not found in all folds, falling back to checkpoint_best.pth")
            checkpoint_name = "checkpoint_best.pth"
        
        return super().initialize_from_trained_model_folder(
            model_training_output_dir,
            use_folds,
            checkpoint_name
        )

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict):
        self.network.eval()
        all_fold_features = []
        
        with torch.no_grad():
            x = torch.from_numpy(input_image).cuda(self.device, non_blocking=True)
            
            for network in self.networks_and_mirrors:
                net = network[0]
                features = net.encoder(x)  # Get bottleneck features
                all_fold_features.append(features.cpu().numpy())
            
            # Average across folds
            ensemble_features = np.mean(all_fold_features, axis=0)
            return ensemble_features
    
    def predict_from_files(self, input_folder: str, output_folder: str, *args, **kwargs):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # Call parent class but capture embeddings
        embeddings = super().predict_from_files(input_folder, output_folder, *args, **kwargs)
        
        # Save embeddings
        np.save(os.path.join(output_folder, 'bottleneck_embeddings.npy'), embeddings)
        return embeddings
