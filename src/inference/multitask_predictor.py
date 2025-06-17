import os
import torch
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict
from pathlib import Path
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile

from src.architectures.multitask_resenc_unet import MultiTaskResEncUNet


class MultiTaskPredictor:
    """
    Simplified predictor for MultiTask nnUNet models that handles both segmentation and classification
    """

    def __init__(self,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False):

        self.device = device
        self.verbose = verbose

        # Model components
        self.network = None
        self.plans = None
        self.configuration = None

    def initialize_from_trained_model_folder(self,
                                           model_training_output_dir: str,
                                           use_folds: Union[str, List[int]] = [0],
                                           checkpoint_name: str = 'checkpoint_final.pth'):
        """
        Initialize the predictor from a trained model folder
        """
        if self.verbose:
            print(f"Initializing MultiTask predictor from: {model_training_output_dir}")

        # Load plans from the preprocessed folder (not the results folder)
        # We need to find the preprocessed folder path
        dataset_name = model_training_output_dir.split('/')[-2]  # Extract dataset name
        preprocessed_base = model_training_output_dir.replace('nnUNet_results', 'nnUNet_preprocessed')
        preprocessed_base = '/'.join(preprocessed_base.split('/')[:-2])  # Remove trainer and config parts

        plans_file = join(preprocessed_base, dataset_name, 'nnUNetPlans_multitask.json')

        if not isfile(plans_file):
            raise RuntimeError(f"Plans file not found: {plans_file}")

        self.plans = load_json(plans_file)

        # Get 3d_fullres configuration
        if self.plans and 'configurations' in self.plans and '3d_fullres' in self.plans['configurations']:
            self.configuration = self.plans['configurations']['3d_fullres']
        else:
            raise RuntimeError("3d_fullres configuration not found in plans")

        # Determine fold to use
        if isinstance(use_folds, str) and use_folds == 'all':
            fold = 0  # Use first fold for simplicity
        else:
            fold = use_folds if isinstance(use_folds, int) else use_folds[0]

        # Load checkpoint
        fold_dir = join(model_training_output_dir, f'fold_{fold}')
        checkpoint_path = join(fold_dir, checkpoint_name)

        if not isfile(checkpoint_path):
            raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

        if self.verbose:
            print(f"Loading checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Build network architecture
        self.network = self._build_network()

        # Load weights - handle _orig_mod prefix from compiled models
        state_dict = checkpoint['network_weights']

        # Remove _orig_mod. prefix if present (from torch.compile)
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[10:]  # Remove '_orig_mod.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        self.network.load_state_dict(state_dict)
        self.network.eval()
        self.network.to(self.device)

        if self.verbose:
            print(f"✓ Successfully initialized MultiTask predictor")

    def _build_network(self) -> MultiTaskResEncUNet:
        """Build the network architecture from configuration"""
        if not self.configuration:
            raise RuntimeError("Configuration not loaded")

        arch_config = self.configuration['architecture']
        arch_kwargs = arch_config['arch_kwargs'].copy()

        # Import required classes
        import importlib
        for key in arch_config['_kw_requires_import']:
            if key in arch_kwargs and isinstance(arch_kwargs[key], str):
                module_path, class_name = arch_kwargs[key].rsplit('.', 1)
                module = importlib.import_module(module_path)
                arch_kwargs[key] = getattr(module, class_name)

        # Build network
        network = MultiTaskResEncUNet(
            input_channels=1,  # CT images have 1 channel
            n_stages=arch_kwargs['n_stages'],
            features_per_stage=arch_kwargs['features_per_stage'],
            conv_op=arch_kwargs['conv_op'],
            kernel_sizes=arch_kwargs['kernel_sizes'],
            strides=arch_kwargs['strides'],
            n_blocks_per_stage=arch_kwargs['n_blocks_per_stage'],
            num_classes=3,  # Background, pancreas, lesion
            n_conv_per_stage_decoder=arch_kwargs.get('n_conv_per_stage_decoder', [2, 2, 2, 1]),
            conv_bias=arch_kwargs.get('conv_bias', True),
            norm_op=arch_kwargs.get('norm_op'),
            norm_op_kwargs=arch_kwargs.get('norm_op_kwargs', {}),
            dropout_op=arch_kwargs.get('dropout_op'),
            dropout_op_kwargs=arch_kwargs.get('dropout_op_kwargs', {}),
            nonlin=arch_kwargs.get('nonlin'),
            nonlin_kwargs=arch_kwargs.get('nonlin_kwargs', {}),
            deep_supervision=False  # No deep supervision for inference
        )

        return network

    def predict_from_files(self,
                          list_of_lists_or_source_folder: Union[str, List[List[str]]],
                          output_folder_or_list_of_truncated_output_files: Union[str, List[str]],
                          save_probabilities: bool = False,
                          overwrite: bool = True,
                          num_processes_preprocessing: int = 2,
                          num_processes_segmentation_export: int = 2,
                          folder_with_segs_from_previous_stage: str = None,
                          num_parts: int = 1,
                          part_id: int = 0) -> Dict[str, Dict]:
        """
        Predict from files and return both segmentation and classification results
        """
        if isinstance(list_of_lists_or_source_folder, str):
            # Source folder provided
            source_folder = list_of_lists_or_source_folder
            if not isinstance(output_folder_or_list_of_truncated_output_files, str):
                raise ValueError("When source folder is provided, output must be a folder path string")
            output_folder = output_folder_or_list_of_truncated_output_files

            # Get list of input files
            input_files = [f for f in os.listdir(source_folder) if f.endswith('.nii.gz')]
            input_files.sort()

            # Create list of lists format
            list_of_lists = []
            for f in input_files:
                list_of_lists.append([join(source_folder, f)])

            # Create output file names
            output_files = []
            for f in input_files:
                case_id = f.replace('_0000.nii.gz', '')
                output_files.append(join(output_folder, f'{case_id}.nii.gz'))

        else:
            list_of_lists = list_of_lists_or_source_folder
            if not isinstance(output_folder_or_list_of_truncated_output_files, list):
                raise ValueError("When input is list of lists, output must be a list of file paths")
            output_files = output_folder_or_list_of_truncated_output_files

        # Ensure output folder exists
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            os.makedirs(output_folder_or_list_of_truncated_output_files, exist_ok=True)

        # Process each case
        classification_results = {}

        for i, (input_file_list, output_file) in enumerate(zip(list_of_lists, output_files)):
            if self.verbose:
                print(f"Processing {i+1}/{len(list_of_lists)}: {input_file_list[0]}")

            # Get case identifier
            case_id = Path(input_file_list[0]).stem.replace('_0000', '').replace('.nii', '')

            # Skip if output exists and not overwriting
            if not overwrite and os.path.exists(output_file):
                if self.verbose:
                    print(f"Skipping {case_id} (output exists)")
                continue

            # Predict single case
            seg_result, cls_result = self._predict_single_case(input_file_list, output_file, save_probabilities)

            # Store classification result
            classification_results[case_id] = cls_result

        return classification_results

    def _predict_single_case(self,
                           input_files: List[str],
                           output_file: str,
                           save_probabilities: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Predict a single case and return segmentation and classification results
        """
        # Load and preprocess image
        data, properties = self._load_and_preprocess_case(input_files)

        # Predict with network
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            data_tensor = torch.from_numpy(data).float().to(self.device).unsqueeze(0)

            # Forward pass through network
            if self.network is None:
                raise RuntimeError("Network not initialized")
            self.network.eval()
            output = self.network(data_tensor)

            seg_pred = torch.softmax(output['segmentation'], dim=1)
            cls_pred = torch.softmax(output['classification'], dim=1)

            # Remove batch dimension
            seg_prediction = seg_pred.cpu().numpy()[0]
            cls_prediction = cls_pred.cpu().numpy()[0]

        # Post-process segmentation
        seg_result = self._postprocess_segmentation(seg_prediction, properties, output_file, save_probabilities)

        # Process classification - handle unet_decoder output properly
        if self.configuration and 'architecture' in self.configuration:
            head_type = self.configuration['architecture'].get('classification_head', {}).get('head_type', 'mlp')

            if head_type == 'unet_decoder':
                # For unet_decoder: extract classification from lesion regions only
                cls_result = self._process_unet_decoder_classification(cls_prediction, seg_prediction)
            else:
                # For other head types: use direct classification output
                cls_result = {
                    'probabilities': cls_prediction.tolist(),
                    'predicted_class': int(np.argmax(cls_prediction)),
                    'confidence': float(np.max(cls_prediction))
                }
        else:
            # Fallback
            cls_result = {
                'probabilities': cls_prediction.tolist(),
                'predicted_class': int(np.argmax(cls_prediction)),
                'confidence': float(np.max(cls_prediction))
            }

        return seg_result, cls_result

    def _load_and_preprocess_case(self, input_files: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Load and preprocess a single case using SimpleITK
        """
        # Load image using SimpleITK
        image = sitk.ReadImage(input_files[0])
        data = sitk.GetArrayFromImage(image)

        # Get image properties
        original_spacing = np.array(image.GetSpacing())[::-1]  # ITK uses reverse order
        original_shape = data.shape

        # Get target spacing and patch size from configuration
        if not self.configuration:
            raise RuntimeError("Configuration not loaded")
        target_spacing = np.array(self.configuration['spacing'])
        patch_size = self.configuration['patch_size']

        # Resample to target spacing
        if not np.allclose(original_spacing, target_spacing, rtol=0.01):
            # Calculate new shape after resampling
            zoom_factors = original_spacing / target_spacing
            new_shape = [int(round(s * z)) for s, z in zip(original_shape, zoom_factors)]

            # Resample using scipy
            from scipy.ndimage import zoom
            data = zoom(data, zoom_factors, order=1)  # Linear interpolation

        # Crop or pad to patch size
        data = self._crop_or_pad_to_size(data, patch_size)

        # Add channel dimension if needed
        if len(data.shape) == 3:
            data = data[None]  # Add channel dimension

        # CT normalization using foreground properties from plans
        data = data.astype(np.float32)

        # Use the foreground intensity properties from the plans for normalization
        if self.plans and 'foreground_intensity_properties_per_channel' in self.plans:
            fg_props = self.plans['foreground_intensity_properties_per_channel']['0']
            # Clip to percentiles and normalize
            data = np.clip(data, fg_props['percentile_00_5'], fg_props['percentile_99_5'])
            data = (data - fg_props['mean']) / (fg_props['std'] + 1e-8)
        else:
            # Fallback normalization
            data = np.clip(data, -1000, 1000)
            data = (data - data.mean()) / (data.std() + 1e-8)

        # Store properties for post-processing
        properties = {
            'original_spacing': original_spacing,
            'original_shape': original_shape,
            'target_spacing': target_spacing,
            'patch_size': patch_size,
            'preprocessed_shape': data.shape[1:]
        }

        return data, properties

    def _crop_or_pad_to_size(self, data: np.ndarray, target_size: List[int]) -> np.ndarray:
        """
        Crop or pad data to target size
        """
        current_shape = data.shape

        # Calculate padding/cropping for each dimension
        processed_data = data
        for i, (current, target) in enumerate(zip(current_shape, target_size)):
            if current > target:
                # Crop - take center
                start = (current - target) // 2
                end = start + target
                slices = [slice(None)] * len(processed_data.shape)
                slices[i] = slice(start, end)
                processed_data = processed_data[tuple(slices)]
            elif current < target:
                # Pad - pad symmetrically
                pad_total = target - current
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                pad_width = [(0, 0)] * len(processed_data.shape)
                pad_width[i] = (pad_before, pad_after)
                processed_data = np.pad(processed_data, pad_width, mode='constant', constant_values=0)

        return processed_data

    def _postprocess_segmentation(self,
                                seg_prediction: np.ndarray,
                                properties: Dict,
                                output_file: str,
                                save_probabilities: bool = False) -> np.ndarray:
        """
        Post-process segmentation prediction and save to file
        """
        # Convert to segmentation map
        seg_map = np.argmax(seg_prediction, axis=0).astype(np.uint8)

        # Save segmentation
        seg_image = sitk.GetImageFromArray(seg_map)
        seg_image.SetSpacing(properties['original_spacing'][::-1])  # ITK uses reverse order
        sitk.WriteImage(seg_image, output_file)

        # Save probabilities if requested
        if save_probabilities:
            prob_file = output_file.replace('.nii.gz', '_probabilities.nii.gz')
            prob_image = sitk.GetImageFromArray(seg_prediction)
            prob_image.SetSpacing(properties['original_spacing'][::-1])
            sitk.WriteImage(prob_image, prob_file)

        return seg_map

    def _process_unet_decoder_classification(self,
                                           cls_prediction: np.ndarray,
                                           seg_prediction: np.ndarray) -> Dict:
        """
        Process unet_decoder classification output by extracting predictions from lesion regions

        Args:
            cls_prediction: [4, H, W, D] classification probabilities (classes 0, 1, 2, 3)
            seg_prediction: [3, H, W, D] segmentation probabilities (background, pancreas, lesion)

        Returns:
            Dictionary with processed classification result
        """
        # Get lesion mask from segmentation prediction
        seg_map = np.argmax(seg_prediction, axis=0)  # [H, W, D]
        lesion_mask = (seg_map == 2)  # Lesion regions

        if lesion_mask.sum() > 0:
            # Extract classification predictions within lesion regions
            # Only consider classes 0, 1, 2 (ignore class 3)
            lesion_cls_pred = cls_prediction[:3, lesion_mask]  # [3, N] where N is number of lesion pixels

            # Compute average probability per class within lesion regions
            class_probs = lesion_cls_pred.mean(axis=1)  # [3]

            # Normalize probabilities (should already be normalized from softmax, but ensure)
            class_probs = class_probs / (class_probs.sum() + 1e-8)

            # Get predicted class
            predicted_class = int(np.argmax(class_probs))
            confidence = float(class_probs[predicted_class])

        else:
            # No lesion detected - default to class 0
            predicted_class = 0
            confidence = 1.0
            class_probs = np.array([1.0, 0.0, 0.0])

        return {
            'probabilities': class_probs.tolist(),
            'predicted_class': predicted_class,
            'confidence': confidence,
            'lesion_pixels': int(lesion_mask.sum()) if lesion_mask.sum() > 0 else 0
        }

    def save_classification_results(self,
                                  classification_results: Dict[str, Dict],
                                  output_file: str):
        """
        Save classification results to CSV file
        """
        # Prepare data for CSV
        csv_data = []
        for case_id, result in classification_results.items():
            csv_data.append({
                'Names': case_id,
                'Subtype': result['predicted_class']
            })

        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)

        if self.verbose:
            print(f"✓ Saved classification results to: {output_file}")
            print(f"Classification distribution: {df['Subtype'].value_counts().to_dict()}")
