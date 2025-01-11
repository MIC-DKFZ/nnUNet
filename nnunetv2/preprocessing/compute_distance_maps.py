# Compute the multi-class distance transform for each ground truth segmentation

import os
import numpy as np
import edt
import argparse
import time
import pickle

def compute_distance_transforms(input_dir, multiclass=False, overwrite=False):
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('_seg.npy')]

    for label_filename in npy_files:
        case_num = label_filename.replace('_seg.npy', '')
        save_path = os.path.join(input_dir, f"{case_num}_dist.npy")

        # Skip already existing files unless overwrite is True
        if os.path.exists(save_path) and not overwrite:
            print(f"Skipping {case_num} as distance map already exists and overwrite is set to False.")
            continue

        print(f"Processing file: {label_filename}")
        label_path = os.path.join(input_dir, label_filename)
        label_data = np.load(label_path) # Shape: (1, <image shape>)

        # Load spacing information from .pkl file
        pkl_path = os.path.join(input_dir, f"{case_num}.pkl")
        with open(pkl_path, 'rb') as f:
            label_info = pickle.load(f)
        spacing = label_info['spacing']

        if multiclass:
            # Compute multi-class distance map
            distance_maps = edt.edt(label_data[0] + 1, anisotropy=spacing)  # Adding 1 makes the "background" also a class to provide distance map inside and out
            distance_maps = np.expand_dims(distance_maps, axis=0)
        else:
            # Compute distance map for each class (excluding background)
            num_classes = int(np.amax(label_data))
            distance_maps = []
            for class_value in range(1, num_classes + 1):
                class_mask = (label_data[0] == class_value)
                if not np.any(class_mask):
                    print(f"---- Class {class_value} is empty, setting distance map to zero.")
                    distance_map = np.zeros(label_data.shape)
                else:
                    print(f"---- Computing distance transform for class: {class_value}")
                    distance_map = edt.edt(class_mask + 1, anisotropy=spacing)
                distance_maps.append(distance_map)

            # Combine distance maps along the first axis (one per class)
            if distance_maps:
                distance_maps = np.stack(distance_maps, axis=0)
                
        distance_maps = distance_maps.astype(np.float16)
        start_time = time.time()
        print(f"---- Saving distance map to {save_path} at {start_time}")
        np.save(save_path, distance_maps)
        end_time = time.time()
        print(f"---- Finished saving distance map at {end_time}")
        print(f"---- Time taken to save: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute distance transforms for segmentation labels.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing labels.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing distance maps if set.')
    parser.add_argument('--multi_class', action='store_true', help='Compute multi-class distance transform instead of distance transforms for each class (significantly saves space)')
    args = parser.parse_args()

    compute_distance_transforms(args.input_dir, args.multi_class, args.overwrite)


# python3 compute_distance_maps.py --input_dir /home/andyding/tbone-seg-nnunetv2/01_nnUNetv2_tbone/nnUNet_preprocessed/Dataset101_TemporalBone/nnUNetPlans_3d_fullres/ --multi_class --overwrite