import sys
import os.path
from os import listdir
from os.path import isfile, join
from shutil import copy
import argparse
from argparse import RawTextHelpFormatter

def create_folders(num_folds):
    for i in range(1, num_folds + 1):  # Create fold folder
        if not os.path.exists(f'Fold_{i}'):
            os.mkdir(f'Fold_{i}')
            for j in ["Test", "Train"]:  # Test and Train folders
                for x in ["Images", "Segmentations"]:  # Image and Segmentation folders
                    if not os.path.exists(os.path.join(f'Fold_{i}', j, x)):
                        os.makedirs(os.path.join(f'Fold_{i}', j, x))

def populate_folders(test_segmentations, test_images, model_type, num_folds, split_ratio):
    segmentations = os.listdir(test_segmentations)
    images = os.listdir(test_images)

    for fold_num in range(1, num_folds + 1): # loop through each of the 10 fold folders
        print(f"Processing Fold_{fold_num}")
        # Calculate split indices based on user-defined split ratio

        print(f"Using split ratio: {split_ratio}")
        num_seg_total = len(segmentations)
        print(f"Total segmentation files found: {num_seg_total}")
        num_img_total = len(images)
        print(f"Total image files found: {num_img_total}")
        if num_seg_total != num_img_total or num_seg_total % num_img_total != 0:
            print("Error: There is a mismatch in the number of segmentation files and the number of image files.")
            sys.exit(1)
        num_test_segs = int(num_seg_total * split_ratio)
        print(f"Number of test segmentation files per fold: {num_test_segs}")
        # Select every Nth segmentation for test set, where N = int(1/split_ratio)
        N = int(1 / split_ratio)
        test_indices_segs = set(i for i in range(num_seg_total) if (i % N) == (fold_num - 1))
        print(f"Test indices for this fold: {sorted(test_indices_segs)}")

        # SEGMENTATIONS
        for seg_file_index in range(num_seg_total):
            if seg_file_index in test_indices_segs:  # Test data
                if not os.path.exists(os.path.join(f'Fold_{fold_num}', "Test", "Segmentations", segmentations[seg_file_index])):
                    copy(os.path.join(test_segmentations, segmentations[seg_file_index]), os.path.join(f'Fold_{fold_num}', "Test", "Segmentations"))
                    #print(f"Copied {segmentations[seg_file_index]} to Fold_{fold_num}/Test/Segmentations")
            else:   # Train data
                if not os.path.exists(os.path.join(f'Fold_{fold_num}', "Train", "Segmentations", segmentations[seg_file_index])):
                    copy(os.path.join(test_segmentations, segmentations[seg_file_index]), os.path.join(f'Fold_{fold_num}', "Train", "Segmentations"))
                    #print(f"Copied {segmentations[seg_file_index]} to Fold_{fold_num}/Train/Segmentations")  

        # IMAGES
        num_test_img = int(num_img_total * split_ratio)
        print(f"Number of test image files per fold: {num_test_img}")
        # Select every Nth segmentation for test set, where N = int(1/split_ratio)
        N = int(1 / split_ratio)
        test_indices_imgs = set(i for i in range(num_img_total) if (i % N) == (fold_num - 1))
        print(f"Test indices for this fold: {sorted(test_indices_imgs)}")

        for img_file_index in range(num_img_total):
            if img_file_index in test_indices_imgs:  # Test data
                if (model_type == 0 or model_type == 2) and (not os.path.exists(os.path.join(f'Fold_{fold_num}', "Test", "Images", images[img_file_index]))):
                    copy(os.path.join(test_images, images[img_file_index]), os.path.join(f'Fold_{fold_num}', "Test", "Images"))
                if (model_type == 1 or model_type == 2) and img_file_index + 1 < num_img_total and (not os.path.exists(os.path.join(f'Fold_{fold_num}', "Test", "Images", images[img_file_index + 1]))):
                    copy(os.path.join(test_images, images[img_file_index + 1]), os.path.join(f'Fold_{fold_num}', "Test", "Images"))
            else: # Train data
                if (model_type == 0 or model_type == 2) and (not os.path.exists(os.path.join(f'Fold_{fold_num}', "Train", "Images", images[img_file_index]))):
                    copy(os.path.join(test_images, images[img_file_index]), os.path.join(f'Fold_{fold_num}', "Train", "Images"))
                if (model_type == 1 or model_type == 2) and img_file_index + 1 < num_img_total and (not os.path.exists(os.path.join(f'Fold_{fold_num}', "Train", "Images", images[img_file_index + 1]))):
                    copy(os.path.join(test_images, images[img_file_index + 1]), os.path.join(f'Fold_{fold_num}', "Train", "Images"))
def main():
    parser = argparse.ArgumentParser(
        description='Create and populate 10-fold cross validation folders',
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("--seg-folder", required=True, help="Path to segmentation files")
    parser.add_argument("--img-folder", required=True, help="Path to image files")
    parser.add_argument("--model-type", type=int, choices=[0, 1, 2], default=2,
                       help="Model type: 0 for t1 only, 1 for t2 only, 2 for both")
    parser.add_argument("--num-folds", type=int, choices=range(1, 11), default=10,
                       help="Number of folds to create (1-10), default is 10")
    parser.add_argument("--split-ratio", type=float, default=0.1,
                       help="Ratio of test data in each fold (e.g., 0.1 for 10%% test, 90%% train), default is 0.1")
    args = parser.parse_args()

    create_folders(args.num_folds)
    populate_folders(args.seg_folder, args.img_folder, args.model_type, args.num_folds, args.split_ratio)

if __name__ == "__main__":
    main()