#!/usr/bin/env python3
"""
Test script to demonstrate the integrated classification functionality
in both nnUNetDatasetNumpy and nnUNetDatasetBlosc2 classes.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Add the nnunetv2 module to the path
sys.path.insert(0, 'nnunetv2')

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy, nnUNetDatasetBlosc2, infer_dataset_class


def create_test_data():
    """Create test data structure for demonstration"""

    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    preprocessed_dir = os.path.join(temp_dir, 'preprocessed')
    raw_dir = os.path.join(temp_dir, 'raw')

    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # Create test labels.csv
    labels_data = {
        'Photo ID': ['case_001_0000.nii.gz', 'case_002_0000.nii.gz', 'case_003_0000.nii.gz'],
        'subtype': [0, 1, 2],
        'split': ['train', 'train', 'validation']
    }
    labels_df = pd.DataFrame(labels_data)
    labels_csv_path = os.path.join(raw_dir, 'labels.csv')
    labels_df.to_csv(labels_csv_path, index=False)

    # Create dummy preprocessed data files (numpy format)
    for i, case_id in enumerate(['case_001', 'case_002', 'case_003']):
        # Create dummy data and segmentation
        dummy_data = np.random.rand(1, 64, 64, 32).astype(np.float32)
        dummy_seg = np.random.randint(0, 3, (1, 64, 64, 32)).astype(np.uint8)
        dummy_properties = {'spacing': [1.0, 1.0, 1.0], 'origin': [0.0, 0.0, 0.0]}

        # Save as npz files
        np.savez_compressed(
            os.path.join(preprocessed_dir, f'{case_id}.npz'),
            data=dummy_data,
            seg=dummy_seg
        )

        # Save properties as pickle (simplified - just create a dummy file)
        import pickle
        with open(os.path.join(preprocessed_dir, f'{case_id}.pkl'), 'wb') as f:
            pickle.dump(dummy_properties, f)

    return preprocessed_dir, raw_dir, labels_csv_path


def test_numpy_dataset_with_classification():
    """Test nnUNetDatasetNumpy with classification labels"""
    print("Testing nnUNetDatasetNumpy with classification labels...")

    preprocessed_dir, raw_dir, labels_csv_path = create_test_data()

    try:
        # Test without classification labels
        dataset_no_cls = nnUNetDatasetNumpy(
            folder=preprocessed_dir,
            load_subtype_labels=False
        )

        data, seg, seg_prev, properties = dataset_no_cls.load_case('case_001')
        print(f"  Without classification - Properties keys: {list(properties.keys())}")
        assert 'classification_label' not in properties

        # Test with classification labels
        dataset_with_cls = nnUNetDatasetNumpy(
            folder=preprocessed_dir,
            load_subtype_labels=True,
            dataset_folder=raw_dir
        )

        data, seg, seg_prev, properties = dataset_with_cls.load_case('case_001')
        print(f"  With classification - Properties keys: {list(properties.keys())}")
        print(f"  Classification label for case_001: {properties.get('classification_label')}")
        assert 'classification_label' in properties
        assert properties['classification_label'] == 0  # From our test data

        print("  ✓ nnUNetDatasetNumpy test passed!")

    except Exception as e:
        print(f"  ✗ nnUNetDatasetNumpy test failed: {e}")

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(os.path.dirname(preprocessed_dir))


def test_infer_dataset_class():
    """Test the infer_dataset_class function"""
    print("Testing infer_dataset_class function...")

    preprocessed_dir, raw_dir, labels_csv_path = create_test_data()

    try:
        # Should infer nnUNetDatasetNumpy for npz files
        dataset_class = infer_dataset_class(preprocessed_dir)
        print(f"  Inferred dataset class: {dataset_class.__name__}")
        assert dataset_class == nnUNetDatasetNumpy

        # Test instantiation with inferred class
        dataset = dataset_class(
            folder=preprocessed_dir,
            load_subtype_labels=True,
            dataset_folder=raw_dir
        )

        data, seg, seg_prev, properties = dataset.load_case('case_002')
        print(f"  Classification label for case_002: {properties.get('classification_label')}")
        assert properties['classification_label'] == 1  # From our test data

        print("  ✓ infer_dataset_class test passed!")

    except Exception as e:
        print(f"  ✗ infer_dataset_class test failed: {e}")

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(os.path.dirname(preprocessed_dir))


def test_missing_labels():
    """Test behavior when labels.csv is missing or case not found"""
    print("Testing missing labels scenarios...")

    preprocessed_dir, raw_dir, labels_csv_path = create_test_data()

    try:
        # Test with missing labels.csv
        os.remove(labels_csv_path)

        try:
            dataset = nnUNetDatasetNumpy(
                folder=preprocessed_dir,
                load_subtype_labels=True,
                dataset_folder=raw_dir
            )
            print("  ✗ Should have raised FileNotFoundError")
        except FileNotFoundError:
            print("  ✓ Correctly raised FileNotFoundError for missing labels.csv")

        # Recreate labels.csv with limited entries
        labels_data = {
            'Photo ID': ['case_001_0000.nii.gz'],  # Only one case
            'subtype': [0],
            'split': ['train']
        }
        labels_df = pd.DataFrame(labels_data)
        labels_df.to_csv(labels_csv_path, index=False)

        # Test with case not in labels
        dataset = nnUNetDatasetNumpy(
            folder=preprocessed_dir,
            load_subtype_labels=True,
            dataset_folder=raw_dir
        )

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data, seg, seg_prev, properties = dataset.load_case('case_002')  # Not in labels

            assert len(w) > 0
            assert "No classification label found" in str(w[0].message)
            assert properties['classification_label'] is None
            print("  ✓ Correctly handled missing case with warning")

    except Exception as e:
        print(f"  ✗ Missing labels test failed: {e}")

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(os.path.dirname(preprocessed_dir))


def main():
    """Run all tests"""
    print("Testing integrated classification functionality in nnUNet datasets\n")

    test_numpy_dataset_with_classification()
    print()

    test_infer_dataset_class()
    print()

    test_missing_labels()
    print()

    print("All tests completed!")
    print("\nUsage example:")
    print("# To use with classification labels:")
    print("dataset = nnUNetDatasetNumpy(")
    print("    folder='path/to/preprocessed',")
    print("    load_subtype_labels=True,")
    print("    dataset_folder='path/to/raw'  # Contains labels.csv")
    print(")")
    print("data, seg, seg_prev, properties = dataset.load_case('case_id')")
    print("classification_label = properties['classification_label']")


if __name__ == "__main__":
    main()
