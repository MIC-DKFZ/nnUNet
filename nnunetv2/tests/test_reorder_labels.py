"""
nnunetv2.utilities.reorder_labels: a model with permuted label ids predicts the same label names in every voxel.
"""
import tempfile
import unittest

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunetv2.inference.predict_streaming import nnUNetPredictorStreaming
from nnunetv2.tests.test_predict_streaming_batch import CASES, NUM_CLASSES, _image, write_model_folder
from nnunetv2.utilities.reorder_labels import label_permutation, reorder_model_labels


def _predict(model):
    p = nnUNetPredictorStreaming(tile_step_size=0.5, use_mirroring=False, device=torch.device('cpu'), allow_tqdm=False,
                                 quiet=True)
    p.initialize_from_folder_base_trainer(model, [0])
    return [p.predict_single_npy_array_streaming(_image(c[0], c[1]), {'spacing': list(c[2])}) for c in CASES]


class TestReorderLabels(unittest.TestCase):
    def test_same_segmentation_new_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, new = join(tmp, 'old'), join(tmp, 'new')
            write_model_folder(old)
            old_labels = load_json(join(old, 'dataset.json'))['labels']
            rng = np.random.default_rng(0)
            ids = [0, *(rng.permutation(NUM_CLASSES - 1) + 1).tolist()]          # background stays 0
            new_labels = {name: ids[i] for name, i in old_labels.items()}
            reorder_model_labels(old, new, new_labels)
            self.assertEqual(load_json(join(new, 'dataset.json'))['labels'],
                             dict(sorted(new_labels.items(), key=lambda kv: kv[1])))
            old_to_new = np.array([new_labels[name] for name, _ in sorted(old_labels.items(), key=lambda kv: kv[1])])
            for a, b in zip(_predict(old), _predict(new)):
                self.assertGreater(len(np.unique(a)), 2)
                np.testing.assert_array_equal(old_to_new[a], b)

    def test_rejects(self):
        with self.assertRaisesRegex(ValueError, 'names differ'):
            label_permutation({'background': 0, 'a': 1}, {'background': 0, 'b': 1})
        with self.assertRaisesRegex(ValueError, 'ids must be'):
            label_permutation({'background': 0, 'a': 1}, {'background': 0, 'a': 2})
        with self.assertRaisesRegex(ValueError, 'region'):
            label_permutation({'background': 0, 'a': (1, 2)}, {'background': 0, 'a': 1})


if __name__ == '__main__':
    unittest.main()
