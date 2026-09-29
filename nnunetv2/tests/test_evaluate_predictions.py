import itertools
import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk

from nnunetv2.evaluation.evaluate_predictions import compute_confusion_matrix, \
    compute_tp_fp_fn_tn_for_labels_or_regions, compute_tp_fp_fn_tn, region_or_label_to_mask, compute_metrics, \
    compute_metrics_on_folder, load_summary_json
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.utilities.label_handling.label_handling import LabelManager


def reference_counts(seg_ref, seg_pred, labels_or_regions, ignore_label=None):
    """
    The implementation compute_metrics used before the confusion matrix. Serves as the oracle.
    """
    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None
    return {r: compute_tp_fp_fn_tn(region_or_label_to_mask(seg_ref, r), region_or_label_to_mask(seg_pred, r),
                                   ignore_mask)
            for r in labels_or_regions}


def reference_metrics(seg_ref, seg_pred, labels_or_regions, ignore_label=None):
    metrics = {}
    for r, (tp, fp, fn, tn) in reference_counts(seg_ref, seg_pred, labels_or_regions, ignore_label).items():
        metrics[r] = {
            'Dice': np.nan if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn),
            'IoU': np.nan if tp + fp + fn == 0 else tp / (tp + fp + fn),
            'FP': fp, 'TP': tp, 'FN': fn, 'TN': tn, 'n_pred': fp + tp, 'n_ref': fn + tp,
        }
    return metrics


def random_segmentation_pair(rng, shape, values, block=4, noise=0.1, ignore_label=None, ignore_fraction=0.15,
                             dtype=np.float32):
    """
    Blocky reference so that the labels form contiguous structures, prediction = reference with a fraction of the
    voxels replaced by random values. If ignore_label is given, blocks of the reference are set to it (as in sparse
    annotations) and the prediction contains it at random locations as well (a network must not predict it, but
    evaluation must not care either way).
    """
    values = np.array(values)
    coarse_shape = [max(1, int(np.ceil(s / block))) for s in shape]
    coarse = rng.choice(values, size=coarse_shape)
    if ignore_label is not None:
        coarse[rng.random(coarse_shape) < ignore_fraction] = ignore_label
    ref = np.kron(coarse, np.ones([block] * len(shape), dtype=coarse.dtype))[tuple(slice(0, s) for s in shape)]
    pred = ref.copy()
    if ignore_label is not None:
        # the prediction cannot know where the ignored voxels are
        pred[pred == ignore_label] = rng.choice(values, size=int(np.sum(pred == ignore_label)))
    flip = rng.random(shape) < noise
    pred_values = np.append(values, ignore_label) if ignore_label is not None else values
    pred[flip] = rng.choice(pred_values, size=int(flip.sum()))
    return ref.astype(dtype), pred.astype(dtype)


def brute_force_confusion_matrix(seg_ref, seg_pred, labels, ignore_label=None):
    n = len(labels)
    to_bin = lambda v: labels.index(v) if v in labels else n
    cm = np.zeros((n + 1, n + 1), dtype=np.int64)
    for r, p in zip(seg_ref.ravel().tolist(), seg_pred.ravel().tolist()):
        if ignore_label is not None and r == ignore_label:
            continue
        cm[to_bin(r), to_bin(p)] += 1
    return cm


# BraTS style: nested regions
BRATS_LABELS = {'background': 0, 'whole_tumor': [1, 2, 3], 'tumor_core': [2, 3], 'enhancing_tumor': 3}
# overlapping but not nested regions, a single label region written as int and one written as list
OVERLAPPING_LABELS = {'background': 0, 'a': [1, 2], 'b': [2, 3], 'c': 4, 'd': [5], 'ab': [1, 2, 3], 'e': [4, 5]}


class TestConfusionMatrix(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        for labels, ignore_label in [([1, 2, 3], None), ([1, 2, 3], 4), ([0, 1, 2, 3], 4), ([2, 7], 3),
                                     ([5], None), ([1, 3], 3), ([0], 0)]:
            with self.subTest(labels=labels, ignore_label=ignore_label):
                ref, pred = random_segmentation_pair(rng, (1, 9, 11, 13), list(range(0, 9)),
                                                     ignore_label=ignore_label)
                np.testing.assert_array_equal(compute_confusion_matrix(ref, pred, labels, ignore_label),
                                              brute_force_confusion_matrix(ref, pred, labels, ignore_label))

    def test_chunk_boundaries(self):
        rng = np.random.default_rng(1)
        for ignore_label in [None, 6]:
            ref, pred = random_segmentation_pair(rng, (1, 17, 19, 23), list(range(0, 6)), ignore_label=ignore_label)
            expected = brute_force_confusion_matrix(ref, pred, [1, 2, 3, 4, 5], ignore_label)
            for chunk_size in [1, 2, 7, 1000, ref.size - 1, ref.size, ref.size + 1, 2 ** 22]:
                with self.subTest(chunk_size=chunk_size, ignore_label=ignore_label):
                    np.testing.assert_array_equal(
                        compute_confusion_matrix(ref, pred, [1, 2, 3, 4, 5], ignore_label, chunk_size), expected)
                    # the counts must cover every voxel that is not ignored
                    self.assertEqual(compute_confusion_matrix(ref, pred, [1, 2, 3, 4, 5], ignore_label,
                                                              chunk_size).sum(),
                                     ref.size - (np.sum(ref == ignore_label) if ignore_label is not None else 0))

    def test_values_outside_lut_range(self):
        # values below the smallest and above the largest label (including negative ones and ones close to the dtype
        # limits) must end up in the 'other' bin
        rng = np.random.default_rng(2)
        for dtype, values in [(np.int16, [-32768, -5, -1, 0, 3, 4, 9, 32767]),
                              (np.uint16, [0, 3, 4, 9, 65535]),
                              (np.int64, [-2 ** 40, 0, 3, 4, 9, 2 ** 40]),
                              (np.float32, [-1000, -1, 0, 3, 4, 9, 1e6])]:
            with self.subTest(dtype=dtype):
                ref, pred = random_segmentation_pair(rng, (1, 8, 8, 8), values, dtype=dtype)
                np.testing.assert_array_equal(compute_confusion_matrix(ref, pred, [3, 4], 9),
                                              brute_force_confusion_matrix(ref, pred, [3, 4], 9))

    def test_no_labels(self):
        ref = np.zeros((1, 4, 4, 4), dtype=np.float32)
        np.testing.assert_array_equal(compute_confusion_matrix(ref, ref, []), [[64]])
        ref[0, 0] = 1
        np.testing.assert_array_equal(compute_confusion_matrix(ref, ref, [], ignore_label=1), [[48]])

    def test_rejects_invalid_input(self):
        seg = np.zeros((1, 4, 4, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_confusion_matrix(seg, np.zeros((1, 4, 4, 5), dtype=np.float32), [1])
        non_integer = seg.copy()
        non_integer[0, 0, 0, 0] = 1.5
        with self.assertRaises(ValueError):
            compute_confusion_matrix(seg, non_integer, [1])
        with self.assertRaises(ValueError):
            compute_confusion_matrix(non_integer, seg, [1])
        with self.assertRaises(AssertionError):
            compute_confusion_matrix(seg, seg, [1, 1])


class TestCountsMatchPreviousImplementation(unittest.TestCase):
    def assert_counts_equal(self, seg_ref, seg_pred, labels_or_regions, ignore_label):
        expected = reference_counts(seg_ref, seg_pred, labels_or_regions, ignore_label)
        actual = compute_tp_fp_fn_tn_for_labels_or_regions(seg_ref, seg_pred, labels_or_regions, ignore_label)
        self.assertEqual(list(actual.keys()), list(expected.keys()))
        for r in labels_or_regions:
            self.assertEqual(tuple(int(i) for i in actual[r]), tuple(int(i) for i in expected[r]), msg=f'{r}')

    def test_labels(self):
        rng = np.random.default_rng(3)
        for seed, num_labels, shape in itertools.product(range(3), [1, 2, 5, 30], [(1, 40, 50), (1, 20, 24, 28)]):
            with self.subTest(seed=seed, num_labels=num_labels, shape=shape):
                ref, pred = random_segmentation_pair(rng, shape, list(range(num_labels + 1)))
                self.assert_counts_equal(ref, pred, list(range(1, num_labels + 1)), None)

    def test_labels_with_ignore_label(self):
        rng = np.random.default_rng(4)
        for seed, num_labels in itertools.product(range(3), [1, 2, 5, 30]):
            with self.subTest(seed=seed, num_labels=num_labels):
                # nnU-Net convention: ignore label is the highest value
                ref, pred = random_segmentation_pair(rng, (1, 20, 24, 28), list(range(num_labels + 1)),
                                                     ignore_label=num_labels + 1)
                self.assert_counts_equal(ref, pred, list(range(1, num_labels + 1)), num_labels + 1)

    def test_ignore_label_in_unusual_places(self):
        rng = np.random.default_rng(5)
        # ignore label between the labels, below them, and itself listed as a label (it is then always empty in the
        # reference but can still have false positives)
        for labels, ignore_label in [([1, 3, 4], 2), ([2, 3], 1), ([1, 2, 3], 3), ([1, 2], 0)]:
            with self.subTest(labels=labels, ignore_label=ignore_label):
                ref, pred = random_segmentation_pair(rng, (1, 20, 24, 28), [0] + labels, ignore_label=ignore_label)
                self.assert_counts_equal(ref, pred, labels, ignore_label)

    def test_dtypes(self):
        rng = np.random.default_rng(6)
        for dtype in [np.uint8, np.uint16, np.int16, np.int32, np.int64, np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                ref, pred = random_segmentation_pair(rng, (1, 20, 24, 28), list(range(6)), ignore_label=6,
                                                     dtype=dtype)
                self.assert_counts_equal(ref, pred, [1, 2, 3, 4, 5], 6)
                self.assert_counts_equal(ref, pred, [(1, 2, 3), (2, 3), (3,), 4], 6)

    def test_stray_and_missing_values(self):
        rng = np.random.default_rng(7)
        # the data contains values we do not evaluate (7, 200, 255) and we evaluate labels that are absent (9, 10)
        ref, pred = random_segmentation_pair(rng, (1, 20, 24, 28), [0, 1, 2, 7, 200, 255], ignore_label=3)
        for labels_or_regions in [[1, 2, 9, 10], [(1, 2), (9, 10), (2, 9), 7]]:
            with self.subTest(labels_or_regions=labels_or_regions):
                self.assert_counts_equal(ref, pred, labels_or_regions, 3)
                self.assert_counts_equal(ref, pred, labels_or_regions, None)

    def test_sparse_label_values(self):
        rng = np.random.default_rng(8)
        ref, pred = random_segmentation_pair(rng, (1, 20, 24, 28), [0, 2, 10, 1000, 60000], ignore_label=30000,
                                             dtype=np.uint16)
        self.assert_counts_equal(ref, pred, [2, 10, 1000, 60000], 30000)
        self.assert_counts_equal(ref, pred, [(2, 1000), (10, 60000), (2, 10, 1000, 60000)], 30000)

    def test_regions(self):
        rng = np.random.default_rng(9)
        for label_dict in [BRATS_LABELS, OVERLAPPING_LABELS]:
            for use_ignore, seed in itertools.product([False, True], range(3)):
                with self.subTest(label_dict=label_dict, use_ignore=use_ignore, seed=seed):
                    ld = dict(label_dict)
                    if use_ignore:
                        ld['ignore'] = max(max(v) if isinstance(v, list) else v for v in label_dict.values()) + 1
                    lm = LabelManager(ld, regions_class_order=list(range(1, len(label_dict))))
                    self.assertTrue(lm.has_regions)
                    self.assertEqual(lm.has_ignore_label, use_ignore)
                    ref, pred = random_segmentation_pair(rng, (1, 20, 24, 28), lm.all_labels,
                                                         ignore_label=lm.ignore_label)
                    # exactly what the trainer passes
                    self.assert_counts_equal(ref, pred, lm.foreground_regions, lm.ignore_label)
                    # and label based evaluation of the same data
                    self.assert_counts_equal(ref, pred, lm.filter_background(lm.all_labels), lm.ignore_label)

    def test_unusual_regions(self):
        rng = np.random.default_rng(10)
        ref, pred = random_segmentation_pair(rng, (1, 20, 24, 28), list(range(5)), ignore_label=5)
        # region including background, a label listed twice, a region consisting of all labels, a single label as
        # tuple and as int, numpy ints, and a region that includes the ignore label
        regions = [(0, 1), (1, 1, 2), (0, 1, 2, 3, 4), (3,), 3, np.int64(4), (np.int64(2), np.int64(4)), (4, 5)]
        for ignore_label in [None, 5]:
            with self.subTest(ignore_label=ignore_label):
                self.assert_counts_equal(ref, pred, regions, ignore_label)

    def test_empty_segmentations(self):
        ref = np.zeros((1, 10, 10, 10), dtype=np.float32)
        pred = ref.copy()
        self.assert_counts_equal(ref, pred, [1, 2, (1, 2)], None)
        ref[0, :5] = 3
        self.assert_counts_equal(ref, pred, [1, 2, (1, 2)], 3)
        # everything ignored
        self.assert_counts_equal(np.full_like(ref, 3), pred, [1, 2, (1, 2)], 3)


class TestComputeMetricsOnFolder(unittest.TestCase):
    def _run(self, label_dict, regions_class_order):
        lm = LabelManager(label_dict, regions_class_order=regions_class_order)
        labels_or_regions = lm.foreground_regions if lm.has_regions else lm.foreground_labels
        rng = np.random.default_rng(11)
        rw = SimpleITKIO()
        with TemporaryDirectory() as folder_ref, TemporaryDirectory() as folder_pred:
            for i in range(6):
                ref, pred = random_segmentation_pair(rng, (18, 20, 22), lm.all_labels, ignore_label=lm.ignore_label,
                                                     dtype=np.uint8)
                if i == 0:
                    # a case in which some labels are missing entirely, so that some Dice scores are nan
                    ref[ref != lm.ignore_label] = 0
                    pred[:] = 0
                sitk.WriteImage(sitk.GetImageFromArray(ref), os.path.join(folder_ref, f'case_{i}.nii.gz'))
                sitk.WriteImage(sitk.GetImageFromArray(pred), os.path.join(folder_pred, f'case_{i}.nii.gz'))
            output_file = os.path.join(folder_pred, 'summary.json')
            result = compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, '.nii.gz',
                                               labels_or_regions, lm.ignore_label, num_processes=2, chill=False)

            self.assertEqual(len(result['metric_per_case']), 6)
            per_case_expected = []
            for case in result['metric_per_case']:
                seg_ref = rw.read_seg(case['reference_file'])[0]
                seg_pred = rw.read_seg(case['prediction_file'])[0]
                expected = reference_metrics(seg_ref, seg_pred, labels_or_regions, lm.ignore_label)
                per_case_expected.append(expected)
                self.assertEqual(list(case['metrics'].keys()), list(expected.keys()))
                for r in labels_or_regions:
                    for m, v in expected[r].items():
                        np.testing.assert_equal(case['metrics'][r][m], v, err_msg=f'{r} {m}')
                        # json export must have turned everything into python types
                        self.assertIsInstance(case['metrics'][r][m], float if m in ('Dice', 'IoU') else int)
            self.assertTrue(any(np.isnan(e[r]['Dice']) for e in per_case_expected for r in labels_or_regions))

            for r in labels_or_regions:
                for m in per_case_expected[0][r].keys():
                    np.testing.assert_allclose(result['mean'][r][m],
                                               np.nanmean([e[r][m] for e in per_case_expected]), rtol=1e-12)
            np.testing.assert_allclose(result['foreground_mean']['Dice'],
                                       np.mean([result['mean'][r]['Dice'] for r in labels_or_regions]), rtol=1e-12)

            loaded = load_summary_json(output_file)
            self.assertEqual(set(loaded['mean'].keys()), set(labels_or_regions))
            for r in labels_or_regions:
                np.testing.assert_equal(loaded['mean'][r]['Dice'], result['mean'][r]['Dice'])

            # compute_metrics on a single case agrees too
            single = compute_metrics(os.path.join(folder_ref, 'case_1.nii.gz'),
                                     os.path.join(folder_pred, 'case_1.nii.gz'), rw, labels_or_regions,
                                     lm.ignore_label)
            for r in labels_or_regions:
                for m, v in per_case_expected[1][r].items():
                    np.testing.assert_equal(single['metrics'][r][m], v)

    def test_labels(self):
        self._run({'background': 0, 'a': 1, 'b': 2, 'c': 3}, None)

    def test_labels_with_ignore_label(self):
        self._run({'background': 0, 'a': 1, 'b': 2, 'c': 3, 'ignore': 4}, None)

    def test_regions(self):
        self._run(BRATS_LABELS, [1, 2, 3])

    def test_regions_with_ignore_label(self):
        self._run({**BRATS_LABELS, 'ignore': 4}, [1, 2, 3])

    def test_overlapping_regions_with_ignore_label(self):
        self._run({**OVERLAPPING_LABELS, 'ignore': 6}, list(range(1, len(OVERLAPPING_LABELS))))


if __name__ == '__main__':
    unittest.main()
