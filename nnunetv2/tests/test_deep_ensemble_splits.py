import json
import unittest
from copy import deepcopy

from nnunetv2.utilities.deep_ensemble_splits import create_or_update_deep_ensemble_splits


class TestDeepEnsembleSplits(unittest.TestCase):
    def test_creates_cv_splits_and_appends_deep_ensemble_splits(self):
        cases = ["case_003", "case_001", "case_002", "case_004", "case_000", "case_005"]

        splits = create_or_update_deep_ensemble_splits(
            None, cases, num_members=2, num_cv_folds=3, seed=12345)

        self.assertEqual(len(splits), 5)
        self.assertTrue(all("deep_ensemble" not in s for s in splits[:3]))

        all_cases = sorted(cases)
        for member, split in enumerate(splits[3:]):
            self.assertEqual(split["train"], all_cases)
            self.assertEqual(split["val"], all_cases)
            self.assertTrue(split["deep_ensemble"])
            self.assertEqual(split["deep_ensemble_member"], member)

    def test_preserves_existing_non_deep_ensemble_splits(self):
        existing_splits = [
            {"train": ["case_001", "case_002"], "val": ["case_000"], "custom": {"keep": True}},
            {"train": ["case_000", "case_002"], "val": ["case_001"]},
        ]
        expected_existing_splits = deepcopy(existing_splits)

        splits = create_or_update_deep_ensemble_splits(
            existing_splits, ["case_002", "case_000", "case_001"], num_members=2)

        self.assertEqual(splits[:2], expected_existing_splits)
        self.assertEqual(len(splits), 4)
        self.assertTrue(all(s["deep_ensemble"] for s in splits[2:]))
        self.assertEqual(splits[2]["train"], ["case_000", "case_001", "case_002"])

    def test_existing_split_count_does_not_need_to_be_five(self):
        existing_splits = [
            {"train": ["case_001", "case_002"], "val": ["case_000"]},
            {"train": ["case_000", "case_002"], "val": ["case_001"]},
            {"train": ["case_000", "case_001"], "val": ["case_002"]},
        ]
        expected_existing_splits = deepcopy(existing_splits)

        splits = create_or_update_deep_ensemble_splits(
            existing_splits, ["case_002", "case_000", "case_001"], num_members=2, num_cv_folds=5)

        self.assertEqual(splits[:3], expected_existing_splits)
        self.assertEqual(len(splits), 5)
        self.assertEqual(splits[3]["deep_ensemble_member"], 0)
        self.assertEqual(splits[4]["deep_ensemble_member"], 1)
        self.assertEqual(splits[3]["train"], ["case_000", "case_001", "case_002"])
        self.assertEqual(splits[4]["val"], ["case_000", "case_001", "case_002"])

    def test_existing_deep_ensemble_splits_raise_unless_overwrite_enabled(self):
        existing_splits = [
            {"train": ["case_000"], "val": ["case_001"]},
            {"train": ["old"], "val": ["old"], "deep_ensemble": True, "deep_ensemble_member": 7},
            {"train": ["case_001"], "val": ["case_000"], "note": "preserve"},
        ]

        with self.assertRaisesRegex(RuntimeError, "deep ensemble splits"):
            create_or_update_deep_ensemble_splits(existing_splits, ["case_000", "case_001"], num_members=1)

        splits = create_or_update_deep_ensemble_splits(
            existing_splits, ["case_001", "case_000"], num_members=2,
            overwrite_deep_ensemble_splits=True)

        self.assertEqual(splits[:2], [existing_splits[0], existing_splits[2]])
        self.assertEqual(len(splits), 4)
        self.assertEqual(splits[2]["train"], ["case_000", "case_001"])
        self.assertEqual(splits[2]["deep_ensemble_member"], 0)
        self.assertEqual(splits[3]["deep_ensemble_member"], 1)

    def test_deterministic_output_and_sorted_case_identifiers(self):
        cases = ["case_c", "case_a", "case_b", "case_d"]

        first = create_or_update_deep_ensemble_splits(None, cases, num_members=1, num_cv_folds=2, seed=12345)
        second = create_or_update_deep_ensemble_splits(None, cases, num_members=1, num_cv_folds=2, seed=12345)

        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(first[-1]["train"], sorted(cases))
        self.assertEqual(first[-1]["val"], sorted(cases))


if __name__ == "__main__":
    unittest.main()
