# Deep Ensemble with nnU-Net

nnU-Net's standard cross-validation folds are useful for model selection, performance estimation, and test-time ensembling. However, a cross-validation (CV) ensemble is not the same as a deep ensemble (DE) for uncertainty estimation [1].

In a deep ensemble, all members are trained on the same training set, but with different random initializations and stochastic training trajectories [2]. In a CV ensemble, individual models are trained on different training subsets. As a result, CV ensemble disagreement reflects not only model uncertainty, but also differences caused by incomplete or varying data exposure [1].

`nnUNetv2_create_deep_ensemble_splits` provides a lightweight way to train DE members within the existing nnU-Net fold infrastructure. It appends full-training-set folds to `splits_final.json`. These folds can then be trained with the standard command:

```bash
nnUNetv2_train DATASET CONFIGURATION FOLD
```

This utility does not change nnU-Net training, checkpointing, inference defaults, or model selection defaults.

## Workflow

### 1. Plan and preprocess the dataset

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### 2. Create cross-validation and deep ensemble splits

Create the standard CV folds, if needed, and append deep ensemble folds:

```bash
nnUNetv2_create_deep_ensemble_splits DATASET_ID 3d_fullres --num_members 5
```

This preserves existing non-deep-ensemble splits and appends the requested number of deep ensemble folds.

### 3. Train the CV folds

Use the standard CV folds for model selection and performance estimation:

```bash
for f in 0 1 2 3 4; do
    nnUNetv2_train DATASET_ID 3d_fullres $f --npz
done
```

### 4. Run model selection on CV folds only

```bash
nnUNetv2_find_best_configuration DATASET_ID -f 0 1 2 3 4
```

Do not include DE folds in model selection, because they are trained on the full training set.

### 5. Train the deep ensemble folds

For example, if the dataset has five standard CV folds and five deep ensemble folds were appended, the deep ensemble folds will be `5 6 7 8 9`. These indices depend on the number of existing non-deep-ensemble folds; see [Fold indexing](#fold-indexing) below.

```bash
for f in 5 6 7 8 9; do
    nnUNetv2_train DATASET_ID 3d_fullres $f
done
```

### 6. Predict with the deep ensemble

Explicitly select the deep ensemble folds at inference time:

```bash
nnUNetv2_predict \
    -d DATASET_ID \
    -c 3d_fullres \
    -i INPUT_FOLDER \
    -o OUTPUT_FOLDER \
    -f 5 6 7 8 9
```

## Fold indexing

If there are already `K` existing non-deep-ensemble folds and you request `N` deep ensemble members, the appended deep ensemble folds will be:

```text
K, K + 1, ..., K + N - 1
```

For example:

- existing CV folds: `0 1 2 3 4`
- requested deep ensemble members: `N = 3`
- appended deep ensemble folds: `5 6 7`

The `deep_ensemble_member` metadata stored in `splits_final.json` is 0-based.

## Existing split files

Existing split files do not need to contain exactly five folds.

If `splits_final.json` already exists, all non-deep-ensemble splits are preserved exactly. In this case, `--num_cv_folds` is ignored. The `--num_cv_folds` argument is only used when no split file exists and the utility needs to create the initial cross-validation splits.

Existing deep ensemble splits are not overwritten by default. To replace them, use `--overwrite_deep_ensemble_splits`.

## Important validation warning

Deep ensemble folds use all available training cases for both `train` and `val` in `splits_final.json`. This is done for compatibility with the existing nnU-Net fold pipeline.

Validation metrics from these folds are biased and must **not** be used for model selection or performance estimation.

Use:

- standard CV folds for model selection and performance estimation;
- deep ensemble folds for final full-data ensemble training.

## Inference

This utility does not change nnU-Net inference behavior. `nnUNetv2_predict` will not automatically use deep ensemble folds. The desired folds must be passed explicitly with `-f`.

For example:

```bash
nnUNetv2_predict \
    -d DATASET_ID \
    -c 3d_fullres \
    -i INPUT_FOLDER \
    -o OUTPUT_FOLDER \
    -f 5 6 7 8 9
```

*If you need per-member predictions or probabilities for custom uncertainty analysis, run prediction separately for each deep ensemble fold and aggregate the outputs externally.*

## Recommendation

Deep ensembles and CV ensembles should not be treated as interchangeable uncertainty estimators [1]. Use CV folds for model selection and unbiased performance estimation. Use deep ensemble folds when the goal is to train a final full-data ensemble for uncertainty estimation.

## Citation

If you use Deep Ensemble within nnU-Net in your work, please cite [1]:

```bibtex
@misc{kirscher2026lostinfolds,
      title={Lost in the Folds: When Cross-Validation Is Not a Deep Ensemble for Uncertainty Estimation}, 
      author={Tristan Kirscher and Markus Bujotzek and Yannick Kirchhoff and Maximilian Rokuss and Fabian Isensee and Kim-Celine Kahl and Balint Kovacs and Klaus Maier-Hein},
      year={2026},
      eprint={2605.18329},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.18329}, 
}
```

## References

[1] Kirscher et al. *Lost in the Folds: When Cross-Validation Is Not a Deep Ensemble for Uncertainty Estimation*. MICCAI, 2026.

[2] Lakshminarayanan et al. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS, 2017.
