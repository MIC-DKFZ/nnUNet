# CLI Overview

This page groups the main nnU-Net v2 command-line entry points by workflow stage.

## Dataset preparation

- `nnUNetv2_convert_old_nnUNet_dataset`: convert nnU-Net v1 datasets
- `nnUNetv2_convert_MSD_dataset`: convert Medical Segmentation Decathlon datasets

## Planning and preprocessing

- `nnUNetv2_plan_and_preprocess`: run fingerprinting, planning, and preprocessing in one step
- `nnUNetv2_extract_fingerprint`: fingerprint only
- `nnUNetv2_plan_experiment`: planning only
- `nnUNetv2_preprocess`: preprocessing only

## Training

- `nnUNetv2_train`: train a configuration and fold

## Inference

- `nnUNetv2_predict`: run prediction using dataset id and stored results
- `nnUNetv2_predict_from_modelfolder`: run prediction from an explicit model folder
- `nnUNetv2_ensemble`: ensemble multiple prediction folders
- `nnUNetv2_apply_postprocessing`: apply postprocessing to prediction outputs

## Evaluation and model selection

- `nnUNetv2_find_best_configuration`: compare configurations and determine postprocessing
- `nnUNetv2_accumulate_crossval_results`: aggregate cross-validation results
- `nnUNetv2_determine_postprocessing`: determine postprocessing separately
- `nnUNetv2_evaluate_folder`: evaluate a prediction folder
- `nnUNetv2_evaluate_simple`: simpler evaluation entry point

## Model packaging and sharing

- `nnUNetv2_export_model_to_zip`: export a trained model bundle
- `nnUNetv2_install_pretrained_model_from_zip`: install a model bundle from zip
- `nnUNetv2_download_pretrained_model_by_url`: install a model bundle from a URL

## Utilities

- `nnUNetv2_move_plans_between_datasets`: reuse plans across datasets
- `nnUNetv2_plot_overlay_pngs`: create overlay PNGs for visualization

## Help and discovery

All commands support `-h`:

```bash
nnUNetv2_train -h
nnUNetv2_predict -h
nnUNetv2_plan_and_preprocess -h
```

## Recommended starting path

If you are new to nnU-Net, do not start from this page. Start with:

- [Installation and setup](../getting-started/installation-and-setup.md)
- [Prepare a dataset](../how-to/prepare-a-dataset.md)
- [Plan and preprocess](../how-to/plan-and-preprocess.md)
- [Train models](../how-to/train-models.md)
