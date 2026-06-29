# Run Inference

This guide covers prediction, optional ensembling, and postprocessing.

## Before you start

Input images must match the trained dataset's naming convention and file endings. See:

- [Dataset and input format reference](../reference/dataset-format.md)

If you previously ran `nnUNetv2_find_best_configuration`, use the commands it generated in `inference_instructions.txt` whenever possible.

## Predict with a trained configuration

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION
```

If you want to ensemble probability outputs from multiple configurations, add `--save_probabilities`:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
```

By default, inference uses the 5 trained folds as an ensemble. If you trained the `all` fold and want to use only that model:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION -f all
```

## Ensemble multiple configuration outputs

```bash
nnUNetv2_ensemble -i FOLDER1 FOLDER2 -o OUTPUT_FOLDER -np NUM_PROCESSES
```

The input folders must contain probability files produced with `--save_probabilities`.

## Apply postprocessing

```bash
nnUNetv2_apply_postprocessing \
  -i FOLDER_WITH_PREDICTIONS \
  -o OUTPUT_FOLDER \
  --pp_pkl_file POSTPROCESSING_FILE \
  -plans_json PLANS_FILE \
  -dataset_json DATASET_JSON_FILE
```

For single-configuration predictions, `plans.json` and `dataset.json` are usually copied automatically. For ensemble outputs, provide them explicitly.

## Predict from a model folder

If you want to run inference directly from an exported or copied model folder:

```bash
nnUNetv2_predict_from_modelfolder -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```

## Export and import your own trained model

To move a trained model to another machine:

1. Export it:

```bash
nnUNetv2_export_model_to_zip -d DATASET_NAME_OR_ID -o MODEL.zip
```

2. Install it on the target machine:

```bash
nnUNetv2_install_pretrained_model_from_zip MODEL.zip
```

The target machine still needs a compatible nnU-Net installation and all dependencies.

If the model was trained with a custom `nnUNetTrainer` subclass, the target machine also
needs that trainer class to be importable. See:

- [Share models trained with a custom trainer](share-models-with-custom-trainers.md)

## Public pretrained models

The old page on pretrained-model inference remains here:

- [How to run inference with pretrained models](../run_inference_with_pretrained_models.md)

Check that page for the current status before relying on it.
