# Finetuning from nnssl (self-supervised) checkpoints

This guide describes how to finetune an nnU-Net segmentation model from a checkpoint that
was self-supervised pre-trained with [nnssl](https://github.com/MIC-DKFZ/nnssl) (e.g. the
[OpenMind models on Hugging Face](https://huggingface.co/collections/MIC-DKFZ/openmind-models-6819c21c7fe6f0aaaab7dadf)).

Pre-training on a large unlabeled dataset and then finetuning on your labeled downstream
dataset can improve segmentation performance and reduce training time. nnssl produces the
pre-trained encoder; the tooling described here adapts that checkpoint into an nnU-Net
segmentation network and finetunes it.

> This is an additive extension. It uses its own entrypoints and trainers and does **not**
> change the standard nnU-Net commands. If you are looking for supervised transfer of plans
> between datasets, see [pretraining_and_finetuning.md](pretraining_and_finetuning.md) instead.

## How it works

An nnssl checkpoint (`.pth`) embeds an `nnssl_adaptation_plan`: the architecture it was
trained with, the pre-training patch size/spacing, and the keys that locate the encoder,
stem, input-projection and positional-embedding weights inside the state dict.

The workflow has two steps:

1. **Plan / preprocess.** Read the checkpoint's adaptation plan and write an nnU-Net
   plans file that embeds a `pretrain_info` block (checkpoint path, key mappings, patch
   size, architecture kwargs). Two entrypoints exist depending on the strategy (below).
2. **Train.** `nnUNetv2_train_pretrained` instantiates a `PretrainedTrainer`, which builds
   the network from the plan, loads and **adapts** the pre-trained weights (repeats the
   stem across input channels when they differ, interpolates learnable positional
   embeddings to the new patch size, and, for the dynamic trainer, adapts convolution
   kernel sizes and network depth), **skips the segmentation head**, and finetunes with a
   warmup → train learning-rate schedule.

## Prerequisites

- Environment variables set up: [Setting up paths](setting_up_paths.md) /
  [set_environment_variables.md](set_environment_variables.md).
- A raw dataset in nnU-Net format: [Dataset format](dataset_format.md).
- A pre-trained nnssl checkpoint. This can be either a local `.pth` file, or a Hugging Face
  URL (e.g. `https://huggingface.co/AnonRes/ResEncL-OpenMind-MAE`), which is downloaded
  automatically.

## Choosing a strategy

There are two ways to bring the pre-trained weights into your downstream training:

- **Keep an existing (ResEnc) architecture** (`nnUNetv2_plan_like_dynamic`). An existing
  nnU-Net plan (e.g. a ResEnc preset such as `nnUNetResEncUNetLPlans`) is kept as-is, and
  any mismatch (kernel sizes, depth) between its architecture and the pre-trained weights
  is reconciled at load time. This **requires that base plan to already exist**. Use
  `DynamicPretrainedTrainer`.
- **Adapt the architecture to the checkpoint** (`nnUNetv2_preprocess_like_nnssl`). The
  downstream network is built to match the pre-trained architecture, and the data is
  preprocessed to match the pre-training spacing/normalization. This is the recommended
  default. Use `PretrainedTrainer` (CNN/ResEnc checkpoints) or `PretrainedTrainer_Primusx`
  (Primus transformer checkpoints).

## Workflow A: keep an existing (ResEnc) architecture

This route reuses an existing nnU-Net plan and its architecture, so **the base plan must
already exist**. `nnUNetv2_plan_like_dynamic` reads it and errors out if it is missing. For
the OpenMind ResEnc-L checkpoints, create the ResEnc-L plan first, then layer the
pretraining info on top of it.

```bash
# 1) Create the ResEnc-L plan and preprocess the dataset (this is the base plan for step 2).
#    See documentation/resenc_presets.md for the M/L/XL presets.
nnUNetv2_plan_and_preprocess -d <DATASET_ID> -pl nnUNetPlannerResEncL

# 2) Add pretrain_info on top of the existing ResEnc-L plan. -pl selects the base plan,
#    whose ResEnc architecture is kept; mismatches vs. the checkpoint are handled at load time.
nnUNetv2_plan_like_dynamic \
    -d <DATASET_ID> \
    -n <UniquePretrainingName> \
    -pc <path/to/checkpoint.pth  OR  https://huggingface.co/.../model> \
    -pl nnUNetResEncUNetLPlans

# This writes a plans file named  ptPlans_dynamic__<UniquePretrainingName>

# 3) Finetune with the dynamic trainer.
nnUNetv2_train_pretrained <DATASET_ID> 3d_fullres <FOLD> \
    -p ptPlans_dynamic__<UniquePretrainingName> \
    -tr DynamicPretrainedTrainer
```

> Use the ResEnc preset that matches your checkpoint (`nnUNetPlannerResEncM/L/XL` maps to
> `nnUNetResEncUNetM/L/XLPlans`). If the dataset was already preprocessed for another plan,
> use `nnUNetv2_plan_experiment -d <DATASET_ID> -pl nnUNetPlannerResEncL` in step 1 to skip
> re-preprocessing.

## Workflow B: adapt architecture to the checkpoint 

The downstream network is built to match the checkpoint's architecture and the data is
preprocessed to match the pre-training. This route is **self-contained**: you do **not**
need to run planning first; the command below does fingerprint extraction and experiment
planning for you before adapting.

```bash
# 1) Create the plan and preprocess the dataset to match the pre-training.
#    No prior planning step is needed: this runs fingerprint extraction and
#    experiment planning internally, then adapts everything to the checkpoint.
nnUNetv2_preprocess_like_nnssl \
    -d <DATASET_ID> \
    -n <UniquePretrainingName> \
    -pc <path/to/checkpoint.pth  OR  https://huggingface.co/.../model> \
    -am like_pretrained

# This writes a plans file named  ptPlans__<UniquePretrainingName>...
# It contains everything needed (weights path, key mappings, architecture),
# so you no longer need to pass the checkpoint at training time.

# 2) Finetune. Note: dataset / configuration / fold are POSITIONAL arguments.
nnUNetv2_train_pretrained <DATASET_ID> 3d_fullres <FOLD> \
    -p ptPlans__<UniquePretrainingName>... \
    -tr PretrainedTrainer
```

For a Primus (transformer) checkpoint, use `-tr PretrainedTrainer_Primusx` in step 2.

### Adaptation modes (`-am`)

Controls the target spacing used to preprocess the downstream data:

| Mode | Behaviour |
|------|-----------|
| `like_pretrained` | Use the pre-training spacing (often 1×1×1 mm isotropic). **Recommended default.** |
| `default_nnunet` | Use nnU-Net's usual median-spacing plan for the dataset. Prefer this when your data has much smaller spacings than the pre-training. |
| `fixed` | Use an explicit spacing you provide via `-os X Y Z`. |
| `no_resample` | Do not resample (spacing left as-is). |

## Available trainers

Selected with `-tr`. Each family also ships preset variants (e.g. `_150ep`, `_300ep`,
`_nomirroring`, `_smallerlr`, Adam-based variants). Run the command and look at the
class names in `nnunetv2/training/nnUNetTrainer/pretraining/` for the full list.

- `PretrainedTrainer`. Base finetuning trainer (CNN / ResEnc checkpoints), warmup → train
  schedule.
- `PretrainedTrainer_Primusx`. For Primus transformer checkpoints.
- `DynamicPretrainedTrainer` (and `DynamicPretrainedTrainer_adam_150ep`). Reconciles
  architecture mismatches (kernel sizes, depth) against an existing nnU-Net/ResEnc plan.

## Citing the pre-trained checkpoint

nnssl checkpoints carry the citations of the papers behind the pre-trained weights. These
are read from the checkpoint, stored inside the generated plan (`pretrain_info`), and
printed to the training log at the start of training. When you publish results obtained by
finetuning a pre-trained checkpoint, please cite the papers listed there.

## Training a "from scratch" baseline

To train with the **same configuration** but **without** loading the pre-trained weights
(a fair baseline), add `--from_scratch`. This uses a separate `__from_scratch` plan name
and skips weight loading:

```bash
nnUNetv2_train_pretrained <DATASET_ID> 3d_fullres <FOLD> \
    -p ptPlans__<UniquePretrainingName>... -tr PretrainedTrainer --from_scratch
```

## Acknowledgements

Developed by the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php)
at the [German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html), as part
of The Human Radiome Project ([THRP](https://hfmi.helmholtz.de/pilot-projects/thrp/)) of the
Helmholtz Foundation Model Initiative. Original authors: Constantin Ulrich & Tassilo Wald.
