# Installation and Setup

This guide consolidates the setup steps needed for a first nnU-Net v2 run.

## 1. Check the basics

- Use Python 3.10 or newer.
- Linux is the primary target, but Windows and macOS are also supported.
- GPU is strongly recommended for training.
- Apple `mps` can be used, but 3D convolutions may still require CPU fallback.

## 2. Install PyTorch first

Install PyTorch for your hardware before installing `nnunetv2`:

<https://pytorch.org/get-started/locally/>

Choose the build that matches your environment:

- `cuda` for NVIDIA GPUs
- `mps` for Apple Silicon
- `cpu` if no accelerator is available

Do not install `nnunetv2` before PyTorch is in place.

## 3. Install nnU-Net

For normal use:

```bash
pip install nnunetv2
```

If you want a local editable checkout for development:

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

On **Intel (non-Apple-Silicon) macOS**, add the `intel_macos` extra to pin compatible
versions of torch and numpy:

```bash
pip install "nnunetv2[intel_macos]"
```

or with an editable install:

```bash
pip install -e ".[intel_macos]"
```

## 4. Create the three storage locations

nnU-Net needs three locations:

- `nnUNet_raw`: raw datasets in nnU-Net format
- `nnUNet_preprocessed`: preprocessed data used during training
- `nnUNet_results`: trained models and installed pretrained models

Recommended layout:

```text
/path/to/nnUNet_raw
/path/to/nnUNet_preprocessed
/path/to/nnUNet_results
```

Inside `nnUNet_raw`, each dataset lives in its own `DatasetXXX_Name` folder.

Example:

```text
nnUNet_raw/
└── Dataset001_MyDataset
    ├── dataset.json
    ├── imagesTr
    ├── imagesTs
    └── labelsTr
```

## 5. Set environment variables

### Linux and macOS

For a persistent setup, add this to your shell profile such as `.bashrc` or `.zshrc`:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

For a temporary setup, run the same commands in the current shell before using nnU-Net.

### Windows PowerShell

```powershell
$Env:nnUNet_raw = "C:/path/to/nnUNet_raw"
$Env:nnUNet_preprocessed = "C:/path/to/nnUNet_preprocessed"
$Env:nnUNet_results = "C:/path/to/nnUNet_results"
```

### Windows Command Prompt

```bat
set nnUNet_raw=C:/path/to/nnUNet_raw
set nnUNet_preprocessed=C:/path/to/nnUNet_preprocessed
set nnUNet_results=C:/path/to/nnUNet_results
```

## 6. Verify the setup

Check that the variables are visible in your shell.

Linux and macOS:

```bash
echo "$nnUNet_raw"
echo "$nnUNet_preprocessed"
echo "$nnUNet_results"
```

PowerShell:

```powershell
echo $Env:nnUNet_raw
echo $Env:nnUNet_preprocessed
echo $Env:nnUNet_results
```

Command Prompt:

```bat
echo %nnUNet_raw%
echo %nnUNet_preprocessed%
echo %nnUNet_results%
```

## 7. Optional extras

`hiddenlayer` enables network topology plots:

```bash
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

If you train on a fast GPU, you may also want to tune `nnUNet_n_proc_DA` for data augmentation throughput.

## 8. Next steps

1. [Prepare a dataset](../how-to/prepare-a-dataset.md)
2. [Plan and preprocess](../how-to/plan-and-preprocess.md)
3. [Train models](../how-to/train-models.md)

## Related reference pages

- [Installation instructions](../installation_instructions.md)
- [Setting up paths](../setting_up_paths.md)
- [How to set environment variables](../set_environment_variables.md)
