# Share Models Trained With a Custom Trainer

If you trained a model with a custom `nnUNetTrainer` subclass, the trainer class must be
importable on any machine that runs inference or continues training from your checkpoint.
This page covers the four ways to make that happen and when each option makes sense.

nnU-Net resolves the trainer class by name from `checkpoint["trainer_name"]`. The resolver
first looks inside `nnunetv2.training.nnUNetTrainer`; if it does not find the class there
and the `nnUNet_extTrainer` environment variable is set, it also searches the directories
listed in that variable. If neither lookup succeeds, inference and training fail.

Pick one of the options below based on your constraints.

## 1. Rename the trainer in the checkpoint to a built-in trainer

Rewrite `trainer_name` in your checkpoint so it points to a trainer that ships with
nnU-Net (typically `nnUNetTrainer`). The weights then load against the built-in trainer
directly and users don't need anything beyond a stock nnU-Net install.

**Prerequisites**
- You did not modify `build_network_architecture` (the network topology must match what
  the built-in trainer produces from the plans).
- You manually verified that inference with the renamed checkpoint reproduces the
  expected predictions.

**When to use**
- You need to obfuscate the trainer (proprietary code you cannot share, e.g.
  nnInteractive).
- You want the setup for users to be as simple as possible; they don't need to do
  anything beyond installing nnU-Net.

**Downsides**
- Reproducibility is lost: the checkpoint no longer points back to the code that
  produced it.
- Requires manual verification on your side that the swap is behaviorally equivalent.

## 2. Ship the trainer separately and use `nnUNet_extTrainer`

Distribute the trainer file(s) alongside the checkpoint. Users set the `nnUNet_extTrainer`
environment variable to point at the directory containing the trainer, and nnU-Net picks
it up at load time without patching anything.

**When to use**
- You need to distribute a custom trainer, e.g. because you modified
  `build_network_architecture` or changed the training procedure.
- You want to stay transparent so users can reproduce what you did.
- You just want to distribute a trained model and don't want to rename it in the
  checkpoint (option 1).

**Downsides**
- Slightly more setup for users (copy the trainer file, set an env variable), which
  may feel unnecessary if the only goal is inference.

**How to use it**

`nnUNet_extTrainer` is a list of directories separated by the OS path separator
(`:` on Linux/macOS, `;` on Windows). Each directory is temporarily added to
`sys.path` and scanned recursively for a module that defines a class with the requested
trainer name.

Example layout:

```text
/opt/my_custom_trainers
└── my_pkg
    ├── __init__.py
    └── my_trainer.py   # defines class MyCustomTrainer(nnUNetTrainer)
```

Linux/macOS:

```bash
export nnUNet_extTrainer="/opt/my_custom_trainers"
```

Multiple directories:

```bash
export nnUNet_extTrainer="/opt/my_custom_trainers:/opt/other_trainers"
```

Windows PowerShell:

```powershell
$Env:nnUNet_extTrainer = "C:/opt/my_custom_trainers"
```

Notes:

- The built-in search runs first. External paths are only consulted if the trainer is
  not found inside `nnunetv2.training.nnUNetTrainer`.
- The trainer class must still be a subclass of `nnUNetTrainer`. This is enforced.
- Any `import`s the trainer makes from your package must be resolvable under the
  directory you set, which is why pointing at the parent of a package (as in the
  example above) is the cleanest setup.

## 3. Editable install plus the trainer copied into the source tree

Users install nnU-Net with `pip install -e .` and then drop the trainer file into
`nnunetv2/training/nnUNetTrainer/` (or a subfolder of it).

**When to use**
- You want to distribute a full development environment in which users can modify and
  iterate on the trainer themselves.

**Downsides**
- Generally not recommended. It is hacky, brittle across nnU-Net versions, and users
  have to re-copy the trainer every time they reinstall.

## 4. Distribute your own nnU-Net fork

Maintain a fork of nnU-Net with the trainer already in the source tree and have users
install your fork instead of upstream.

**When to use**
- You need to distribute a development environment, or your users need to train their
  own models against your setup.
- You are required to release source code (e.g. for a challenge submission).
- You need stability: the fork guarantees that users run exactly the nnU-Net version
  your checkpoint was trained with.
- It may be less effort than option 2 because you can ship a single
  `pip install git+https://...` command.

**Downsides**
- Unless you manually keep the fork in sync with upstream, it gets outdated.

## Choosing between fork, `nnUNet_extTrainer`, and checkpoint hacking

If you require stability and want users to be guaranteed compatibility with your
checkpoint, a fork is the safest way to achieve that. The alternative is to pin the
exact nnU-Net version that users must install, and document it alongside the checkpoint.

For pure inference distribution where you control the model but not the user's
environment, option 1 (rename) is the lowest-friction path when it is applicable, and
option 2 (`nnUNet_extTrainer`) is the right choice when it is not.
