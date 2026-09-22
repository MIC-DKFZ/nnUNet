# Working in the nnU-Net repository

Notes for coding agents and new contributors. This file is about *how to change this codebase safely*;
it does not duplicate the user documentation.

- **Users start here:** [`readme.md`](readme.md) and [`documentation/`](documentation/).
- **Contributors start here:** [`CONTRIBUTING.md`](CONTRIBUTING.md). It says what we do and do not accept.
- **Deliberately deferred simplifications:** [`DEFERRED_CLEANUPS.md`](DEFERRED_CLEANUPS.md).

## Before you merge a branch into master

This file ships to everyone who clones nnU-Net, so it has to read as guidance about *the codebase*, not as a
record of whatever branch it came from. Feature branches accumulate the opposite: cluster and host names, job
ids, absolute paths on someone's machine, links to internal reports, and status notes about work in progress.
That is fine while the branch is alive — it is how the notes travel between machines — but **none of it may
reach master**.

So when you merge, re-read `CLAUDE.md` (and `AGENTS.md`, if present) and strip anything internal or
branch-specific before the merge lands: paths, infrastructure names, internal links, and anything that is
about one experiment rather than about working in this repository. If a note is only true for your branch, it
belongs on your branch. Keep the branch history out of it too — squash, or drop the commits that carried the
internal notes, because cleaning the file does not clean the commits that introduced it.

## The one rule that matters most

**nnU-Net is used by a lot of people, and much of that use happens in custom trainers and custom code that
lives outside this repository.** They subclass `nnUNetTrainer`, read its attributes, import our utilities and
parse our output files. A rename or a signature change that looks harmless here silently breaks them, and
they will not find out until a training crashes hours in.

So: prefer additive changes. If a change does break third-party code, it needs an entry in
[`documentation/changelog.md`](documentation/changelog.md) under "Changes that affect custom trainers",
spelling out the old and the new way. If the right cleanup cannot be done without breaking people, do not do
it quietly and do not leave it in a commit message — add it to `DEFERRED_CLEANUPS.md` under the release that
can execute it, saying what to do, why it is deferred and what breaks.

## Environment variables

nnU-Net locates everything through three variables; nothing works without them. See
[`documentation/set_environment_variables.md`](documentation/set_environment_variables.md) and
[`documentation/setting_up_paths.md`](documentation/setting_up_paths.md).

| variable | holds |
|---|---|
| `nnUNet_raw` | datasets in nnU-Net raw format |
| `nnUNet_preprocessed` | fingerprints, plans, preprocessed data |
| `nnUNet_results` | trained models, logs, checkpoints, validation output |

`nnUNet_compile`, `nnUNet_n_proc_DA` and friends are optional and documented in the same place.

## Layout

| path | what |
|---|---|
| `nnunetv2/experiment_planning/` | fingerprint extraction, planners, preprocessing |
| `nnunetv2/training/nnUNetTrainer/` | `nnUNetTrainer` and its variants. Most extension happens here. |
| `nnunetv2/training/` | loss functions, dataloading, data augmentation, LR schedules |
| `nnunetv2/inference/` | sliding-window prediction, export |
| `nnunetv2/evaluation/` | metrics, best-configuration selection |
| `nnunetv2/run/` | `nnUNetv2_train` and the DDP launch paths |
| `nnunetv2/utilities/` | shared helpers, including `ddp.py` |
| `nnunetv2/tests/` | unit tests and `integration_tests/` |

Every user-facing command is a console entry point declared in `pyproject.toml` — that table is the quickest
map from a command name to the code behind it.

## Distributed training: ranks

DDP runs on one node (`-num_gpus X`, where nnU-Net spawns the workers itself) or across several
(`torchrun`, which spawns them). Both funnel into the same code path, so code below the launcher never needs
to know which was used. User-facing details: [`documentation/multi_gpu_training.md`](documentation/multi_gpu_training.md).

`get_ddp_topology()` in `nnunetv2/utilities/ddp.py` provides four values, and the distinction is load-bearing:

| attribute | meaning |
|---|---|
| `self.local_rank` | index within the node → **the CUDA device index** |
| `self.global_rank` | index within the whole job → **decides who writes files** |
| `self.world_size` | processes in the job |
| `self.local_world_size` | processes on this node |

**Guard every write to `nnUNet_results` with `global_rank == 0`.** On one node the two ranks coincide, so a
mistake here is invisible until someone trains across nodes: there is one `local_rank == 0` *per node*, and on
a shared filesystem they all write the same log file, checkpoints and progress plot. Use `local_rank` only
where you mean the GPU.

Other things that are easy to get wrong:

- Collectives must be reached by **all** ranks. A `return` or an exception inside a rank-0 branch that skips a
  barrier hangs the job until the NCCL watchdog fires.
- Order barriers so that ranks wait *before* reading what other ranks wrote, not after.
- `SIGUSR1` sets a flag that is reduced across the job at the epoch boundary, so one signalled rank stops all
  of them. Do not log from the signal handler.

## Tests

```bash
pytest nnunetv2/tests                     # unit tests, fast
nnunetv2/tests/integration_tests/run_integration_test.sh            # full pipeline, slow
nnunetv2/tests/integration_tests/run_integration_test_trainingOnly_DDP.sh   # DDP path
```

Integration tests need the environment variables above and real data; see
`nnunetv2/tests/integration_tests/readme.md`. Anything touching planning, preprocessing or training should be
checked against them, because unit tests do not cover the pipeline end to end.

## Conventions

- Match the surrounding style rather than imposing a new one; the codebase predates most current formatters
  and a reformat-in-passing buries the actual change in a large diff.
- Training must stay reproducible: seeds, plans and preprocessed data determine the result. A change that
  moves segmentation quality needs numbers against the previous behaviour, not an argument that it should be
  equivalent.
- Performance claims need a measurement. `documentation/benchmarking.md` and
  `nnUNetTrainerBenchmark_5epochs` exist for this; compare the fastest epoch, not the mean, and never compare
  across different GPU models.
