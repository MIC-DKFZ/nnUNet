# Deferred cleanups

Simplifications we know we want but cannot do without breaking something for users or for custom trainers
outside this repository. Rather than losing them in commit messages and PR threads, they are collected here so
they can be executed in one pass when a major version lets us break compatibility.

**This is not a TODO list.** Only entries that are *deliberately deferred* belong here: we know what the right
shape is, we know why we are not doing it yet, and we know what would break. Ordinary bugs and unfinished work
go to the issue tracker.

**Adding an entry.** Add it in the same PR as the change that created the debt, under the release that can
execute it. Say what to do, why it is deferred, and what breaks — the last one is what a future reader cannot
reconstruct.

**Executing entries.** When the blocking release arrives, work through the section top to bottom, delete each
entry as it lands, and mention the removals in `documentation/changelog.md`.

---

## For nnU-Net v3

### Make the streaming predictor the default, in inference and in validation

Streaming inference (`nnUNetPredictorStreaming`) never builds the `(num_classes, *image_shape)` volume. Today it is
opt-in: `nnUNetv2_predict_streaming` and the `_SV` trainers. `nnUNetv2_predict` and the final validation of
`nnUNetv2_train` still use the predictor that does build it.

- **Do:**
  - Commands: `nnUNetv2_predict` runs the streaming predictor, `nnUNetv2_predict_legacy` the current one, and
    `nnUNetv2_predict_streaming` goes away. The `nnUNetPredictor` class stays as it is.
  - Validation: the trainer's final validation streams. The `_SV` trainers stay as aliases.
  - Fall back to the current predictor, with a notice, where streaming cannot reproduce it:
    - cascades;
    - probability resampling functions streaming does not implement (the notice says what to change to stream);
    - label managers that override `convert_logits_to_segmentation`, `convert_probabilities_to_segmentation` or
      `apply_inference_nonlin`, or use a non-standard nonlinearity.
  - Memory: after a GPU out-of-memory error the predictor keeps input, output and ring buffer in RAM for the rest
    of its life. Users can also choose that from the start.
  - Probabilities are `.b2nd` everywhere:
    - `nnUNetv2_train --save_probabilities` writes them (`--npz` stays as a deprecated alias);
    - `nnUNetv2_ensemble` and `nnUNetv2_find_best_configuration` read `.b2nd` and `.npz` and merge slab by slab;
    - in-memory probabilities come back as a blosc2 array (uint16, or float32); dense tensors stay with the
      current predictor.
- **Deferred because:** it changes what the main commands do and write.
  - Segmentations differ from the current predictor's at label boundaries (floating-point rounding).
  - Probability files change from `.npz` to `.b2nd`.
  - A command disappears and the validation flag is renamed.
- **Breaks:**
  - scripts that call `nnUNetv2_predict_streaming` or read `.npz` probabilities from `nnUNetv2_predict` or
    validation;
  - comparisons that expect the current predictor's output bit for bit;
  - custom trainers that rely on the stock `perform_actual_validation` writing `.npz`.

### Collapse the four DDP rank attributes into the DDPTopology they came from

`nnUNetTrainer.__init__` unpacks `get_ddp_topology()` into `self.local_rank`, `self.global_rank`,
`self.world_size` and `self.local_world_size`. Those four attributes *are* the `DDPTopology` NamedTuple, spelled
out one field at a time, and the trainer carries both the parts and no whole.

- **Do:** store `self.ddp_topology = get_ddp_topology()` and read `self.ddp_topology.global_rank` etc. at the
  use sites. In the non-DDP case construct `DDPTopology(0, 0, 1, 1)` instead of assigning four zeros/ones.
- **Deferred because:** `self.local_rank` predates this and custom trainers outside the repository read it
  directly (`if self.local_rank == 0:`). `self.world_size` and `self.global_rank` are new but will have been
  public for a full release cycle by then.
- **Breaks:** every custom trainer that touches any of the four attributes. Needs a changelog entry of the same
  shape as the `local_rank` -> `global_rank` one.

### Remove AllGatherGrad

`nnunetv2/utilities/ddp_allgather.AllGatherGrad` is no longer used anywhere in nnU-Net. The losses use
`AllReduceGrad` (`nnunetv2/utilities/ddp.py`), which computes the same forward and backward while moving
`world_size` times less data — `AllGatherGrad.apply(x).sum(0)` is exactly `AllReduceGrad.apply(x)`.

- **Do:** delete the class. `ddp_allgather.py` contains nothing else, so the module goes with it.
- **Deferred because:** custom losses and trainers outside this repository import it. It currently raises a
  `DeprecationWarning` pointing at the replacement, so by v3 users will have had a release cycle of warning.
- **Breaks:** `from nnunetv2.utilities.ddp_allgather import AllGatherGrad` in third-party code.

### Decide whether `-num_gpus` survives

nnU-Net supports two ways of starting DDP: `-num_gpus X` (we `mp.spawn` the workers, single node only) and
`torchrun` (the launcher spawns them, single or multi node). The first now exists only so that users do not have
to change their launch tooling — internally it emulates the torchrun environment and then follows the identical
code path.

- **Do:** consider dropping `-num_gpus` and documenting `torchrun --nproc_per_node=X` as the single answer. That
  deletes `run_intranode_ddp`, the `MASTER_PORT` search and the launcher-detection branch.
- **Deferred because:** `-num_gpus` is the documented interface, it works from a pip install without a cloned
  repository (torchrun needs a path to `run_training.py`), and removing it would break every existing script.
- **Breaks:** all `-num_gpus` users. Only worth doing if `torchrun` can be invoked via a console entry point so
  the pip-install case keeps working.

### Make the deep supervision weights identical with and without DDP

`nnUNetTrainer._build_loss` sets the lowest deep supervision weight to `0` normally but to `1e-6` under DDP,
because a weight of exactly `0` leaves those parameters out of the backward graph and DDP complains about unused
parameters. So DDP and single-GPU training do not optimize exactly the same objective.

- **Do:** drop the weight to `0` in both cases and deal with the unused parameters properly — either build the
  network without the unused segmentation head, or pass `static_graph=True` to DDP (which makes unused
  parameters legal, at the cost of erroring on trainers whose graph changes between iterations).
- **Deferred because:** it changes the loss, so it needs a Dice-parity run before it can land, and the current
  difference is numerically negligible.
- **Breaks:** nothing structurally; results shift by an immeasurable amount.

### Give the per-rank foreground oversampling its own attribute

`oversample_foreground_percent` is a property: setting it sets the value for the whole batch, and after
`_set_batch_size_and_oversample` has split the batch across DDP ranks, reading it returns this rank's share. One
name for two values, so that the method can be called more than once without splitting a share again.

- **Do:** keep `oversample_foreground_percent` as the plain attribute for the whole batch and store the split in
  `self.oversample_foreground_percent_this_rank` (equal to it without DDP); `get_dataloaders` passes that one to the
  data loaders.
- **Deferred because:** custom trainers outside the repository read `self.oversample_foreground_percent` when they
  build their own data loaders, and would silently get the batch's value instead of their rank's share under DDP.
- **Breaks:** custom `get_dataloaders` (and anything else) that reads `oversample_foreground_percent` after the split
  and expects the rank's share.
