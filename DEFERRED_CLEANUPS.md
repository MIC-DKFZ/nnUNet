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
