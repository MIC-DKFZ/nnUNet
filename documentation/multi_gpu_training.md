# Multi‑GPU Training with nnU‑Net v2

If multiple GPUs are at your disposal, the **most efficient** way to use them is usually to run **independent nnU‑Net 
trainings in parallel**, one on each GPU. This avoids the scaling inefficiencies of data parallelism, which 
rarely achieves linear speed‑up with the relatively small networks used by nnU‑Net.

Multiple GPUs for training become particularly interesting for custom nnU-Net configurations that exceed the VRAM capacity of 
single GPUs or become prohibitively long to train.

## Recommended: One training per GPU (parallel folds)

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_NAME_OR_ID 2d 0 [--npz] &  # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train DATASET_NAME_OR_ID 2d 1 [--npz] &  # train on GPU 1
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train DATASET_NAME_OR_ID 2d 2 [--npz] &  # train on GPU 2
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train DATASET_NAME_OR_ID 2d 3 [--npz] &  # train on GPU 3
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train DATASET_NAME_OR_ID 2d 4 [--npz] &  # train on GPU 4
wait
```

(or simply run one after the other on the same GPU)

On older versions of nnU-Net one had to wait for the first training to extract the .npz preprocessed training data 
into uncompressed .npy files. This is no longer necessary thanks to a switch to blosc2 files. You can just start all 
trainings at once!

---

## Option 1 — Intra‑node DDP via `-num_gpus`

You can train on multiple GPUs **within a single node** using PyTorch Distributed Data Parallel (DDP) without `torchrun`.
This works with a cloned and installed nnU-Net repository and with the pip installed nnU-Net. 

```bash
# Example: use 2 GPUs on the current machine
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train DATASET_NAME_OR_ID 2d 0 [--npz] -num_gpus 2
```

**Notes for `-num_gpus`:**
1. **GPU selection:** If your node has more GPUs than you want to use, restrict them with `CUDA_VISIBLE_DEVICES`.
2. **Batch size limit:** You cannot use more GPUs than you have samples per minibatch. If `batch_size=2`, `-num_gpus 2` is the maximum.
3. **Divisibility:** Make sure your batch size is divisible by the number of GPUs for efficient use.
4. **Scaling:** DDP can be slower than running separate folds on separate GPUs unless you have increased model, patch, or batch sizes.

Use this option for **quick single‑node DDP** without changing your launch tooling.

---

## Option 2 — Multi‑node / Advanced DDP via `torchrun`

`torchrun` is PyTorch’s standard DDP launcher and supports **both intra‑ and multi‑node** training. With 
`torchrun`, **do not pass** `-num_gpus` — process count and GPU assignment are controlled by `torchrun` itself. 
(Our runner detects `torchrun` via environment variables and initializes/destroys the process group accordingly.)

You need to have cloned and installed the nnU-Net repository for this to work!

### Single node, 4 GPUs
```bash
torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_addr="localhost" --master_port=12345 NNUNET_REPO_LOCATION/nnunetv2/run/run_training.py DATASETID CONFIGURATION FOLD [--npz]
```

### Two nodes, 4 GPUs each
Run **on all nodes** (replace `MASTER_ADDR` with the hostname/IP of rank‑0 node):
```bash
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=YOUR_CUSTOM_INTEGER --rdzv_backend=c10d --rdzv_endpoint=MASTER_ADDR:MASTER_PORT NNUNET_REPO_LOCATION/nnunetv2/run/run_training.py DATASETID CONFIGURATION FOLD [--npz]
```

**Notes for `torchrun`:**
- **Do not use** `-num_gpus` with `torchrun`. nnU-Net detects the launcher and raises an `AssertionError` if you do,
because in this mode the launcher (not nnU-Net) decides how many processes there are and which GPU each one gets.
- You may still use `torchrun` with a **single GPU** (e.g., to debug the full DDP setup). The runner will enter the DDP path when `torchrun` environment variables are present.
- For SLURM clusters, `torchrun` integrates well with `srun`/`sbatch` and multi‑node jobs.

---

## Which option should I use?

- **Best throughput for standard nnU‑Net configs:** run **one training per GPU** (parallel folds) as shown at the top.
- **Need synchronized training across GPUs on one node?** use **`-num_gpus`**.
- **Need multi‑node DDP, or prefer PyTorch’s standard launcher?** use **`torchrun`**.

---

## Troubleshooting tips

- When using `torchrun`, ensure you **omit** `-num_gpus`.
- If using only a subset of GPUs, set `CUDA_VISIBLE_DEVICES` accordingly.
- Batch size must be divisible by the number of GPUs; otherwise you will underutilize hardware or see errors.

---

## Stopping gracefully when a cluster job runs out of time

nnU-Net installs a `SIGUSR1` handler. The signal does not interrupt the running epoch: it sets a flag, and at the
next epoch boundary nnU-Net writes `checkpoint_latest.pth` and exits with status 0. Resubmit with `--c` to continue
exactly where it stopped. (No handler is installed on Windows, which has no `SIGUSR1`.)

Under DDP it is enough for **one** rank to receive the signal. The flag is reduced across the job at every epoch
boundary, so all ranks stop together. Without that, a single signalled rank would exit on its own and leave the
others waiting in a collective until the NCCL watchdog fires.

That matters, because how much of your job actually gets the signal differs per site and per launcher:

- **SLURM:** `#SBATCH --signal=B:USR1@600` sends `SIGUSR1` to the **batch script** 10 minutes before the time limit;
  `--signal=USR1@600` (no `B:`) sends it to the **job step tasks**. Manually, `scancel --signal=USR1 --full JOBID`.
- **LSF:** `bsub -wa USR1 -wt 10` requests a warning signal; manually, `bkill -s USR1 JOBID`.
- **`torchrun`:** it does **not** forward `SIGUSR1` to its workers. Its agent only handles the signals listed in
  `TORCHELASTIC_SIGNALS_TO_HANDLE` (default `SIGTERM,SIGINT,SIGHUP,SIGQUIT`), and those kill the workers rather than
  letting them stop gracefully. Signal the **workers** and not the launcher, i.e. torchrun's children:
  ```bash
  torchrun ... nnunetv2/run/run_training.py DATASETID CONFIGURATION FOLD &
  TORCHRUN_PID=$!
  trap 'pkill -USR1 -P $TORCHRUN_PID' USR1   # -P: only the children, never torchrun itself
  wait $TORCHRUN_PID
  ```
  Do **not** use `pkill -USR1 -f nnunetv2/run/run_training.py`: the launcher's own command line contains the script
  path too, so that also signals torchrun, which dies on the unhandled signal and takes the workers with it before
  they can checkpoint. Because one rank is enough, signalling the workers on any single node is sufficient.

---

## Notes for custom trainers

`self.global_rank` is the process index within the whole job and `self.local_rank` is its index within its node
(and therefore the CUDA device index). **Anything that writes to `nnUNet_results` must be guarded with
`self.global_rank == 0`**, not `self.local_rank == 0` — on a multi-node job there is one `local_rank == 0` per node,
and they would all write the same files. `self.world_size` and `self.local_world_size` are the matching process
counts.
