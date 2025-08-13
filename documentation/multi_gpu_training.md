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
into uncompressed .npy files. This is no longer necessayr thanks to a switch to blosc2 files. You can just start all 
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
- **Do not use** `-num_gpus` with `torchrun`. It will be ignored/blocked by the runner.
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
