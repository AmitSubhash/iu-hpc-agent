# Troubleshooting Guide

## Job Failed? Start Here

```
sacct -j JOBID --format=JobID,ExitCode,State,MaxRSS,Elapsed

Exit code format: return_code:signal
  0:0   -> Succeeded (check script logic or .out file)
  1:0   -> Script error (check .err for traceback)
  0:1   -> SIGHUP (lost connection or node issue)
  0:9   -> SIGKILL (OOM or walltime exceeded)
  0:15  -> SIGTERM (preempted, cancelled, or checkpoint signal)
```

---

## Out of Memory

### Host Memory (CPU RAM)

**Symptom:** `slurmstepd: Exceeded job memory limit` or signal 9

```
--mem too low?
  -> sacct -j JOBID --format=MaxRSS  (check actual usage)
  -> Set --mem to 1.5-2x MaxRSS

Used --mem=0?
  -> Always specify explicit --mem. --mem=0 locks entire node.

Multiple processes each loading full dataset?
  -> Use shared memory or reduce per-process memory
  -> For multiprocessing: fork shares parent memory (COW)

Caches eating RAM?
  -> du -sh ~/.conda ~/.cache/pip ~/.cache/huggingface
  -> Symlink to Slate (see 05-storage-envs.md)
```

### GPU Memory (CUDA OOM)

**Symptom:** `RuntimeError: CUDA out of memory`

```
Batch size too large?
  -> Halve batch_size, re-run
  -> Use gradient accumulation to keep effective batch size

Model doesn't fit 1 GPU (>35 GB total)?
  -> Enable BF16: torch.cuda.amp.autocast(dtype=torch.bfloat16)
  -> Activation checkpointing (trades 30% speed for 60% memory)
  -> FSDP across 4 GPUs (see 03-gpu-optimization.md)

Memory leak (grows over epochs)?
  -> torch.cuda.memory_summary() to inspect
  -> Common: appending loss (not loss.item()) to a list
  -> Always .detach() before logging

Fragmentation?
  -> export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Job Stuck in PENDING

```bash
squeue -u $USER -o "%.10i %.20j %.10P %.8T %.10M %.10l %.6D %R"
```

| Reason | Cause | Fix |
|--------|-------|-----|
| `(Priority)` | Low fair-share from heavy usage | `sshare -u $USER`; reduce requests, wait (4-day decay) |
| `(Resources)` | Not enough free nodes | Fewer GPUs, shorter `--time`, try `gpu-debug` |
| `(QOSMaxJobsPerUserLimit)` | Hit concurrent job limit | `gpu-interactive`: max 1. Wait for others to finish |
| `(AssocGrpCPUMinutesLimit)` | Account limit reached | Contact PI or RT support |
| `(ReqNodeNotAvail)` | Nodes down for maintenance | Check `sinfo -p gpu -o "%20P %6D %6t"` |

**General fixes:**
- `gpu-debug` starts nearly instantly (1h limit)
- Shorter `--time` is the single biggest backfill factor
- Smaller `--mem` avoids locking unused resources
- Off-peak (weekends, evenings ET) has less contention

---

## NCCL / Distributed Training

| Error | Cause | Fix |
|-------|-------|-----|
| `Cuda failure 'out of memory'` | GPU OOM in allreduce | Reduce batch, enable BF16 |
| `Call to ibv_reg_mr failed` | Pinned memory limit | `ulimit -l` should be unlimited |
| Timeout in `init_process_group` | Wrong MASTER_ADDR/port | Use `scontrol show hostnames` |
| `No such device` | Wrong network interface | `export NCCL_SOCKET_IFNAME=hsn` |
| Slow multi-node | TCP fallback | Verify `NCCL_IB_DISABLE` is NOT set |
| Hang after first batch | Mismatched collectives | All ranks must execute same ops |

**Debug mode:**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

---

## Filesystem Issues

### Home Quota Full

**Symptom:** `Disk quota exceeded`

```bash
quota -s
du -sh ~/.conda ~/.cache/pip ~/.cache/huggingface ~/.local 2>/dev/null | sort -rh

# Fix: move caches to Slate
mkdir -p /N/slate/$USER/.cache/{pip,huggingface,conda}
rm -rf ~/.cache/pip && ln -sf /N/slate/$USER/.cache/pip ~/.cache/pip
rm -rf ~/.cache/huggingface && ln -sf /N/slate/$USER/.cache/huggingface ~/.cache/huggingface
```

### Scratch Data Disappeared

Files >30 days untouched are purged automatically. No warning, no recovery.
Use Slate for anything persistent. Scratch is for truly temporary output only.

### Slow I/O

| Cause | Fix |
|-------|-----|
| Many small files (>10K) | Convert to HDF5, WebDataset, or LMDB |
| Reading from home (`/N/u/`) | Home is NFS. Move to Slate (`/N/slate/`) |
| Not using local tmpfs | Stage to `/tmp/${SLURM_JOB_ID}/` (126 GB RAM-backed) |
| Large checkpoint writes | `lfs setstripe -c 4` on checkpoint directory |

---

## Module / Environment Issues

### ModuleNotFoundError on Compute Node

- Activate conda **inside** the sbatch script, not just on login
- Always: `module purge && module load ... && conda activate ...`
- Login node env does NOT transfer to compute nodes

### CUDA Version Mismatch

**Symptom:** `CUDA error: no kernel image is available`

```bash
module list                                    # what's loaded
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA
nvidia-smi                                     # driver CUDA

# Fix: match module to PyTorch build
module purge && module load cudatoolkit/12.6
```

---

## Training Issues

### Diverges After Resume

- Save/restore RNG states (`torch.random` + `torch.cuda`)?
- Save/restore LR scheduler state?
- Save/restore optimizer state (momentum, etc.)?
- See `03-gpu-optimization.md` for complete save/load pattern

### GPU Utilization Low (<30%)

| Cause | Fix |
|-------|-----|
| Data loading bottleneck | `num_workers=4-8`, `pin_memory=True`, `persistent_workers=True` |
| Small batch size | Increase until GPU mem ~80%. Use gradient accumulation if limited |
| CPU transforms in training loop | Move to DataLoader workers or pre-process offline |
| Model too small for GPU | May not be worth GPU. Check with `nvidia-smi dmon -d 5` |

---

## Quick Diagnostic Commands

```bash
# Job status and efficiency
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocGRES

# Why pending?
squeue -u $USER -o "%.10i %.20j %.10P %.8T %.10M %.10l %.6D %R"

# Fair-share
sshare -u $USER

# Partition availability
sinfo -p gpu -o "%20P %5a %10l %6D %6t"

# Storage
quota -s && lfs quota -hu $USER /N/slate/ && lfs quota -hu $USER /N/scratch/

# GPU health (on compute node)
nvidia-smi && nvidia-smi dmon -d 5 -c 3

# Network (NCCL debug)
ip link show | grep -E "hsn|mlx"
```
