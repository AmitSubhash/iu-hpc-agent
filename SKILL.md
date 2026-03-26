# IU HPC Skill: Big Red 200, Quartz, and IU Research Computing

Quick reference for IU's HPC infrastructure. Verified on-cluster 2026-03-19.

---

## User Configuration

Add to your `~/.bashrc` on BR200/Quartz:

```bash
export SLURM_ACCOUNT="your_account_id"   # from projects.rt.iu.edu
export PROJ="/N/slate/$USER/your_project"
export ENVS="/N/slate/$USER/envs"
```

All templates use `<account>` for `#SBATCH -A` and `$USER` for paths. Replace as needed.

---

## Task Routing

Read only the file(s) relevant to the question. Do NOT load all files.

| Task | Read |
|------|------|
| **First time on BR200/Quartz? Start here** | `00-quickstart.md` |
| Hardware specs (CPU, GPU, NVLink, network) | `01-hardware.md` |
| SLURM partitions, billing, QOS, scheduler tuning | `02-slurm.md` |
| GPU training (DDP, FSDP, mixed precision, checkpointing) | `03-gpu-optimization.md` |
| CPU optimization (NUMA, SMT, BLAS threading) | `04-cpu-optimization.md` |
| Storage tiers, I/O tuning, conda, containers, modules | `05-storage-envs.md` |
| Copy-paste SBATCH templates | `06-templates.md` |
| Workflow recipes, tips, efficiency checklist | `07-recipes-tips.md` |
| Allocations, ACCESS, Jetstream2, support channels | `08-access-support.md` |
| Estimate resources for a task, generate SBATCH | `09-resource-estimator.md` |
| **Job failed? Debug here** | `10-troubleshooting.md` |

---

## System Comparison

| | Big Red 200 | Quartz |
|--|-------------|--------|
| CPU nodes | 640 (2x EPYC 7742, 128c, 256 GB) | 92 (2x EPYC 7742, 128c, 512 GB) |
| GPU nodes | 66 (1x EPYC 7713, 64c, 512 GB, 4x A100-40GB) | 22 V100-32GB + 12 H100 |
| Interconnect | Slingshot-10 (200 Gbps, ~1.8 us) | HDR IB / Ethernet |
| Best for | Tightly-coupled parallel, deep learning | High-throughput, memory-hungry CPU |
| SSH | `bigred200.uits.iu.edu` | `quartz.uits.iu.edu` |

**When to use which:**
- Multi-node distributed training: **BR200** (Slingshot + NVLink)
- Single/multi-GPU training: **BR200** (A100 Tensor Cores)
- H100 training: **Quartz** (12 nodes x 4x H100)
- Memory-intensive CPU (>256 GB): **Quartz** (512 GB/node)
- Embarrassingly parallel: **either** (job arrays, wherever has capacity)

---

## CPU vs GPU Decision Heuristic

```
Profile first: nvidia-smi dmon -d 5
  GPU util < 30%  -> CPU-appropriate (use general partition)
  GPU util 30-70% -> optimize data loading, increase batch size
  GPU util > 70%  -> good GPU fit
```

**CPUs win:** preprocessing, small models (<10M params), compilation, file conversion, feature extraction, embarrassingly parallel.
**GPUs win:** training (>1M params), batched matrix ops (>512x512), Tensor Core workloads, mixed-precision.

**The "30 CPUs vs 1 GPU" strategy:** `general` has 638 nodes (vs 62 GPU), 4-day limit (vs 2-day), starts faster, no GPU fair-share penalty.

---

## Standard SBATCH Header

```bash
#!/bin/bash
#SBATCH -J <job_name>
#SBATCH -A <account>         # your account from projects.rt.iu.edu
#SBATCH -p gpu               # general | gpu | gpu-debug | debug
#SBATCH -N 1
#SBATCH --gres=gpu:4         # omit for CPU jobs
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out -e logs/%x-%j.err

set -euo pipefail
```
