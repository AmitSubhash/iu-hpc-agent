# IU HPC Agent

A comprehensive, verified reference for Indiana University's High-Performance Computing infrastructure -- designed to be consumed by AI coding agents (Claude Code, Copilot, Cursor, etc.) so they can write correct SLURM jobs, optimize resource usage, and build multi-stage HPC pipelines without guessing.

## What This Is

A single, dense Markdown file (`SKILL.md`) containing **everything** an AI agent (or human) needs to know to effectively use IU's Big Red 200 supercomputer, Quartz cluster, and supporting infrastructure. Every number in this document was either verified by SSHing into the cluster and running `scontrol`/`sinfo`/`sacctmgr`, or sourced from official IU documentation.

**2,000+ lines. 70 KB. 21 sections. All verified on-cluster 2026-03-19.**

## Why This Exists

HPC documentation is scattered across KB articles, man pages, and tribal knowledge. When you ask an AI to "write me a SLURM job," it guesses -- wrong partition limits, wrong memory defaults, wrong billing weights. This file eliminates guessing.

Drop `SKILL.md` into your AI agent's context and it will:
- Know that GPU nodes have **512 GB RAM** (not 256 GB as widely documented)
- Know that 1 GPU costs **16x** more than 1 CPU core in fair-share billing
- Know the exact scheduler config (backfill every 5 min, 4-day fair-share decay, max 500 array tasks)
- Write NUMA-aware jobs for the dual-socket EPYC 7742 (8 NUMA nodes, not 4)
- Generate correct `--hint=nomultithread` + thread control for scientific Python
- Build dependency-chained pipelines (preprocess on CPU -> train on GPU -> evaluate)

## IU HPC Infrastructure at a Glance

### Big Red 200 (Primary Supercomputer)

| Spec | Value |
|------|-------|
| Architecture | HPE Cray EX (Shasta) -- first production Cray Shasta worldwide |
| Peak Performance | ~7 PFLOPS |
| CPU Nodes | 640 (2x AMD EPYC 7742, 128 cores, 256 GB DDR4) |
| GPU Nodes | 66 (1x AMD EPYC 7713, 64 cores, **512 GB DDR4**, 4x NVIDIA A100-SXM4-40GB) |
| Total GPUs | 264x A100-40GB with NVLink 3.0 full mesh (NV4) |
| Total CPU Cores | 86,144 |
| Interconnect | HPE Slingshot-10, 200 Gbps, Dragonfly topology, <1.8 us latency |
| Storage | Lustre (Slate 800 GB, Scratch 100 TB), Home 100 GB, SDA tape archive |
| Scheduler | SLURM 24.05.4, backfill with fair-share priority |
| OS | SUSE Linux Enterprise Server 15 SP6 |

### GPU Node Details (Verified on Cluster)

| Spec | Value |
|------|-------|
| GPU | NVIDIA A100-SXM4-40GB |
| GPU Memory BW | ~1,350 GB/s measured |
| NVLink P2P BW | 93.5 GB/s unidirectional (93% theoretical) |
| NVLink P2P Latency | ~2.2 us |
| Host RAM | **512 GB** (500 GB usable) |
| Compute Node /tmp | 126 GB RAM-backed tmpfs |
| FP16/BF16 | 624 TFLOPS per GPU |
| FP8 | Via MS-AMP (O1/O2/O3) |

### CPU Node Details (AMD EPYC 7742 Rome)

| Spec | Value |
|------|-------|
| Sockets | 2 per compute node |
| Cores | 128 physical (256 with SMT) |
| NUMA Nodes | **8 per compute node** (16 cores, ~32 GB each) |
| L3 Cache | 16 MB per CCX (not shared across CCXs!) |
| Memory BW | ~170-185 GB/s per socket (STREAM Triad) |
| ISA | AVX2 + FMA (NO AVX-512) |
| Compiler Target | `-march=znver2 -mtune=znver2` |

### Quartz (High-Throughput Cluster)

| Spec | Value |
|------|-------|
| CPU Nodes | 92 (2x EPYC 7742, 128 cores, **512 GB** RAM) |
| GPU Nodes | 22 (V100-32GB) + 12 (H100) |
| Best For | Memory-hungry CPU work (512-768 GB/node), high-throughput independent tasks |

### SLURM Partitions

| Partition | Nodes | GPUs | Time Limit | Default QOS Max Nodes |
|-----------|-------|------|------------|-----------------------|
| `general` | 638 | -- | 4 days | 200 |
| `gpu` | 62 | 4x A100 | 2 days | 36 |
| `gpu-debug` | 2 | 4x A100 | 1 hour | 2 |
| `gpu-interactive` | 2 | 4x A100 | 4 hours | 1 |
| `debug` | 2 | -- | 1 hour | 2 |

### TRES Billing Weights

| Partition | CPU Weight | Memory Weight | GPU Weight |
|-----------|-----------|---------------|-----------|
| CPU (`general`) | 1.0 per core | 0.512 per GB | -- |
| GPU (`gpu`) | 1.0 per core | 0.128 per GB | **16.0 per GPU** |

1 GPU = 16 CPU cores in fair-share cost. Under-utilizing GPUs burns your priority.

### Scheduler Configuration

| Parameter | Value |
|-----------|-------|
| Backfill Interval | 300s (5 min) |
| Backfill Window | 7 days |
| Priority Decay Half-Life | **4 days** |
| Priority Max Age | 12 hours |
| Age/FairShare/QOS Weights | Equal (100000 each) |
| Max Array Size | **500** |

## What's in SKILL.md

| Section | What It Covers |
|---------|---------------|
| **How to Use** | Lookup table -- "I need to X" -> go to section Y |
| **Architecture** | Full hardware specs for BR200 + Quartz, verified node topology |
| **NUMA/CPU Deep Dive** | 8-NUMA dual-socket layout, cache hierarchy, memory bandwidth |
| **Network** | Slingshot-10 specs, NCCL config, bandwidth hierarchy |
| **Storage** | All 5 tiers with quotas, purge policies, symlink tricks |
| **Partitions + QOS** | Every partition, QOS limit, billing weight, scheduler param |
| **Software** | All module versions (Python 3.10-3.13, CUDA 11.4-12.6, cuDNN, NCCL, Apptainer) |
| **Allocations** | IU's open-access model, RT Projects, ACCESS/Jetstream2 |
| **CPU vs GPU** | Decision framework with concrete examples |
| **CPU Optimization** | SMT, NUMA binding, BLAS threading, compiler flags, MKL-on-AMD fix |
| **SLURM Optimization** | Job arrays, backfill exploitation, dependency chains, right-sizing |
| **GPU Optimization** | DDP/FSDP decision tree, mixed precision, PyTorch checklist |
| **Checkpointing** | Signal handling, auto-resubmission, robust save/load |
| **Storage I/O** | tmpfs staging, Lustre striping, small-file pathology fixes |
| **Environment** | Conda on Slate, Apptainer containers, module stacking |
| **Common Pitfalls** | OOM, NCCL errors, quota exhaustion -- with fixes |
| **Jetstream2 + ACCESS** | Cloud VMs, SU rates, allocation tiers |
| **Templates** | 8 copy-paste SBATCH templates for common scenarios |
| **Workflow Recipes** | Decision flowchart, brain imaging pipeline, ML training with auto-retry, mass parallel processing, post-job analysis |

## Key Discoveries (Things the Official Docs Get Wrong)

| What's Documented | What's Actually True |
|---|---|
| GPU nodes have 256 GB RAM | **512 GB RAM** (verified via `scontrol show node`) |
| Login node NUMA = compute node NUMA | Compute nodes are **dual-socket with 8 NUMA nodes** (login is single-socket with 4) |
| No local fast storage | `/tmp` is **126 GB RAM-backed tmpfs** on compute nodes |
| Default job time is reasonable | **1 hour if you forget `--time`** (gpu-interactive: 30 min) |
| MaxArraySize is 1001 | **500** on BR200 |
| GPU billing is proportional | 1 GPU = **16x** the billing of 1 CPU core |

## How to Use

### For AI Agents (Claude Code, Cursor, etc.)

Drop `SKILL.md` into your agent's skill/rules directory:

```bash
# Claude Code
cp SKILL.md ~/.claude/skills/iu-hpc/SKILL.md

# Or include in any project's .cursor/rules/, .github/copilot/, etc.
```

### For Humans

Read `SKILL.md` directly. Start with the "How to Use This Skill" lookup table at the top, then jump to the section you need.

### On Big Red 200

```bash
# Copy to your home directory for reference
scp SKILL.md br200:~/hpc-reference.md
```

## Quick Start: Your First Optimized Job

```bash
#!/bin/bash
#SBATCH -J my_job
#SBATCH -A r00602
#SBATCH -p general
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
mkdir -p logs
module purge

# Thread control (prevents BLAS oversubscription)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_DEBUG_CPU_TYPE=5  # Fix MKL on AMD

conda activate /N/slate/$USER/envs/myenv
python my_script.py
```

## Contributing

If you have verified data from other IU systems (Quartz GPU node specs, Jetstream2 benchmarks), open a PR. All data must be verified on-cluster -- no guessing.

## License

MIT. Use freely.

## Acknowledgments

- Indiana University Research Technologies
- IU Knowledge Base (kb.iu.edu)
- Verified hardware data from on-cluster benchmarks
