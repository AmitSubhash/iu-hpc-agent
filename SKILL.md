# IU HPC Mastery: Big Red 200, Quartz, and the Full IU Research Computing Ecosystem

Complete reference for Indiana University's HPC infrastructure. Covers hardware architecture,
SLURM optimization, resource strategies, allocation management, and production workflow patterns
for Big Red 200 (BR200), Quartz, Jetstream2, and supporting storage/network systems.

**User context**: atsubhas, account r00602 (secondary: workshop), PhD Neuroengineering at IU.
Primary workloads: PyTorch distributed training, Monte Carlo photon transport (MCX), scientific computing.

---

## How to Use This Skill

**For Claude:** When Amit asks anything about SLURM, HPC, Big Red, Quartz, job submission,
GPU/CPU optimization, or brain imaging pipelines, read the relevant section below. DO NOT
guess at partition limits, billing weights, or scheduler behavior -- the verified data is here.

**Quick navigation by task:**

| "I need to..." | Go to |
|----------------|-------|
| Submit a job and not sure what partition/resources | Section 21.1 (Decision Flowchart) |
| Write a SBATCH script from scratch | Section 21.2 (Smart Defaults Template) |
| Optimize CPU-heavy Python (NumPy, nibabel, scikit-learn) | Section 9A (CPU Optimization) |
| Understand GPU billing cost vs CPU | Section 6.3 (TRES Billing Weights) |
| Run a multi-stage pipeline (preproc -> train -> eval) | Section 21.6 (Brain Imaging Recipe) |
| Debug why my job is pending | Section 6.4 (QOS Limits) + 6.5 (Scheduler Config) |
| Decide GPU vs CPU for a workload | Section 9 (Decision Framework) |
| Stage data for fast I/O | Section 21.3 (Local Storage) + 13 (Storage Optimization) |
| Set up checkpointing for long training | Section 12 (Checkpointing) |
| Fix NCCL / distributed training errors | Section 15 (Common Pitfalls) |
| Understand NUMA and thread binding | Section 21.4 (Dual-Socket NUMA) + 9A.2 |
| Check what software is available | Section 7 (Software) + 9A.8 (CPU Software) |
| Request an allocation or get access | Section 8 (Allocations) |
| Use Jetstream2 or ACCESS | Section 16-17 |

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Big Red 200 Architecture](#2-big-red-200-architecture)
3. [Quartz Architecture](#3-quartz-architecture)
4. [Network and Interconnect](#4-network-and-interconnect)
5. [Storage Systems](#5-storage-systems)
6. [SLURM Partitions (Both Systems)](#6-slurm-partitions)
7. [Software Environment](#7-software-environment)
8. [Allocations and Access](#8-allocations-and-access)
9. [CPU vs GPU Decision Framework](#9-cpu-vs-gpu-decision-framework)
10. [SLURM Job Optimization](#10-slurm-job-optimization)
11. [GPU Utilization Optimization](#11-gpu-utilization-optimization)
12. [Checkpointing and Fault Tolerance](#12-checkpointing-and-fault-tolerance)
13. [Storage and I/O Optimization](#13-storage-and-io-optimization)
14. [Environment Management](#14-environment-management)
15. [Common Pitfalls and Debugging](#15-common-pitfalls-and-debugging)
16. [Jetstream2 Cloud Computing](#16-jetstream2-cloud-computing)
17. [ACCESS Allocations](#17-access-allocations)
18. [Support Channels](#18-support-channels)
19. [Ready-to-Use Templates](#19-ready-to-use-templates)
20. [On-Cluster Verification Commands](#20-on-cluster-verification-commands)
21. [Practical Workflow Recipes](#21-practical-workflow-recipes) **<-- START HERE**
22. [Non-Obvious Tricks and Unique Connections](#22-non-obvious-tricks-and-unique-connections)

---

## 1. System Overview

| System | Type | Peak | CPU Nodes | GPU Nodes | GPUs | Interconnect |
|--------|------|------|-----------|-----------|------|--------------|
| **Big Red 200** | HPE Cray EX (Shasta) | ~7 PFLOPS | 640 (638 in `general`) | 66 (62 in `gpu`) | 264x A100-40GB | Slingshot-10 (200 Gbps) |
| **Quartz** | High-throughput cluster | ~423 TFLOPS | 92 | 22 (V100) + 12 (H100) | 88 V100 + 48 H100 | HDR IB / Ethernet |
| **Jetstream2** | NSF cloud (OpenStack) | 8 PFLOPS | Variable | 90 (A100) | 360x A100 | 100 Gbps Ethernet |

**When to use which:**

| Workload | Best System | Why |
|----------|-------------|-----|
| Multi-node distributed training | BR200 | Slingshot-10 + NVLink, A100 Tensor Cores |
| Single-GPU or 1-4 GPU training | BR200 (gpu) | A100-40GB, NVLink mesh within node |
| H100-class training | Quartz | 12 nodes with 4x H100 each |
| CPU-heavy preprocessing, many-core parallelism | Quartz | 512 GB RAM/node, 128 cores/node |
| Memory-intensive CPU work (>256 GB) | Quartz | 512 GB (general) or 768 GB (GPU nodes) |
| Embarrassingly parallel (1000s of tasks) | Either | Job arrays on whichever has capacity |
| Cloud VMs, Kubernetes, custom networking | Jetstream2 | Full VM control via OpenStack |
| Quick prototyping, Jupyter | BR200 (gpu-interactive) or Jetstream2 | 4-hour interactive or persistent VM |

---

## 2. Big Red 200 Architecture

### 2.1 Overview

| Attribute | Value |
|-----------|-------|
| Vendor | HPE (Cray EX / Shasta) |
| First production Cray Shasta deployment worldwide | Yes |
| Online | CPU: Jan 2020, GPU: Apr 2022 |
| Hostname | `bigred200.uits.iu.edu` (SSH alias: `br200`) |
| OS | SUSE Linux Enterprise Server 15 SP6 |
| Cost | ~$9.6M |
| Cooling | Warm water direct liquid cooling (ASHRAE W3) |
| TOP500 (Nov 2024) | CPU: Rank 458, GPU: Rank 312 |

### 2.2 CPU Compute Nodes (Phase 1)

| Spec | Value |
|------|-------|
| Node count | 640 total (638 in `general`, 2 in `debug`) |
| Node IDs | nid[0001-0638] (general), nid[0639-0640] (debug) |
| Blade model | HPE Cray EX425 |
| CPUs per node | 2x AMD EPYC 7742 (Rome), 2 sockets |
| Cores per CPU | 64 cores / 128 threads (SMT-2) |
| Clock | 2.25 GHz base (1.5 GHz min), 225W TDP, frequency boost enabled |
| ISA extensions | AVX2, FMA, SSE4.2, AES-NI, SHA (NO AVX-512) |
| Total CPU cores | 81,920 physical (163,840 threads with SMT) |
| Memory per node | 256 GB DDR4-3200 (**250 GB usable**, 10 GB reserved via MemSpecLimit) |
| Memory BW per socket | ~204.8 GB/s theoretical (8 channels x DDR4-3200) |
| Memory BW measured | ~170-185 GB/s per socket (STREAM Triad) |
| Total CPU memory | ~160 TB |
| Swap | 2 GB (effectively unusable for HPC) |

### 2.2.1 CPU Microarchitecture (AMD Zen 2 Rome) -- Verified on Cluster

| Spec | Value |
|------|-------|
| Architecture | Zen 2 chiplet design |
| Chiplets per socket | 8 CCDs (Core Complex Dies) |
| CCXs per CCD | 2 (each CCX = 4 cores + 16 MB shared L3) |
| **NUMA nodes per socket** | **4** (each = 2 CCDs = 16 physical cores) |
| NUMA nodes per node | **4** (login node shows 1 socket); **8 on dual-socket compute nodes** |
| **L1d cache** | 32 KB/core, ~1.5 ns latency |
| **L1i cache** | 32 KB/core |
| **L2 cache** | 512 KB/core, ~5 ns latency |
| **L3 cache** | 16 MB/CCX (256 MB total per socket), ~17 ns same-CCX, ~35-40 ns cross-CCX |
| **DRAM latency** | ~120-140 ns local NUMA, ~190-220 ns remote NUMA |
| NUMA distance | 10 (local), 12 (remote) -- verified via `numactl --hardware` |
| Memory per NUMA node | ~32 GB on compute (8 NUMA x 32 GB = 256 GB); ~64 GB on login (4 NUMA x 64 GB) |

**Key insight:** L3 is per-CCX (16 MB), NOT shared across the whole socket. 4 cores share
16 MB of L3. Cross-CCX access within the same NUMA node is 2x slower than same-CCX.

### 2.2.2 NUMA Node Layout (Verified on Compute Node)

**Login node** is single-socket (4 NUMA nodes). **Compute nodes are DUAL-SOCKET with 8 NUMA nodes:**

```
Socket 0:                          Socket 1:
  Node 0: CPUs 0-15, 128-143        Node 4: CPUs 64-79, 192-207
  Node 1: CPUs 16-31, 144-159       Node 5: CPUs 80-95, 208-223
  Node 2: CPUs 32-47, 160-175       Node 6: CPUs 96-111, 224-239
  Node 3: CPUs 48-63, 176-191       Node 7: CPUs 112-127, 240-255

Each NUMA node: ~32 GB RAM, 16 physical cores
Total: 128 physical cores (256 with SMT), 8 x 32 GB = 256 GB
```

Physical cores: 0-127. SMT siblings: 128-255.
Local `/tmp`: 126 GB tmpfs. `/dev/shm`: 126 GB tmpfs. Both RAM-backed, shared pool.

### 2.3 GPU Accelerated Nodes (Phase 2)

| Spec | Value |
|------|-------|
| Node count | 66 total (62 in `gpu`, 2 in `gpu-debug`/`gpu-interactive` nid[0703-0704]) |
| Node IDs | nid[0641-0702] (gpu), nid[0703-0704] (gpu-debug + gpu-interactive) |
| Blade model | HPE Cray EX235n |
| CPU per node | 1x AMD EPYC 7713 (Milan), 64 cores/128 threads, 2.0 GHz, 1 socket |
| GPUs per node | 4x NVIDIA A100-SXM4-40GB |
| Total GPUs | 264 (66 nodes x 4) |
| **Host memory** | **512 GB DDR4** (**500 GB usable**, 10 GB MemSpecLimit) -- VERIFIED on cluster |
| Default mem per GPU | 125,440 MB (~122.5 GB) if not specified |
| GPU memory | 40 GB HBM2e per GPU (160 GB total per node) |
| GPU memory BW | ~1,555 GB/s theoretical, ~1,350 GB/s measured |
| GPU compute | 19.5 TFLOPS FP64, 312 TFLOPS TF32, 624 TFLOPS FP16/BF16 per GPU |
| MIG support | Yes (up to 7 instances per A100) |
| FP8 | Via MS-AMP (optimization levels O1/O2/O3) |

### 2.4 Intra-Node GPU Topology (Verified 2026-03-10)

| Spec | Value |
|------|-------|
| Interconnect | NVLink 3.0 |
| Topology | **NV4 full mesh** (every GPU pair directly connected) |
| Links per GPU | 12 total (4 links to each of 3 other GPUs) |
| Per-link speed | 25 GB/s per direction |
| Per-GPU NVLink BW | 300 GB/s per direction (600 GB/s bidirectional) |
| Measured P2P BW | 93.5 GB/s uni / 185 GB/s bidi per pair (93% theoretical) |
| Measured P2P latency | ~2.2 us (vs 12-53 us without NVLink) |

**Note:** BR200 does NOT use NVSwitch. The 4-GPU SXM4 config uses a direct NVLink mesh,
not the 8-GPU DGX A100 design with NVSwitch.

### 2.5 System Totals

| Metric | Value |
|--------|-------|
| Total nodes | 706 (640 CPU + 66 GPU) |
| Total CPU cores | 86,144 (81,920 CPU + 4,224 GPU-node) |
| Total GPUs | 264x A100-SXM4-40GB |
| Total host memory | ~197 TB (640x256GB + 66x512GB) + 10.6 TB GPU HBM2e |
| SLURM version | 24.05.4 |
| Cluster name | `br200` |

---

## 3. Quartz Architecture

### 3.1 Overview

| Attribute | Value |
|-----------|-------|
| Online | December 2020 |
| Hostname | `quartz.uits.iu.edu` |
| Design | High-throughput computing (many independent jobs, longer runtimes) |

### 3.2 General Compute Nodes

| Spec | Value |
|------|-------|
| Node count | 92 |
| CPUs per node | 2x AMD EPYC 7742 (Rome) |
| Cores per node | 128 (2 x 64-core) |
| Memory per node | **512 GB** DDR4 (2x BR200 CPU nodes) |
| Total cores | ~11,776 |
| Peak per node | >4,608 GFLOPS |

### 3.3 GPU Nodes

| GPU Type | Nodes | GPUs/Node | Total GPUs | Memory/Node |
|----------|-------|-----------|------------|-------------|
| NVIDIA V100 | 22 | 4x V100-32GB | 88 | **768 GB** |
| NVIDIA H100 | 12 | 4x H100 | 48 | TBD (verify on cluster) |

### 3.4 Key Differences from BR200

| Feature | Quartz | Big Red 200 |
|---------|--------|-------------|
| CPU node RAM | **512 GB** | 256 GB |
| GPU node RAM | **768 GB** | **512 GB** (was documented as 256 GB, actually 512 GB!) |
| GPU type | V100-32GB + H100 | A100-40GB |
| Interconnect | Ethernet/IB (slower) | Slingshot-10 (200 Gbps, low latency) |
| Best for | High-throughput, memory-hungry CPU | Tightly-coupled parallel, deep learning |
| Total CPU nodes | 92 | 640 |

---

## 4. Network and Interconnect

### 4.1 Big Red 200: HPE Slingshot-10

| Spec | Value |
|------|-------|
| Switch ASIC | Rosetta (TSMC 16nm, 250W) |
| Ports per switch | 64 x 200 Gb/s bidirectional |
| Switch bisection BW | 25.6 Tb/s bidirectional |
| Topology | Dragonfly (max 3 switch hops) |
| NIC | Mellanox ConnectX-5 100G (RoCE) |
| NICs per GPU node | 2 (mlx5_0, mlx5_1) |
| Injection BW per NIC | 100 Gb/s |
| Unloaded latency | ~1.8 us |
| Tail latency (p99) | <8.7 us |
| Intra-group cabling | Copper (up to 2.6m, fully connected) |
| Inter-group cabling | Optical (up to 100m) |

### 4.2 NCCL Configuration for BR200

```bash
# Standard settings
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_IFNAME=hsn          # High-Speed Network (Slingshot)

# If InfiniBand issues (fallback):
# export NCCL_IB_DISABLE=1
```

### 4.3 Bandwidth Hierarchy (BR200)

```
GPU HBM2e Local:     1,350 GB/s
NVLink (intra-node): 93.5 GB/s per GPU pair
Slingshot (inter-node): ~12.5 GB/s per NIC (100 Gbps)
Lustre I/O:          ~1-10 GB/s (shared, variable)
```

**Implication:** Keep communication within a single node (4 GPUs over NVLink) whenever possible.
Inter-node is ~7.5x slower than intra-node GPU-to-GPU.

---

## 5. Storage Systems

### 5.1 Storage Tier Overview

| Storage | Path | Type | Quota | Purge | Backed Up | Best For |
|---------|------|------|-------|-------|-----------|----------|
| **Home (Geode)** | `/N/u/atsubhas/BigRed200/` | NFS | 100 GB, 800K files | No | No | Scripts, configs, small code |
| **Slate** | `/N/slate/atsubhas/` | Lustre | 800 GB (1.6 TB on request) | No | **No** | Active data, checkpoints, envs |
| **Slate-Project** | `/N/slate/project/<name>/` | Lustre | Free up to 15 TB | No | **No** | Shared group/project data |
| **Scratch** | `/N/scratch/atsubhas/` | Lustre | 100 TB, 10M inodes | **30 days** | No | Large temporary outputs |
| **SDA** | HPSS tape | Tape archive | ~42 PB capacity | No | **Yes** (2 copies, 2 sites) | Long-term archival |

### 5.2 Critical Storage Rules

1. **Home is shared** across BR200 and Quartz (same mount).
2. **Slate is NOT backed up** -- you are responsible for critical data.
3. **Scratch purges files not accessed for 30+ days** without notification.
   Emergency purge when filesystem >80% -- oldest files go first.
4. **SDA** is the only storage with backup (2 geographically separated tape copies: IUB + IUPUI).
5. **Home fills fast** from conda envs, pip cache, HuggingFace cache. Symlink these to Slate:

```bash
# Prevent home quota exhaustion
mkdir -p /N/slate/atsubhas/.cache/{pip,huggingface,conda}
ln -sf /N/slate/atsubhas/.cache/pip ~/.cache/pip
ln -sf /N/slate/atsubhas/.cache/huggingface ~/.cache/huggingface
ln -sf /N/slate/atsubhas/.cache/conda ~/.conda/pkgs
```

### 5.3 Checking Quotas

```bash
quota -s                              # Home directory
lfs quota -hu atsubhas /N/slate/      # Slate user quota
lfs quota -hu atsubhas /N/scratch/    # Scratch quota
```

### 5.4 Standard Project Directory Layout

```bash
export PROJ="${PROJ:-/N/slate/$USER/<project_name>}"
export LOGS=$PROJ/logs
export REPOS=$PROJ/repos
export DATA=$PROJ/data
export CKPT=$PROJ/checkpoints
export ENVS=/N/slate/$USER/envs       # Conda envs on Slate, not Home

mkdir -p "$LOGS" "$REPOS" "$DATA" "$CKPT" "$ENVS"
```

---

## 6. SLURM Partitions

### 6.1 Big Red 200

| Partition | Node Type | Cores/Node | RAM | GPUs | Time Limit | Nodes | Use |
|-----------|-----------|------------|-----|------|------------|-------|-----|
| `general` | CPU | 128 | 250 GB usable | -- | **4 days** | 638 | CPU batch, preprocessing |
| `debug` | CPU | 128 | 250 GB usable | -- | 1 hour | 2 | Quick CPU debugging |
| `interactive` | CPU | 128 | 250 GB usable | -- | ~4 hours | subset | Interactive CPU sessions |
| `gpu` | GPU | 64 | **500 GB usable** | 4x A100 | **2 days** | 62 | Main training jobs |
| `gpu-debug` | GPU | 64 | **500 GB usable** | 4x A100 | 1 hour | 2 | Quick GPU debugging |
| `gpu-interactive` | GPU | 64 | **500 GB usable** | 4x A100 | 4 hours | 2 (shared w/ debug) | Jupyter, profiling |

### 6.2 Quartz

| Partition | Node Type | Cores/Node | RAM | GPUs | Time Limit | Nodes | Use |
|-----------|-----------|------------|-----|------|------------|-------|-----|
| `general` | CPU | 128 | 512 GB | -- | **4 days** | 92 | CPU-heavy, memory-hungry |
| `debug` | CPU | 128 | 512 GB | -- | 1 hour | subset | CPU debugging |
| `interactive` | CPU | 128 | 512 GB | -- | ~4 hours | subset | Interactive CPU |
| `gpu` | GPU (V100) | varies | 768 GB | 4x V100 | **2 days** | 22 | V100 training |
| `gpu` | GPU (H100) | varies | TBD | 4x H100 | **2 days** | 12 | H100 training |
| `gpu-debug` | GPU | varies | varies | 4x V100/H100 | 1 hour | subset | GPU debugging |

### 6.3 TRES Billing Weights (Verified 2026-03-19)

How SLURM "bills" resource consumption for fair-share accounting:

**CPU partitions (general, debug):**
```
TRESBillingWeights=CPU=1.0,Mem=0.512G
```
Meaning: 1 CPU-core = 1 billing unit; 1 GB memory = 0.512 billing units.
A full CPU node (128 cores, 250 GB) bills as: 128 + (250 * 0.512) = **256 billing units/hour**

**GPU partitions (gpu, gpu-debug, gpu-interactive):**
```
TRESBillingWeights=CPU=1.0,Mem=0.128G,GRES/gpu=16.0
```
Meaning: 1 GPU = 16 billing units; 1 CPU-core = 1; 1 GB memory = 0.128.
A full GPU node (64 cores, 500 GB, 4 GPUs) bills as: 64 + (500 * 0.128) + (4 * 16) = **192 billing units/hour**

**Key insight for cost optimization:**
- 1 GPU costs 16x more than 1 CPU core in fair-share terms
- Requesting 4 GPUs = 64 billing units (same as 64 CPU cores!)
- Memory is cheap on GPU nodes (0.128/GB vs 0.512/GB on CPU nodes)
- **Under-using GPUs wastes billing units** -- profile GPU utilization first

### 6.4 QOS Configuration (Verified 2026-03-19)

| QOS Name | Priority | MaxWall | MaxNodes/User | MaxSubmit/User | Flags |
|----------|----------|---------|---------------|----------------|-------|
| `normal` | 0 | -- | -- | 500 | -- |
| `usage` | 0 | 4 days | -- | 500 | DenyOnLimit |
| `debug` | 0 | 1 hour | 2 | 4 | DenyOnLimit |
| `allocated` | 0 | 4 days | **200** | 1000 | DenyOnLimit |
| `allocated-gpu` | 0 | 2 days | **36** | 1000 | DenyOnLimit |
| `radl` | **1** | 4 days | -- | 500 | DenyOnLimit |
| `highprio` | **1** | 4 days | -- | 500 | DenyOnLimit |
| `highprio-gpu` | **1** | 2 days | 12 | 1000 | DenyOnLimit |
| `highnodes` | 0 | 4 days | 256 | 1 | DenyOnLimit |
| `gpu-interactive` | 0 | 4 hours | 1 | 1 | DenyOnLimit |
| `benchmark` | 0 | -- | -- | 4096 | -- |
| `reframe` | 0 | -- | 704 | 2000 | DenyOnLimit,OverPartQOS |

**Your account (r00602):**
- Default QOS on `general`: `allocated` (max 200 nodes, 1000 max submit)
- Default QOS on `gpu`: `allocated-gpu` (max 36 nodes, 1000 max submit)
- `gpu-interactive`: max 1 node, 1 job at a time

### 6.5 Scheduler Configuration (Verified 2026-03-19)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| SchedulerType | `sched/backfill` | Backfill scheduling enabled |
| PriorityType | `priority/multifactor` | Weighted factor priority |
| PriorityWeightAge | **100000** | Wait time matters (equal to fairshare) |
| PriorityWeightFairShare | **100000** | Usage history matters equally |
| PriorityWeightQOS | **100000** | QOS boost matters equally |
| PriorityWeightJobSize | 0 | Job size does NOT affect priority |
| PriorityWeightPartition | 0 | Partition does NOT affect priority |
| PriorityWeightAssoc | 0 | Account does NOT affect priority |
| PriorityDecayHalfLife | **4 days** | Usage penalty halves every 4 days |
| PriorityMaxAge | **12 hours** | Priority from waiting maxes out at 12h |
| **bf_interval** | 300s (5 min) | Backfill runs every 5 minutes |
| **bf_window** | 10080 min (7 days) | Backfill looks 7 days ahead |
| **bf_max_job_test** | 1500 | Tests up to 1500 pending jobs |
| **bf_max_job_user** | 10 | Max 10 of YOUR jobs tested per backfill cycle |
| bf_continue | enabled | Continues backfill across scheduling cycles |
| bf_resolution | 300s | Scheduling resolution for backfill |
| **MaxArraySize** | **500** | Max job array index (0-499) |
| MaxJobCount | 100000 | Max total jobs in system |

**Practical implications:**
- Priority from waiting caps at 12 hours -- beyond that, waiting longer does not help
- Fair-share decay of 4 days means heavy GPU usage last week is 50% forgotten now
- Backfill only tests 10 of your jobs per cycle -- submit your most important jobs first
- Job arrays max out at 500 tasks (not 1001 default)

### 6.6 Partition Defaults and Memory Allocation

| Partition | DefaultTime | DefMemPerCPU | DefMemPerGPU | OverSubscribe |
|-----------|-------------|--------------|--------------|---------------|
| `general` | **1 hour** | 1,920 MB | -- | NO |
| `gpu` | **1 hour** | -- | **125,440 MB (~122.5 GB)** | NO |
| `gpu-debug` | **1 hour** | -- | 125,440 MB | NO |
| `gpu-interactive` | **30 min** | -- | 125,440 MB | NO |
| `debug` | **1 hour** | 1,920 MB | -- | NO |

**Critical:** If you omit `--time`, you get 1 hour. If you omit `--mem` on GPU, each GPU allocates 122.5 GB.

### 6.7 Current Storage Quotas (atsubhas, Verified 2026-03-19)

| Storage | Used | Quota | Files Used | File Limit |
|---------|------|-------|------------|------------|
| Home | 56.9 GB (56%) | 100 GB | 487,555 (59%) | 819,200 |
| Slate | 83.1 GB | 800 GB | 72,521 | 6,400,000 |
| Scratch | 64.9 GB | 100 TB | 120,912 | 10,000,000 |

**Warning:** Home is at 56% capacity and 59% file count. Symlink caches to Slate soon.

---

## 7. Software Environment

### 7.1 Module System (Lmod)

```bash
module spider              # All available modules
module spider <name>       # Search for specific module
module load <name>         # Load module
module list                # Currently loaded
module purge               # Clear all (ALWAYS do this first)
module swap <old> <new>    # Swap modules
```

### 7.2 Compilers (Programming Environments)

| PrgEnv | Compiler | Best For |
|--------|----------|----------|
| `PrgEnv-gnu` | GCC | Open-source software (default choice) |
| `PrgEnv-cray` | Cray CCE | Cray-optimized Fortran/C |
| `PrgEnv-intel` | Intel oneAPI | MKL-heavy, Fortran optimization |
| `PrgEnv-nvidia` | NVIDIA HPC SDK | OpenACC, CUDA Fortran |

### 7.3 CUDA and GPU Libraries (Verified on BR200)

| Module | Versions |
|--------|----------|
| `cudatoolkit` | 11.4, 11.7, 11.8, 12.2, **12.6** |
| `cudnn` | 8.9.2.26_cuda12, 9.5.1.17_cuda12, **9.10.1.4_cuda12** |
| `nccl` | 2.27.7-1 |
| `python/gpu/3.10.10` | GPU Python 3.10 (PyTorch + CUDA) |
| `python/gpu/3.11.5` | GPU Python 3.11 (PyTorch + CUDA) |
| `python/gpu/3.12.5` | GPU Python 3.12 (PyTorch + CUDA) |
| `python/3.11.13` | CPU-only Python 3.11 |
| `python/3.12.11` | CPU-only Python 3.12 |
| `python/3.13.5` | CPU-only Python 3.13 |
| `hpc_llm/gpu/1.0` | LLM-specific stack |
| `apptainer` | **1.4.1** |
| CUDA path | `/N/soft/sles15sp6/cuda/gnu/12.6/` |

### 7.4 Standard Module Loading Pattern

```bash
module purge
module load PrgEnv-gnu    2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || module load cudatoolkit/11.8 2>/dev/null
module load cudnn/9.10.1.4_cuda12 2>/dev/null || true
module load nccl/2.27.7-1 2>/dev/null || true

conda activate /N/slate/atsubhas/envs/<env_name> 2>/dev/null || source activate <env_name>
```

### 7.5 MPI

| Implementation | Notes |
|---------------|-------|
| HPE Cray MPI (MPICH-based) | Default on BR200, optimized for Slingshot |
| OpenMPI | Available, used in GPU partition benchmarks |
| MVAPICH | Available on Quartz |

### 7.6 Container Support (Apptainer)

```bash
# Pull to Slate (images are 5-15 GB)
apptainer pull /N/slate/atsubhas/containers/pytorch.sif docker://nvcr.io/nvidia/pytorch:24.01-py3

# Run with GPU passthrough
apptainer exec --nv /N/slate/atsubhas/containers/pytorch.sif python train.py

# Bind project directories
apptainer exec --nv \
  --bind /N/slate/atsubhas/project:/project \
  /N/slate/atsubhas/containers/pytorch.sif python /project/train.py

# Control GPU visibility inside container
APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1 apptainer exec --nv container.sif python script.py
```

---

## 8. Allocations and Access

### 8.1 IU's Open-Access Model

**IU does NOT use a traditional SU-billing model.** There is no SU balance that decrements to zero.
Instead:
- Allocations grant **access to partitions and resources**
- Job scheduling uses **SLURM fair-share** priority
- Heavy usage lowers your fair-share factor (lower priority), but you are never hard-blocked
- This is fundamentally different from XSEDE/ACCESS or Purdue Anvil

### 8.2 RT Projects Portal

All access is managed through [projects.rt.iu.edu](https://projects.rt.iu.edu/).

| Project Type | Created By | Expires | Notes |
|-------------|------------|---------|-------|
| Research | Faculty/Staff (PI) | June 30 annually | Students added by PI |
| Class | Instructors | End of semester | Students join for coursework |
| HPC for Students | Self-service | End of semester | Search PI "lamhuber", no advisor needed |

### 8.3 How to Get Access

1. Go to [projects.rt.iu.edu](https://projects.rt.iu.edu/)
2. Create a project (research description + PI username)
3. Request allocations for specific systems (BR200, Quartz, Slate)
4. RT staff processes the request (days, not weeks)
5. SSH to `bigred200.uits.iu.edu` or `quartz.uits.iu.edu`

### 8.4 Checking Usage

```bash
# Fair-share standing (how priority-penalized you are)
sshare -u atsubhas -A r00602

# Historical job accounting
sacct -u atsubhas --starttime=2026-01-01 --format=JobID,JobName,Partition,Elapsed,MaxRSS,State

# Association details (limits, QOS)
sacctmgr show assoc where user=atsubhas format=Account,Share,GrpTRESMins,MaxTRESMins,QOS

# Web dashboard
# https://one.iu.edu/task/iu/hpc-user-dashboard
```

### 8.5 Cost

- **BR200 and Quartz: Free** to all IU researchers (no dollar cost)
- **Slate-Project**: Free up to 15 TB; over 15 TB billed to department
- **Jetstream2**: Free via ACCESS allocations (NSF-funded)
- **SDA (tape archive)**: Free

---

## 9. CPU vs GPU Decision Framework

### 9.1 When CPUs Win (Use `general` Partition)

| Workload | Why CPU | Resource Request |
|----------|---------|-----------------|
| Data preprocessing (pandas, numpy, text/image transforms) | I/O-bound, no tensor ops; GPU idle waiting for disk | `--cpus-per-task=32 --mem=64G` on `general` |
| Small model inference (<10M params) | Kernel launch overhead dominates GPU | `--cpus-per-task=16 --mem=32G` |
| Compilation (C++, CUDA, Megatron builds) | CPU-bound, parallelizes with `make -j` | `--cpus-per-task=16 --mem=32G` (4-day limit!) |
| File format conversion (HDF5, NIfTI, DICOM) | I/O-bound, trivially parallelizable | Job arrays, 1-4 cores each |
| Feature extraction (scikit-learn, scipy) | Multi-threaded via OpenBLAS/MKL | Set `OMP_NUM_THREADS` = `--cpus-per-task` |
| Monte Carlo simulations (non-GPU MCX) | Embarrassingly parallel across seeds | Job arrays, one seed per task |
| Memory-intensive CPU work (>250 GB) | CPU nodes have 250 GB usable | **Use Quartz** (512 GB CPU nodes) or **BR200 GPU nodes** (500 GB usable, but costs GPU billing) |

### 9.2 When GPUs Win (Use `gpu` Partition)

- Model training with >1M parameters
- Batched matrix ops (>512x512 matrices)
- Anything with Tensor Core acceleration (convolutions, GEMMs, attention)
- MCX photon transport (GPU-native, massive speedup)
- Mixed-precision training (BF16/FP16 on A100 Tensor Cores)

### 9.3 Decision Heuristic

```
1. Profile with a short GPU run: nvidia-smi dmon -d 5
2. If GPU utilization < 30%: workload is CPU-appropriate
3. If GPU utilization 30-70%: optimize data loading, increase batch size
4. If GPU utilization > 70%: GPU is well-utilized, good fit
```

### 9.4 The "30 CPUs vs 1 GPU" Strategy

For embarrassingly parallel workloads, using 30 CPU cores on `general` can be **faster AND
more available** than waiting for a GPU:

```bash
# Instead of waiting 2 hours for 1 GPU to process 1000 files...
#SBATCH -p general
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 04:00:00        # 4-day limit on general!

# Use Python multiprocessing or GNU parallel
parallel -j 32 python process_file.py {} ::: /N/slate/atsubhas/data/raw/*.nii.gz
```

**Advantages:**
- `general` has 640 nodes (vs 64 GPU nodes) -- much less contention
- 4-day walltime (vs 2-day on GPU)
- No GPU fair-share penalty
- Often starts immediately while GPU queue has hours of wait

---

## 9A. CPU Optimization for EPYC 7742 (Zen 2)

### 9A.1 SMT (Hyperthreading) -- When to Disable

**Rule: Always use `--hint=nomultithread` for scientific Python/NumPy/BLAS workloads.**

| Workload | SMT Effect | Recommendation |
|----------|------------|---------------|
| Dense FP (GEMM, FFT, NumPy BLAS) | **-10 to -20%** | SMT OFF |
| AVX2-heavy (SciPy, convolutions) | **-10 to -20%** | SMT OFF |
| Memory-BW-bound (STREAM, stencils) | ~0% or negative | SMT OFF |
| Monte Carlo (large state per thread) | **-5 to -10%** | SMT OFF |
| I/O-bound (file processing, web) | **+20-40%** | SMT ON |
| Mixed integer + FP, many light tasks | **+10-25%** | SMT ON |

```bash
# Physical cores only (RECOMMENDED for scientific computing)
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=16        # 16 physical cores = 1 NUMA node
```

### 9A.2 NUMA Binding (Most Impactful -- 20-50% Improvement)

**Compute nodes have 8 NUMA nodes (dual-socket), not 4 like the login node!**

```bash
# 8 workers, one per NUMA node (16 cores x 32 GB each):
#SBATCH --ntasks=8 --cpus-per-task=16 --hint=nomultithread
srun --cpu-bind=cores python worker.py

# 2 workers, one per socket (64 cores x 128 GB each):
#SBATCH --ntasks=2 --cpus-per-task=64 --hint=nomultithread
srun --cpu-bind=sockets python worker.py

# Single process, bind to NUMA node 0 (memory-latency-sensitive)
numactl --cpunodebind=0 --membind=0 python script.py

# Interleave memory for bandwidth-bound workloads (streaming/STREAM-like)
numactl --interleave=all python script.py
```

### 9A.3 Thread Control (Prevent Oversubscription)

**Critical:** If you run 4 Python processes but BLAS also spawns 16 threads internally,
you get 64 threads fighting for 16 cores = 2-5x slowdown.

```bash
# For embarrassingly parallel Python (many processes, each single-threaded):
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# For single-process BLAS-heavy work (one big matrix multiply):
export OMP_NUM_THREADS=64           # use all physical cores
export OPENBLAS_NUM_THREADS=64
export OMP_PROC_BIND=spread         # spread across NUMA nodes
export OMP_PLACES=cores

# For hybrid (4 processes x 16 BLAS threads = 64 cores):
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
```

### 9A.4 MKL on AMD Fix

Intel MKL historically throttles on AMD CPUs. Fix:
```bash
export MKL_DEBUG_CPU_TYPE=5    # Force Skylake-X code path (best AVX2 on AMD)
```
Or better: use OpenBLAS instead of MKL:
```bash
conda install numpy "libblas=*=*openblas"
```

### 9A.5 Compiler Flags for EPYC 7742

```bash
# Recommended GCC flags
CFLAGS="-O3 -march=znver2 -mtune=znver2 -mavx2 -mfma -ftree-vectorize"

# Aggressive (with LTO and fast-math)
CFLAGS="-O3 -march=znver2 -mtune=znver2 -mavx2 -mfma -flto -ffast-math"

# Note: NO AVX-512 on Zen 2. Do NOT use -mavx512f.
```

### 9A.6 Multiprocessing Decision Matrix

| Method | Best For | NUMA-Aware | BLAS Control |
|--------|----------|-----------|--------------|
| `multiprocessing.Pool` | CPU-bound, single-node | Manual (`numactl`) | Set env before fork |
| `joblib` (loky backend) | scikit-learn, embarrass. parallel | Partial | Set env before import |
| GNU `parallel` | File-level CLI tasks | Yes (`numactl` per task) | Per-process env |
| `mpi4py` | Multi-node, tightly coupled | Yes (rank binding) | Per-rank env |
| `concurrent.futures` (threads) | I/O-bound (GIL released for I/O) | No | N/A |

### 9A.7 Environment Variables Template for CPU Jobs

```bash
# === Put in SBATCH scripts for CPU workloads ===

# Thread control (match to --cpus-per-task)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MKL_DEBUG_CPU_TYPE=5
export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS

# For embarrassingly parallel (override with 1):
# export OMP_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
```

### 9A.8 Available CPU Software (Verified on BR200)

| Module | Version | Notes |
|--------|---------|-------|
| `matlab` | 2025a | Full toolbox access |
| `julia` | 1.11.6 | Distributed.jl for multi-node |
| `openblas` | 0.3.26 | Optimized BLAS |
| `cray-libsci` | 22.12.1.1, 25.03.0 | Cray-optimized BLAS/LAPACK/ScaLAPACK |
| `cray-fftw` | 3.3.10.3, 3.3.10.10 | Optimized FFT |
| `cray-hdf5` | 1.12.2.1, 1.14.3.5 | Serial + parallel HDF5 |
| `cray-netcdf` | 4.9.0.1, 4.9.0.17 | NetCDF-4 |
| `boost` | 1.86.0 | C++ libraries |
| `openmpi` | 5.0.8 | **Warning: "Expect poor performance compared to cray-mpich"** |
| `cray-mpich` | 8.1.23 | **Default MPI, optimized for Slingshot** |
| `intel/oneapi/compiler` | 2025.2.0 | Intel compilers |
| `intel/oneapi/tbb` | 2022.2 | Threading Building Blocks |

### 9A.9 CPU Partition Current Availability (Snapshot 2026-03-19)

| State | Nodes |
|-------|-------|
| Idle | 11 |
| Mix (partially allocated) | 286 |
| Fully allocated | 335 |
| Reserved | 10 |

**638 total CPU nodes. Right now ~297 nodes have idle cores. CPU partition is less contended than GPU.**

### 9A.10 Performance Profiling on BR200

```bash
# Check IPC (instructions per cycle) -- <1.0 means memory-bound
perf stat -e instructions,cycles,cache-misses python script.py

# Check NUMA balance (remote access should be ~0%)
numastat -p $(pgrep python)

# Check thread placement
ps -eo pid,tid,psr,comm | grep python
```

---

## 10. SLURM Job Optimization

### 10.1 Job Arrays (Embarrassingly Parallel)

```bash
# Basic: 100 tasks, indices 0-99
#SBATCH --array=0-99

# Throttled: max 10 concurrent (backfill-friendly, filesystem-friendly)
#SBATCH --array=0-99%10

# Specific indices
#SBATCH --array=1,5,10,20

# Step pattern
#SBATCH --array=0-100:5     # 0,5,10,...,100
```

**Environment variables in each task:**
- `$SLURM_ARRAY_TASK_ID` -- the index
- `$SLURM_ARRAY_JOB_ID` -- shared job ID
- `$SLURM_ARRAY_TASK_COUNT` -- total tasks

**Pattern: Hyperparameter sweep**
```bash
#!/bin/bash
#SBATCH -J hpsweep
#SBATCH -A r00602
#SBATCH -p gpu
#SBATCH --array=0-49%8
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o logs/hpsweep-%A_%a.out

PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs/hparams.txt)
LR=$(echo "$PARAMS" | cut -d',' -f1)
BS=$(echo "$PARAMS" | cut -d',' -f2)
python train.py --lr "$LR" --batch-size "$BS" --run-id "$SLURM_ARRAY_TASK_ID"
```

**Pattern: Data processing across files**
```bash
#SBATCH --array=0-499%50       # MaxArraySize=500 on BR200!
#SBATCH -p general
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 00:30:00

FILES=($(ls /N/slate/atsubhas/data/raw/*.nii.gz))
INPUT="${FILES[$SLURM_ARRAY_TASK_ID]}"
python preprocess.py --input "$INPUT" --output /N/slate/atsubhas/data/processed/
```

### 10.2 Backfill Exploitation

The SLURM backfill scheduler fills gaps with lower-priority jobs that do not delay higher-priority
jobs. **Accurate, short time limits are the single most important factor.**

| Strategy | Impact | How |
|----------|--------|-----|
| **Tight walltime** | Highest | If job takes ~45 min, request `--time=01:00:00` not `--time=48:00:00` |
| **Use --time-min** | High | `--time=08:00:00 --time-min=02:00:00` -- scheduler picks from [2h,8h] |
| **Request fewer resources** | High | 1 GPU backfills far easier than 4 |
| **Explicit memory** | Medium | `--mem=32G` not `--mem=0` (which locks entire node) |
| **Decompose long jobs** | Medium | 8x 6h jobs chain better than 1x 48h job |
| **Use debug partitions** | High | `gpu-debug` (1h) starts nearly instantly |
| **Off-peak submission** | Medium | Submit Friday evening for weekend scheduling |

### 10.3 Dependency Chains (Multi-Stage Pipelines)

| Type | Triggers When | Use Case |
|------|---------------|----------|
| `afterok:JOBID` | Previous succeeded (exit 0) | Pipeline: preprocess -> train -> eval |
| `afterany:JOBID` | Previous ended (any exit) | Cleanup that always runs |
| `afternotok:JOBID` | Previous failed | Error notification, fallback |
| `aftercorr:JOBID` | Corresponding array task succeeded | Parallel array -> parallel array |
| `singleton` | No other job with same name+user running | Serialized recurring jobs |

**Pattern: Full ML pipeline**
```bash
#!/bin/bash
# submit_pipeline.sh

# Stage 1: Data preprocessing (CPU, 4-day limit)
JOB1=$(sbatch --parsable preprocess.sh)
echo "Preprocessing: $JOB1"

# Stage 2: Training (GPU), depends on preprocessing
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 train.sh)
echo "Training: $JOB2"

# Stage 3: Evaluation (GPU), depends on training
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 eval.sh)
echo "Evaluation: $JOB3"

# Stage 4: Cleanup (always runs)
JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 cleanup.sh)
echo "Cleanup: $JOB4"

echo "Pipeline submitted: $JOB1 -> $JOB2 -> $JOB3 -> $JOB4"
```

**Combining dependencies:**
```bash
--dependency=afterok:111,afterok:222    # Both must succeed (AND)
--dependency=afterok:111?afterok:222    # Either succeeding is enough (OR)
```

### 10.4 Queue Wait Time Minimization

Priority is calculated as weighted sum of: fairshare, age, job size, partition, QOS.

**Actionable strategies (ordered by impact):**

1. **Shortest walltime** -- #1 lever for backfill
2. **Smallest resource footprint** -- 1 GPU instead of 4 if scaling is poor
3. **Use --time-min** -- gives scheduler flexibility
4. **Off-peak submission** -- weekends and nights (US Eastern)
5. **gpu-debug for tests** -- nearly instant, 1 hour
6. **Check estimated start**: `squeue -u atsubhas --start`
7. **Check partition load**: `sinfo -p gpu -o "%20P %5a %10l %6D %6t"`

### 10.5 Resource Right-Sizing

**After a job completes, measure actual usage:**
```bash
sacct -j JOBID --format=JobID,JobName,Elapsed,MaxRSS,ReqMem,NCPUS,CPUTime,State
```

| Resource | Rule of Thumb |
|----------|--------------|
| GPUs | Request 1 unless multi-GPU scaling is verified |
| Memory | 1.5-2x measured MaxRSS |
| CPUs | Match `--cpus-per-task` to `num_workers + 1` |
| Walltime | 1.5x measured runtime |
| Nodes | 1 unless inter-node scaling is proven |

**Over-requesting hurts:** SLURM's fairshare penalizes your account proportional to resources
allocated (not just used). Wasting memory or GPUs burns fairshare.

---

## 11. GPU Utilization Optimization

### 11.1 Multi-GPU Strategy Decision Tree

```
Model fits in 1 GPU (params + activations + optimizer < 35GB)?
  YES -> Use 1 GPU
    Batch size too small?
      -> Use gradient accumulation (no extra memory)
    Want more throughput?
      -> Use DDP with 4 GPUs (single node, NVLink)
  NO -> Model does NOT fit in 1 GPU
    -> Use FSDP (FULL_SHARD) across 4 GPUs
    Still does not fit?
      -> Multi-node FSDP (Slingshot, slower)
```

### 11.2 DDP (DistributedDataParallel) -- Default Choice

Each GPU holds full model replica. Gradient all-reduce over NVLink.

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train.py
```

### 11.3 FSDP (Fully Sharded Data Parallel) -- Large Models

Shards parameters + gradients + optimizer states across GPUs.
- ~58% memory reduction vs DDP (23GB DDP -> 9.6GB FSDP on A100-40GB)
- ~25% speed penalty vs DDP (3.19 vs 4.31 iter/s)

### 11.4 Multi-Node Launch Pattern

```bash
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4

srun --ntasks="$NUM_NODES" --ntasks-per-node=1 \
    bash -lc "torchrun \
        --nnodes=$NUM_NODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train.py [args]"
```

### 11.5 Mixed Precision (BF16 on A100)

```python
# BF16 -- no loss scaling needed (unlike FP16)
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

# Benefits: 2x memory for activations, ~2x throughput from Tensor Cores
```

### 11.6 Activation Checkpointing

Trade ~30% speed for ~60% memory savings:
```python
from torch.utils.checkpoint import checkpoint
output = checkpoint(transformer_block, input, use_reentrant=False)
```

### 11.7 Gradient Accumulation

Simulate larger batch sizes without more memory:
```python
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()
```

### 11.8 PyTorch Performance Checklist

- [ ] `torch.backends.cudnn.benchmark = True` (conv autotuner)
- [ ] `pin_memory=True` in DataLoader
- [ ] `num_workers=4-8` per GPU (set `--cpus-per-task` accordingly)
- [ ] `torch.compile(model)` for kernel fusion (PyTorch 2.0+)
- [ ] Create tensors on GPU directly: `torch.randn(size, device='cuda')`
- [ ] `channels_last` memory format for CNNs
- [ ] Tensor dimensions as multiples of 8 (Tensor Core alignment)
- [ ] `set_to_none=True` in `optimizer.zero_grad()`

### 11.9 GPU Monitoring in Jobs

```bash
# Add to SBATCH script to log GPU metrics every 10 seconds
nvidia-smi dmon -s pucvmet -d 10 -o DT > gpu_monitor_${SLURM_JOB_ID}.txt &
MONITOR_PID=$!
# ... run training ...
kill $MONITOR_PID
```

---

## 12. Checkpointing and Fault Tolerance

### 12.1 SLURM Signal-Based Checkpointing

```bash
#SBATCH --signal=B:USR1@300    # Send SIGUSR1 300 seconds before walltime
#SBATCH --requeue               # Auto-requeue after checkpoint
```

### 12.2 Python Signal Handler

```python
import signal
import sys
import os

def checkpoint_handler(signum, frame):
    """Save checkpoint and exit cleanly on SLURM signal."""
    print(f"Received signal {signum}, saving checkpoint...")
    save_checkpoint(
        model, optimizer, scheduler, epoch, step,
        path=os.path.join(os.environ.get("CKPT", "."), "interrupt_ckpt.pt")
    )
    sys.exit(0)

signal.signal(signal.SIGUSR1, checkpoint_handler)
signal.signal(signal.SIGTERM, checkpoint_handler)
```

### 12.3 Robust Checkpoint Save/Load

```python
def save_checkpoint(model, optimizer, scheduler, epoch, step, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }, path)

def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    torch.random.set_rng_state(ckpt["rng_state"])
    torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    return ckpt["epoch"], ckpt["step"]
```

### 12.4 Auto-Resubmission Chain

```bash
#!/bin/bash
#SBATCH -J longtrain
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH -t 06:00:00

# Training script handles checkpoint/resume internally
python train.py --checkpoint-dir "$CKPT" --auto-resume
```

Or explicit chain:
```bash
# submit_chain.sh
JOB=$(sbatch --parsable train_segment.sh)
for i in $(seq 2 8); do
    JOB=$(sbatch --parsable --dependency=afterany:$JOB train_segment.sh)
done
echo "Chain submitted: 8 segments"
```

### 12.5 Checkpoint Best Practices

- Save every N steps AND on SLURM signal
- Keep only last 3 checkpoints (save disk space)
- Save RNG states for exact reproducibility on resume
- Write to Slate (`$CKPT`), not Scratch (30-day purge!)
- Use `torch.save` with `_use_new_zipfile_serialization=True` (default in PyTorch 2.x)

---

## 13. Storage and I/O Optimization

### 13.1 Stage Data to Node-Local tmpfs (RAM-Backed, NOT SSD)

```bash
# Copy medium datasets to node-local storage at job start
LOCAL_DATA="/tmp/${SLURM_JOB_ID}/data"
mkdir -p "$LOCAL_DATA"
cp -r /N/slate/atsubhas/project/data/processed/ "$LOCAL_DATA/"

# Point training at local data
python train.py --data-dir "$LOCAL_DATA"

# Cleanup at end
trap "rm -rf /tmp/${SLURM_JOB_ID}" EXIT
```

### 13.2 Lustre Stripe Tuning

```bash
# Parallelize I/O across Lustre OSTs for large file directories
lfs setstripe -c 4 /N/slate/atsubhas/project/checkpoints/
# stripe count 4 is good for multi-GB checkpoint files
# Higher counts for very large files only
```

### 13.3 Avoid Small File Pathologies

Lustre excels at large sequential I/O, NOT metadata-heavy operations.
If you have 1M+ small files:
- Use HDF5, WebDataset, or tar archives instead
- `webdataset` library for PyTorch: reads tar files as streaming datasets
- Convert image folders to LMDB or HDF5 for training

### 13.4 RAM-Backed Temporary Storage

```bash
# Ultra-fast temp storage (limited by node RAM)
export TMPDIR=/dev/shm/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
```

### 13.5 DataLoader Optimization

```python
dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,         # Match to --cpus-per-task
    pin_memory=True,       # Faster CPU-to-GPU transfer
    prefetch_factor=2,     # Pre-load 2 batches per worker
    persistent_workers=True,  # Keep workers alive between epochs
)
```

---

## 14. Environment Management

### 14.1 Conda on Slate (Not Home)

```bash
# Create env on Slate to avoid Home quota
conda create --prefix /N/slate/atsubhas/envs/myenv python=3.11 -y

# Activate
conda activate /N/slate/atsubhas/envs/myenv

# Export for reproducibility
conda env export --prefix /N/slate/atsubhas/envs/myenv --from-history > environment.yml

# Recreate
conda env create --prefix /N/slate/atsubhas/envs/newenv --file environment.yml
```

### 14.2 Tips

- Use `--prefix` for envs on Slate (not default `~/.conda/envs/`)
- Use `mamba` or `conda` with `libmamba` solver for 10-50x faster resolution
- Pin CUDA packages: `pytorch=*=*cu126*`
- Export with `--from-history` for portable YAMLs
- Install envs from login node, not compute node (avoids wasting allocation)

### 14.3 Module + Conda Stacking

```bash
# Always purge first to avoid conflicts
module purge
module load PrgEnv-gnu 2>/dev/null || true
module load cudatoolkit/12.6
module load cudnn/9.10.1.4_cuda12
module load nccl/2.27.7-1
conda activate /N/slate/atsubhas/envs/myenv
```

### 14.4 When to Use Containers (Apptainer)

- Reproducibility across clusters (BR200, Quartz, external systems)
- Complex dependency chains that break conda
- NGC containers for optimized framework builds (e.g., Megatron-LM)
- Collaborator needs exact same environment

---

## 15. Common Pitfalls and Debugging

### 15.1 OOM (Out of Memory)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `slurmstepd: Exceeded memory limit` | `--mem` too low | Increase `--mem`; profile with `sacct --format=MaxRSS` |
| `CUDA out of memory` | GPU OOM | Reduce batch size, enable BF16, activation checkpointing, FSDP |
| Linux OOM killer (no SLURM error) | Other processes on node | Specify explicit `--mem`, not `--mem=0` |

### 15.2 NCCL Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `NCCL WARN Cuda failure 'out of memory'` | GPU memory exhaustion | Reduce batch size, enable AMP |
| `Call to ibv_reg_mr failed` | Insufficient pinned memory | Check `ulimit -l` (should be unlimited) |
| Timeout during `init_process_group` | Wrong MASTER_ADDR or port | Use `scontrol show hostnames`; try different port |
| `No such device` | Wrong network interface | Set `NCCL_SOCKET_IFNAME=hsn` |
| Slow multi-node training | TCP instead of IB | Verify `NCCL_IB_DISABLE` is NOT set |

**Debug with:**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL    # Very verbose
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
```

### 15.3 Filesystem Quotas

```bash
# Check all quotas
quota -s                              # Home
lfs quota -hu atsubhas /N/slate/      # Slate
lfs quota -hu atsubhas /N/scratch/    # Scratch
```

**Common cause:** conda/pip/HuggingFace caches filling Home.
**Fix:** Symlink caches to Slate (see Section 5.2).

### 15.4 Other Common Issues

| Pitfall | Prevention |
|---------|-----------|
| Forgetting `-A r00602` | Always include in SBATCH header |
| Job stuck PENDING (Priority) | Check `sshare -u atsubhas`; reduce resource requests |
| Training divergence after resume | Save/restore RNG states + LR scheduler state |
| Data loading bottleneck (GPU ~30%) | Increase `num_workers`, use `pin_memory=True` |
| Stale conda env on compute | Activate after `module load`, not on login node only |
| Scratch data disappeared | 30-day purge! Use Slate for anything you need to keep |

---

## 16. Jetstream2 Cloud Computing

### 16.1 Overview

| Attribute | Value |
|-----------|-------|
| Host | Indiana University Bloomington |
| Hardware | AMD Milan 7713, 128 cores/node, 512 GB RAM |
| GPU nodes | 90 nodes, 4x A100 each (360 total A100s) |
| Large memory | Up to 1 TB RAM per VM |
| Storage | 17.2 PB |
| Access | Via ACCESS allocations (NOT RT Projects) |
| Interface | Exosphere, OpenStack Horizon, CLI |

### 16.2 GPU Instance Options

| Flavor | vCPUs | RAM | GPU | SU/hour |
|--------|-------|-----|-----|---------|
| g3.medium | 8 | 30 GB | 25% A100 (10 GB) | 16 |
| g3.large | 16 | 60 GB | 50% A100 (20 GB) | 32 |
| g3.xl | 32 | 120 GB | 1x full A100 (40 GB) | 64 |
| g3.2xl+ | 64+ | 240+ GB | 2+ A100 | 128+ |

### 16.3 Trial Access (No Cost, No Proposal)

- 90 days, 1 small VM, 10 GB storage, CPU only
- Register at [jetstream-cloud.org](https://jetstream-cloud.org)
- Resources deleted 10 days after expiration

---

## 17. ACCESS Allocations

### 17.1 Tiers

| Tier | Credits | Proposal | Review | Timeline |
|------|---------|----------|--------|----------|
| **Explore** | 400,000 | 1 page | Eligibility check | ~2 weeks |
| **Discover** | 1,500,000 | 1 page | Suitability review | ~2 weeks |
| **Accelerate** | 3,000,000 | 3 pages | Panel merit review | ~2 weeks |
| **Maximize** | No cap | 10 pages + 5 pages code perf | AARC committee | Semi-annual |

### 17.2 PI Eligibility

- US-based researcher/educator at graduate level+
- Institutional email required
- Grad students: eligible for Explore/Discover with advisor co-PI
- NSF GRFP holders: can lead without advisor letter

### 17.3 Jetstream2 SU Rates

1 ACCESS Credit = 1 Jetstream2 SU.

| Resource | Rate |
|----------|------|
| CPU (m3.*) | 1 SU per vCPU-hour |
| Large memory (r3.*) | 2 SU per vCPU-hour |
| GPU A100 (g3.*) | 2 SU per vCPU-hour |

### 17.4 Maximize Review Windows (2025-2027)

| Cycle | Submit | Awards Start |
|-------|--------|-------------|
| 1 | Dec 15 2025 - Jan 31 2026 | Apr 1 2026 |
| 2 | Jun 15 - Jul 31 2026 | Oct 1 2026 |
| 3 | Dec 15 2026 - Jan 31 2027 | Apr 1 2027 |

---

## 18. Support Channels

| Channel | Details |
|---------|---------|
| Help Form | [projects.rt.iu.edu/help/?queue=hps](https://projects.rt.iu.edu/help/?queue=hps) |
| Office Hours | Wednesdays 2-3pm ET via Zoom |
| Email | radl@iu.edu (Research Applications & Deep Learning) |
| Contact RT | [uits.iu.edu/.../contact-research-technologies](https://uits.iu.edu/services/technology-for-research/support/contact-research-technologies.html) |
| IU Research Data Commons | iurdc@iu.edu |
| Software Requests | [uits.iu.edu/.../hpc-software-request](https://uits.iu.edu/services/technology-for-research/support/hpc-software-request.html) |
| Knowledge Base | [kb.iu.edu](https://kb.iu.edu) |
| IT Training | [ittraining.iu.edu](https://ittraining.iu.edu) |

---

## 19. Ready-to-Use Templates

### 19.1 GPU Training Job (BR200)

```bash
#!/bin/bash
#SBATCH -J train_model
#SBATCH -A r00602
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=400G                    # GPU nodes have 500 GB usable!
#SBATCH -t 12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

# Modules
module purge
module load PrgEnv-gnu 2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || module load cudatoolkit/11.8 2>/dev/null
module load nccl/2.27.7-1 2>/dev/null || true

# Environment
conda activate /N/slate/atsubhas/envs/myenv

# Project layout
export PROJ="/N/slate/atsubhas/<project>"
export CKPT="$PROJ/checkpoints"
export DATA="$PROJ/data"
mkdir -p "$CKPT"

# NCCL settings
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

# GPU monitoring (background)
nvidia-smi dmon -s pucvmet -d 30 -o DT > "$PROJ/logs/gpu_${SLURM_JOB_ID}.csv" &

# Launch training
torchrun --nproc_per_node=4 train.py \
    --data-dir "$DATA" \
    --checkpoint-dir "$CKPT" \
    --auto-resume
```

### 19.2 CPU Preprocessing Job (BR200 or Quartz)

```bash
#!/bin/bash
#SBATCH -J preprocess
#SBATCH -A r00602
#SBATCH -p general
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00     # 1 day
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail
module purge
conda activate /N/slate/atsubhas/envs/myenv

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python preprocess.py --workers $SLURM_CPUS_PER_TASK --input /N/slate/atsubhas/data/raw/
```

### 19.3 Job Array (Hyperparameter Sweep)

```bash
#!/bin/bash
#SBATCH -J hpsweep
#SBATCH -A r00602
#SBATCH -p gpu
#SBATCH --array=0-49%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/hpsweep-%A_%a.out
#SBATCH -e logs/hpsweep-%A_%a.err

set -euo pipefail
module purge
module load cudatoolkit/12.6 2>/dev/null || true
conda activate /N/slate/atsubhas/envs/myenv

PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" configs/hparams.csv)
LR=$(echo "$PARAMS" | cut -d',' -f1)
BS=$(echo "$PARAMS" | cut -d',' -f2)
WD=$(echo "$PARAMS" | cut -d',' -f3)

python train.py --lr "$LR" --batch-size "$BS" --weight-decay "$WD" \
    --run-id "sweep_${SLURM_ARRAY_TASK_ID}"
```

### 19.4 Multi-Node Distributed Training (BR200)

```bash
#!/bin/bash
#SBATCH -J distributed
#SBATCH -A r00602
#SBATCH -p gpu
#SBATCH -N 4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -t 1-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

module purge
module load PrgEnv-gnu 2>/dev/null || true
module load cudatoolkit/12.6
module load nccl/2.27.7-1

conda activate /N/slate/atsubhas/envs/myenv

export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=hsn

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4

srun --ntasks="$NUM_NODES" --ntasks-per-node=1 \
    bash -lc "torchrun \
        --nnodes=$NUM_NODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train.py --auto-resume --checkpoint-dir /N/slate/atsubhas/project/checkpoints/"
```

### 19.5 Interactive GPU Session

```bash
# Quick debug (1 hour, starts fast)
salloc -A r00602 -p gpu-debug --gres=gpu:1 --cpus-per-task=8 --mem=16G -t 00:30:00

# Extended interactive (4 hours)
salloc -A r00602 -p gpu-interactive --gres=gpu:1 --cpus-per-task=16 --mem=32G -t 04:00:00

# Once allocated, set up environment:
module purge && module load cudatoolkit/12.6
conda activate /N/slate/atsubhas/envs/myenv
python -c "import torch; print(torch.cuda.get_device_name())"
```

### 19.6 Jupyter on BR200

```bash
# On BR200 (after salloc on gpu-interactive):
jupyter lab --no-browser --port=8888 --ip=0.0.0.0

# From local machine (new terminal):
ssh -L 8888:COMPUTE_NODE:8888 br200
# Then open http://localhost:8888 in browser
```

### 19.7 Pipeline Submission Script

```bash
#!/bin/bash
# submit_pipeline.sh -- Submit a full ML pipeline as dependent jobs
set -euo pipefail

PROJECT="myproject"
LOGDIR="/N/slate/atsubhas/${PROJECT}/logs"
mkdir -p "$LOGDIR"

echo "Submitting pipeline for project: $PROJECT"

# Stage 1: Preprocessing (CPU, Quartz for more memory)
JOB1=$(sbatch --parsable \
    -J "${PROJECT}_preproc" \
    -A r00602 -p general \
    --cpus-per-task=64 --mem=200G -t 12:00:00 \
    -o "${LOGDIR}/preproc-%j.out" \
    preprocess.sh)
echo "  Preprocess: $JOB1"

# Stage 2: Training (GPU)
JOB2=$(sbatch --parsable \
    --dependency=afterok:$JOB1 \
    -J "${PROJECT}_train" \
    -A r00602 -p gpu \
    --gres=gpu:4 --cpus-per-task=32 --mem=200G -t 2-00:00:00 \
    --signal=B:USR1@300 \
    -o "${LOGDIR}/train-%j.out" \
    train.sh)
echo "  Training: $JOB2"

# Stage 3: Evaluation (1 GPU)
JOB3=$(sbatch --parsable \
    --dependency=afterok:$JOB2 \
    -J "${PROJECT}_eval" \
    -A r00602 -p gpu \
    --gres=gpu:1 --cpus-per-task=8 --mem=32G -t 02:00:00 \
    -o "${LOGDIR}/eval-%j.out" \
    eval.sh)
echo "  Evaluation: $JOB3"

echo "Pipeline: $JOB1 -> $JOB2 -> $JOB3"
echo "Monitor: squeue -u atsubhas"
```

### 19.8 Memory-Intensive CPU Job (Quartz)

```bash
#!/bin/bash
#SBATCH -J big_memory
#SBATCH -A r00602
#SBATCH -p general
#SBATCH -N 1
#SBATCH --cpus-per-task=128
#SBATCH --mem=480G           # Quartz general nodes have 512 GB
#SBATCH -t 2-00:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail
module purge
conda activate /N/slate/atsubhas/envs/myenv

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python memory_intensive_analysis.py
```

---

## 20. On-Cluster Verification Commands

Run these on BR200/Quartz to fill in gaps and verify current state:

```bash
# --- Partitions (authoritative) ---
sinfo -o "%20P %10a %10l %6D %8c %10m %25f %G"

# --- QOS configuration ---
sacctmgr show qos format=Name,Priority,MaxWall,MaxTRESPerUser,MaxTRESPerAccount

# --- Your account associations and limits ---
sacctmgr show assoc where user=atsubhas format=Account,Share,GrpTRESMins,MaxTRESMins,QOS

# --- Fair-share standing ---
sshare -u atsubhas -A r00602

# --- Scheduling priority weights ---
sprio -w

# --- SLURM scheduler configuration ---
scontrol show config | grep -E "Priority|Fairshare|Backfill|Preempt|SchedulerType"

# --- Available modules ---
module spider | head -100
module spider python
module spider cuda

# --- Network interface details ---
ibstat 2>/dev/null || echo "No IB"
ip link show | grep -E "ib|mlx|hsn"

# --- Storage quotas ---
quota -s
lfs quota -hu atsubhas /N/slate/
lfs quota -hu atsubhas /N/scratch/

# --- Recent job efficiency ---
sacct -u atsubhas --starttime=$(date -d '7 days ago' +%Y-%m-%d) \
    --format=JobID,JobName,Partition,Elapsed,MaxRSS,ReqMem,AllocGRES,State
```

---

## 21. Practical Workflow Recipes

### 21.1 Quick Decision Flowchart: "What Should I Submit?"

```
START: What kind of work?
  |
  +--> Training a neural net?
  |      YES --> Model fits 1 GPU (<35GB params+optim)?
  |                YES --> sbatch -p gpu --gres=gpu:1 -t 12:00:00
  |                NO  --> Need multi-GPU (FSDP)? --> sbatch -p gpu --gres=gpu:4
  |                        Need multi-node? --> sbatch -p gpu -N 4 --gres=gpu:4
  |
  +--> Processing files (NIfTI, DICOM, images)?
  |      How many files?
  |        <100   --> Single job, 32-64 cores: -p general --cpus-per-task=64
  |        100+   --> Job array: -p general --array=0-N%50 --cpus-per-task=1
  |
  +--> Hyperparameter sweep?
  |      --> Job array on gpu: --array=0-49%8 --gres=gpu:1 per task
  |
  +--> Compiling software / building envs?
  |      --> -p general --cpus-per-task=32 -t 02:00:00
  |      --> DO NOT waste GPU for compilation!
  |
  +--> Quick test / debugging?
  |      GPU: salloc -p gpu-debug --gres=gpu:1 -t 00:30:00
  |      CPU: salloc -p debug -t 00:30:00
  |
  +--> Memory-hungry (>250 GB)?
  |      --> Use Quartz (512 GB nodes) or BR200 GPU nodes (512 GB)
  |      --> GPU nodes give 512 GB even if you don't use GPUs!
  |          BUT: costs 16 billing units per unused GPU
  |
  +--> Running MATLAB?
         --> module load matlab/2025a
         --> -p general --cpus-per-task=16 (parpool size)
```

### 21.2 The "Smart Defaults" SBATCH Template

Copy this into every new job script and adjust only what you need:

```bash
#!/bin/bash
#============================================================
# SMART DEFAULTS -- Adjust the USER SETTINGS block only
#============================================================
# --- CHANGE THESE 6 LINES, THEN SUBMIT ---
#SBATCH -J my_job
#SBATCH -A r00602
#SBATCH -p general                    # general | gpu | gpu-debug | debug
#SBATCH -N 1
#SBATCH --cpus-per-task=64            # physical cores (1-128 CPU, 1-64 GPU)
#SBATCH --mem=128G                    # explicit! (CPU: max 250G, GPU: max 500G)
#SBATCH -t 04:00:00                   # max: general=4d, gpu=2d
#SBATCH --hint=nomultithread          # physical cores only (always for scientific compute)
#SBATCH --signal=B:USR1@300           # checkpoint 5 min before walltime
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
## Uncomment for GPU jobs:
## #SBATCH --gres=gpu:4

set -euo pipefail
mkdir -p logs

# --- MODULES ---
module purge
# Uncomment for GPU jobs:
# module load cudatoolkit/12.6 2>/dev/null || true
# module load nccl/2.27.7-1 2>/dev/null || true

# --- ENVIRONMENT ---
conda activate /N/slate/atsubhas/envs/myenv

# --- THREAD CONTROL (prevents BLAS oversubscription) ---
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_DEBUG_CPU_TYPE=5
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# For embarrassingly parallel (many single-threaded processes):
# export OMP_NUM_THREADS=1 && export OPENBLAS_NUM_THREADS=1

# --- PROJECT LAYOUT ---
export PROJ="/N/slate/atsubhas/my_project"
mkdir -p "${PROJ}/logs" "${PROJ}/checkpoints" "${PROJ}/data"

# ============================================================
# YOUR CODE BELOW
# ============================================================
python train.py
```

### 21.3 Compute Node Local Storage (Verified on Node)

**Both `/tmp` and `/dev/shm` on compute nodes are 126 GB tmpfs (RAM-backed).**
This is NOT disk -- it comes out of the node's 256 GB RAM.

```bash
# Stage input data to node-local RAM-disk for max I/O speed
LOCAL=/tmp/${SLURM_JOB_ID}
mkdir -p ${LOCAL}
trap "rm -rf ${LOCAL}" EXIT   # cleanup on exit

# Copy dataset (~10 GB, takes ~5 seconds from RAM)
cp -r /N/slate/atsubhas/data/processed/ ${LOCAL}/data/

# Point training at local data
python train.py --data-dir ${LOCAL}/data/

# WARNING: /tmp + /dev/shm share the same 126 GB RAM pool
# If your data + /dev/shm + OS > 126 GB, you'll OOM
# Budget: ~100 GB usable for staging (keep 26 GB for OS + buffers)
```

### 21.4 Compute Node NUMA Topology (Verified -- DUAL SOCKET)

**Critical correction:** Login node is single-socket (4 NUMA nodes). Compute nodes are
**dual-socket with 8 NUMA nodes**, each ~32 GB:

```
Node 0: CPUs 0-15, 128-143    (~32 GB) -- Socket 0
Node 1: CPUs 16-31, 144-159   (~32 GB) -- Socket 0
Node 2: CPUs 32-47, 160-175   (~32 GB) -- Socket 0
Node 3: CPUs 48-63, 176-191   (~32 GB) -- Socket 0
Node 4: CPUs 64-79, 192-207   (~32 GB) -- Socket 1
Node 5: CPUs 80-95, 208-223   (~32 GB) -- Socket 1
Node 6: CPUs 96-111, 224-239  (~32 GB) -- Socket 1
Node 7: CPUs 112-127, 240-255 (~32 GB) -- Socket 1

Physical cores: 0-127 (128 total, 64 per socket)
SMT siblings: 128-255
```

**Optimal binding patterns:**
```bash
# 8 independent workers, one per NUMA (32 GB each, 16 cores each):
#SBATCH --ntasks=8 --cpus-per-task=16 --hint=nomultithread
srun --cpu-bind=cores python worker.py

# 2 workers, one per socket (128 GB each, 64 cores each):
#SBATCH --ntasks=2 --cpus-per-task=64 --hint=nomultithread
srun --cpu-bind=sockets python worker.py

# Single process using all resources with interleaved memory:
numactl --interleave=all python big_matrix.py
```

### 21.5 Queue Strategy: GPU vs CPU Right Now (Live Snapshot 2026-03-19)

| Metric | GPU Partition | CPU (general) Partition |
|--------|--------------|----------------------|
| Running jobs | 71 | 498 |
| Pending jobs | **111** | **1,440** |
| Total nodes | 62 | 638 |
| Nodes per pending job | 0.56 | 0.44 |
| Typical wait (Priority) | Minutes to hours | Often immediate |

**Takeaway:** GPU queue is tighter per-node (more contention per resource unit), but CPU
queue has more absolute pending jobs because it's the default for everything. CPU jobs under
64 cores + 128 GB typically start within minutes because so many nodes are in `mix` state.

### 21.6 Recipe: Brain Imaging Preprocessing Pipeline (Your DOT/MCX Workflow)

```bash
#!/bin/bash
# submit_brain_pipeline.sh -- Full brain imaging preprocessing
# Stage 1: CPU preprocessing (NIfTI conversion, registration, segmentation)
# Stage 2: MCX photon transport (GPU)
# Stage 3: CPU postprocessing (reconstruction, analysis)

set -euo pipefail
PROJ="/N/slate/atsubhas/brain_imaging"
LOGDIR="${PROJ}/logs"
mkdir -p "${LOGDIR}"

# Stage 1: CPU preprocessing -- 64 cores, 4-day limit
JOB1=$(sbatch --parsable <<'PREPROC'
#!/bin/bash
#SBATCH -J brain_preproc
#SBATCH -A r00602 -p general
#SBATCH --cpus-per-task=64 --hint=nomultithread
#SBATCH --mem=200G -t 1-00:00:00
#SBATCH -o /N/slate/atsubhas/brain_imaging/logs/preproc-%j.out
set -euo pipefail
module purge
conda activate /N/slate/atsubhas/envs/neuro

export OMP_NUM_THREADS=1   # many files, each single-threaded
export OPENBLAS_NUM_THREADS=1

# Stage data to node-local tmpfs for fast I/O
LOCAL=/tmp/${SLURM_JOB_ID}
mkdir -p ${LOCAL}
trap "rm -rf ${LOCAL}" EXIT
cp -r /N/slate/atsubhas/brain_imaging/data/raw/ ${LOCAL}/raw/

# Process all subjects in parallel (64 workers)
find ${LOCAL}/raw/ -name "*.nii.gz" | \
    parallel --jobs 64 --bar \
    'python /N/slate/atsubhas/brain_imaging/scripts/preprocess.py \
        --input {} --output /N/slate/atsubhas/brain_imaging/data/processed/'
PREPROC
)
echo "Stage 1 (preproc): ${JOB1}"

# Stage 2: MCX simulation -- GPU (depends on preprocessing)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} <<'MCX'
#!/bin/bash
#SBATCH -J brain_mcx
#SBATCH -A r00602 -p gpu
#SBATCH --gres=gpu:4 --cpus-per-task=16
#SBATCH --mem=200G -t 12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /N/slate/atsubhas/brain_imaging/logs/mcx-%j.out
set -euo pipefail
module purge && module load cudatoolkit/12.6
conda activate /N/slate/atsubhas/envs/neuro

python /N/slate/atsubhas/brain_imaging/scripts/run_mcx.py \
    --input-dir /N/slate/atsubhas/brain_imaging/data/processed/ \
    --output-dir /N/slate/atsubhas/brain_imaging/data/mcx_output/ \
    --gpus 4
MCX
)
echo "Stage 2 (MCX): ${JOB2}"

# Stage 3: Reconstruction -- CPU (depends on MCX)
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} <<'RECON'
#!/bin/bash
#SBATCH -J brain_recon
#SBATCH -A r00602 -p general
#SBATCH --cpus-per-task=64 --hint=nomultithread
#SBATCH --mem=200G -t 12:00:00
#SBATCH -o /N/slate/atsubhas/brain_imaging/logs/recon-%j.out
set -euo pipefail
module purge
conda activate /N/slate/atsubhas/envs/neuro

export OMP_NUM_THREADS=64   # single large matrix solve
export OPENBLAS_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

python /N/slate/atsubhas/brain_imaging/scripts/reconstruct.py \
    --mcx-dir /N/slate/atsubhas/brain_imaging/data/mcx_output/ \
    --output-dir /N/slate/atsubhas/brain_imaging/results/
RECON
)
echo "Stage 3 (recon): ${JOB3}"

echo "Pipeline: ${JOB1} -> ${JOB2} -> ${JOB3}"
echo "Monitor: watch -n 30 squeue -u atsubhas"
```

### 21.7 Recipe: ML Training with Auto-Retry and Profiling

```bash
#!/bin/bash
#SBATCH -J ml_train
#SBATCH -A r00602 -p gpu
#SBATCH --gres=gpu:4 --cpus-per-task=32
#SBATCH --mem=400G -t 2-00:00:00
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH -o logs/%x-%j.out -e logs/%x-%j.err
set -euo pipefail

module purge
module load cudatoolkit/12.6 nccl/2.27.7-1
conda activate /N/slate/atsubhas/envs/myenv

export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8    # 32 cpus / 4 gpus = 8 per GPU for DataLoader

# GPU monitoring (background)
nvidia-smi dmon -s pucvmet -d 30 -o DT > logs/gpu_${SLURM_JOB_ID}.csv &

# Stage dataset to local tmpfs if small enough
if [ $(du -sm /N/slate/atsubhas/data/train/ | cut -f1) -lt 80000 ]; then
    echo "Staging data to /tmp..."
    LOCAL=/tmp/${SLURM_JOB_ID}/data
    mkdir -p ${LOCAL}
    cp -r /N/slate/atsubhas/data/train/ ${LOCAL}/
    DATA_DIR=${LOCAL}/train
else
    DATA_DIR=/N/slate/atsubhas/data/train
fi

torchrun --nproc_per_node=4 train.py \
    --data-dir ${DATA_DIR} \
    --checkpoint-dir /N/slate/atsubhas/checkpoints/ \
    --auto-resume \
    --workers-per-gpu 8
```

### 21.8 Maximizing Throughput: Multi-Node CPU for Embarrassingly Parallel

When you have 10,000+ independent tasks, use job arrays across multiple nodes:

```bash
#!/bin/bash
#SBATCH -J mass_process
#SBATCH -A r00602 -p general
#SBATCH --array=0-499%100       # 500 tasks, 100 concurrent (max array size!)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --hint=nomultithread
#SBATCH -t 01:00:00
#SBATCH -o logs/batch-%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Each array task processes a chunk of files
TOTAL_FILES=10000
CHUNK_SIZE=$((TOTAL_FILES / SLURM_ARRAY_TASK_COUNT))
START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))

python process_chunk.py --start ${START} --count ${CHUNK_SIZE}

# This runs 100 nodes x 4 cores = 400 cores simultaneously
# At 1 hour per chunk, 500 chunks = all done in ~5 hours
# vs single-node 64 cores: 10000/64 tasks * 1 min = ~2.6 hours
# Tradeoff: array is more robust (individual retries) but more scheduling overhead
```

### 21.9 Post-Job Analysis: Did I Waste Resources?

```bash
# Run after EVERY job to learn your actual resource usage
sacct -j JOBID --format=JobID,Elapsed,MaxRSS,ReqMem,AllocCPUS,AllocGRES,State,MaxDiskRead,MaxDiskWrite

# Quick efficiency check script -- save as ~/bin/job_eff.sh
#!/bin/bash
JOB=$1
echo "=== Job ${JOB} Efficiency ==="
sacct -j ${JOB} --format=JobID%12,Elapsed%12,MaxRSS%12,ReqMem%12,NCPUS%6,AllocGRES%15,State%10 --noheader
echo ""
echo "Rules of thumb:"
echo "  MaxRSS < 50% of ReqMem? --> Request less memory next time"
echo "  Elapsed < 50% of TimeLimit? --> Request less time (better backfill)"
echo "  GPU job with GPU util < 30%? --> Consider CPU instead"
```

---

## 22. Non-Obvious Tricks and Unique Connections

### 22.1 SLURM Email Notifications

Get notified when jobs finish without polling `squeue`:
```bash
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80    # notify on end, fail, or 80% walltime
#SBATCH --mail-user=atsubhas@iu.edu
```
`TIME_LIMIT_80` is gold -- gives you a heads-up to check if the job needs more time before it dies.

### 22.2 scrontab: SLURM's Cron (Recurring Jobs)

Run periodic jobs without external cron. Useful for monitoring, data syncing, or periodic checkpoints:
```bash
# Edit your SLURM crontab
scrontab -e

# Example: check disk usage every 6 hours
0 */6 * * * -A r00602 -p general --mem=1G -t 00:05:00 ~/bin/check_quotas.sh

# Example: sync results to SDA tape archive every night
0 2 * * * -A r00602 -p general --mem=4G -t 01:00:00 ~/bin/archive_results.sh
```

### 22.3 SSH Config for Fast Connections

Add to `~/.ssh/config` on your Mac:
```
Host br200
    HostName bigred200.uits.iu.edu
    User atsubhas
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 4h
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ForwardAgent yes

Host quartz
    HostName quartz.uits.iu.edu
    User atsubhas
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 4h
```
```bash
mkdir -p ~/.ssh/sockets
```
`ControlMaster` multiplexes connections -- first SSH takes 2s, subsequent ones are instant. `ControlPersist 4h` keeps the tunnel alive for 4 hours after you disconnect.

### 22.4 Globus for Large Data Transfers

`scp` maxes out at ~100 MB/s. Globus uses parallel streams and can hit 1+ GB/s:
```bash
# IU endpoints (pre-configured):
# - "Indiana University BigRed200" for /N/slate/, /N/scratch/
# - "Indiana University Quartz" for Quartz storage
# - Your laptop: install Globus Connect Personal

# Transfer 500 GB dataset: use web UI at app.globus.org
# or CLI:
pip install globus-cli
globus login
globus transfer SOURCE_ENDPOINT:/path/ DEST_ENDPOINT:/path/ --recursive
```
Use Globus when moving >10 GB between systems or to/from external collaborators.

### 22.5 GPU Nodes for Memory-Heavy CPU Work (Billing Hack)

GPU nodes have **512 GB RAM** vs CPU nodes' 256 GB. If you need >250 GB for CPU work:
```bash
# Option A: Quartz (512 GB, no GPU billing) -- preferred
#SBATCH -p general --mem=480G

# Option B: BR200 GPU node (512 GB, but 16 billing units per GPU even if unused)
#SBATCH -p gpu --gres=gpu:0 --mem=480G    # request 0 GPUs but still get 512 GB node
# WARNING: --gres=gpu:0 may not work on all SLURM configs. Test first.
# Alternative: request 1 GPU + use only CPU. Costs 16 extra billing units.
```

### 22.6 tmux on Login Node (Survive Disconnects)

```bash
# Start named session on BR200 login node
ssh br200
tmux new -s work

# Inside tmux: submit jobs, run squeue, edit code
# Detach: Ctrl+B, then D
# Reconnect later:
ssh br200
tmux attach -t work

# Your session survives WiFi drops, laptop sleep, everything.
```

### 22.7 Cross-Cluster Data: BR200 and Quartz Share Storage

Home (`/N/u/atsubhas/`) and Slate (`/N/slate/atsubhas/`) are the **same filesystem** on both BR200 and Quartz. This means:
- Preprocess on Quartz (512 GB RAM), train on BR200 (A100 GPUs) -- **no data copy needed**
- Conda envs on Slate work on both clusters
- Scratch (`/N/scratch/`) may differ -- verify before assuming

### 22.8 Hybrid Quartz+BR200 Pipeline

The shared storage enables true cross-cluster pipelines:
```bash
# Stage 1: Preprocess on Quartz (needs 400 GB RAM)
ssh quartz
sbatch --parsable -p general --mem=480G preprocess.sh > /N/slate/atsubhas/project/.job1_id

# Stage 2: Train on BR200 (needs A100 GPUs) -- poll for Quartz job completion
ssh br200
cat > wait_and_train.sh << 'EOF'
#!/bin/bash
QUARTZ_JOB=$(cat /N/slate/atsubhas/project/.job1_id)
# Check if output exists (Quartz job wrote it)
while [ ! -f /N/slate/atsubhas/project/data/preprocessed_done.flag ]; do
    sleep 60
done
sbatch train_gpu.sh
EOF
bash wait_and_train.sh
```

### 22.9 Remote Development from Claude Code

You can have Claude write and submit jobs directly via SSH:
```bash
# From your Mac, Claude can:
ssh br200 "sbatch /N/slate/atsubhas/project/train.sh"
ssh br200 "squeue -u atsubhas"
ssh br200 "cat /N/slate/atsubhas/project/logs/train-12345.out | tail -50"

# Or edit files remotely:
ssh br200 "cat > /N/slate/atsubhas/project/train.sh << 'EOF'
#!/bin/bash
#SBATCH -J train ...
...
EOF"
```
The SSH alias `br200` + ControlMaster makes this fast. Claude already used this in this session to verify cluster data.

### 22.10 Job Efficiency Dashboard

Bookmark this on your browser -- shows your historical usage and efficiency:
```
https://one.iu.edu/task/iu/hpc-user-dashboard
```

### 22.11 DOT/MCX-Specific: Photon Budget vs GPU Count

For Monte Carlo photon transport (MCX), the key insight is **photons are embarrassingly parallel**:
- 1 A100 runs ~1e9 photons/sec
- 4 A100s on one node = ~4e9 photons/sec (linear scaling, NVLink not needed)
- For large meshes that need >40 GB GPU RAM: split mesh across GPUs or use MIG

But preprocessing (mesh generation, source placement) is **CPU-bound and memory-hungry**:
- Use `general` partition with 64+ cores for iso2mesh/MATLAB meshing
- For neonatal head models with fine resolution: may need >250 GB RAM -> use Quartz

**Optimal DOT pipeline resource mapping:**

| Stage | Partition | Resources | Why |
|-------|-----------|-----------|-----|
| Mesh generation (iso2mesh) | Quartz `general` | 128 cores, 480 GB | Memory-hungry, CPU-bound |
| Source/detector placement | BR200 `general` | 16 cores, 32 GB | Light compute |
| MCX forward simulation | BR200 `gpu` | 4x A100, 16 cores | GPU-native, linear scaling |
| Jacobian computation | BR200 `gpu` | 1-4 A100 | Per-source, parallelizable |
| Image reconstruction | BR200 `general` | 64 cores, 200 GB | Large matrix solve (BLAS-heavy) |
| Statistical analysis | BR200 `general` | 16 cores, 64 GB | Light compute |

### 22.12 The "Free Lunch" Optimization Checklist

Before any HPC run, check these -- they cost zero extra effort but can double performance:

- [ ] `--hint=nomultithread` in every SBATCH (5-20% for scientific compute)
- [ ] `OMP_NUM_THREADS` matches `--cpus-per-task` (prevents 2-5x oversubscription)
- [ ] `--time` is tight (1.5x measured, not max walltime -- better backfill)
- [ ] `--mem` is explicit (not `--mem=0` -- avoids locking unused RAM)
- [ ] `float32` instead of `float64` where precision allows (2x memory BW)
- [ ] `MKL_DEBUG_CPU_TYPE=5` if using MKL on AMD (prevents Intel throttling)
- [ ] Data on Slate, not Home (Lustre parallel I/O vs NFS)
- [ ] Conda envs on Slate (prevents Home quota death)
- [ ] `module purge` at script start (prevents stale module conflicts)

---

## Known Projects (atsubhas)

| Project | Conda Env | Slate Path | Description |
|---------|-----------|------------|-------------|
| `fp8_lm_repro` | `fp8lm` | `/N/slate/atsubhas/fp8_lm_repro/` | FP8 LM pretraining with Megatron + MS-AMP |

---

## Sources

- [About Big Red 200 (KB)](https://kb.iu.edu/d/brcc)
- [Big Red 200 TOP500 CPU](https://www.top500.org/system/179949/) / [GPU](https://top500.org/system/180020/)
- [About Quartz (KB)](https://servicenow.iu.edu/kb?id=kb_article_view&sysparm_article=KB0023985)
- [Run GPU Jobs (KB)](https://kb.iu.edu/d/avjk)
- [SLURM at IU (KB)](https://kb.iu.edu/d/awrz)
- [Compile on BR200 (KB)](https://kb.iu.edu/d/bhow) / [Quartz (KB)](https://kb.iu.edu/d/bgyd)
- [Apptainer (KB)](https://kb.iu.edu/d/aofz)
- [Slate Storage (KB)](https://kb.iu.edu/d/aqnk)
- [Slate-Scratch (KB)](https://kb.iu.edu/d/bgtr)
- [Home Quotas (KB)](https://kb.iu.edu/d/bhrl)
- [RT Projects](https://projects.rt.iu.edu/) / [About (KB)](https://kb.iu.edu/d/bihc)
- [IU HPC Services](https://uits.iu.edu/services/technology-for-research/high-performance-computing-and-storage/)
- [ACCESS Allocations](https://allocations.access-ci.org/)
- [Jetstream2 Docs](https://docs.jetstream-cloud.org/)
- [SLURM Multifactor Priority](https://slurm.schedmd.com/priority_multifactor.html)
- [SLURM Fair Tree](https://slurm.schedmd.com/fair_tree.html)
- [Slingshot Interconnect Analysis](https://www.glennklockwood.com/garden/Slingshot)
- [NVIDIA Blog: IU BR200](https://developer.nvidia.com/blog/indiana-university-unveils-new-gpu-accelerated-supercomputer-big-red-200/)
- Verified hardware data from on-cluster benchmarks (2026-03-10)
