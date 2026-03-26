# Hardware Architecture

## Big Red 200

| Attribute | Value |
|-----------|-------|
| Vendor | HPE Cray EX (Shasta), first production Shasta worldwide |
| OS | SUSE Linux Enterprise Server 15 SP6 |
| SLURM | 24.05.4, cluster name: `br200` |
| Total | 706 nodes (640 CPU + 66 GPU), ~7 PFLOPS |
| Cost | ~$9.6M, warm water direct liquid cooling (ASHRAE W3) |

### CPU Nodes (640)

| Spec | Value |
|------|-------|
| Blade | HPE Cray EX425 |
| CPUs | 2x AMD EPYC 7742 (Rome), 64c/128t per socket, 2.25 GHz |
| ISA | AVX2, FMA, SSE4.2, AES-NI (**NO AVX-512**) |
| Memory | 256 GB DDR4-3200 (250 GB usable, 10 GB MemSpecLimit) |
| Memory BW | ~170-185 GB/s per socket (STREAM Triad) |
| NUMA | 8 nodes per compute node (dual-socket), 4 per socket |
| Local tmpfs | `/tmp` and `/dev/shm`: 126 GB shared pool (RAM-backed) |
| Swap | 2 GB (unusable for HPC) |

See `04-cpu-optimization.md` for NUMA topology, cache hierarchy, and binding.

### GPU Nodes (66)

| Spec | Value |
|------|-------|
| Blade | HPE Cray EX235n |
| CPU | 1x AMD EPYC 7713 (Milan), 64c/128t, 2.0 GHz, single socket |
| GPUs | 4x NVIDIA A100-SXM4-40GB (264 total) |
| Host memory | 512 GB DDR4 (500 GB usable) |
| Default mem/GPU | 125,440 MB (~122.5 GB) if not specified |
| GPU memory | 40 GB HBM2e, ~1,350 GB/s measured BW |
| GPU compute | 19.5 TF FP64, 312 TF TF32, 624 TF FP16/BF16 per GPU |
| FP8 | Via MS-AMP (O1/O2/O3) |
| MIG | Supported (up to 7 instances per A100) |

### GPU Topology (Verified 2026-03-10)

| Spec | Value |
|------|-------|
| Interconnect | NVLink 3.0, **NV4 full mesh** (every GPU pair directly connected) |
| Links per GPU | 12 total (4 to each of 3 peers) |
| Per-link speed | 25 GB/s per direction |
| Measured P2P BW | 93.5 GB/s uni / 185 GB/s bidi per pair (93% theoretical) |
| Measured latency | ~2.2 us (vs 12-53 us without NVLink) |

BR200 uses direct NVLink mesh, NOT NVSwitch (unlike 8-GPU DGX A100).

### System Totals

| Metric | Value |
|--------|-------|
| Total CPU cores | 86,144 (81,920 CPU-node + 4,224 GPU-node) |
| Total GPUs | 264x A100-SXM4-40GB |
| Total host memory | ~197 TB + 10.6 TB GPU HBM2e |

---

## Quartz

| Spec | CPU Nodes (92) | GPU V100 (22) | GPU H100 (12) |
|------|---------------|---------------|---------------|
| CPUs | 2x EPYC 7742, 128c | varies | varies |
| Memory | **512 GB** | **768 GB** | TBD (verify) |
| GPUs | -- | 4x V100-32GB | 4x H100-SXM5-80GB |
| GPU memory | -- | 32 GB HBM2 | **80 GB HBM3**, ~3.35 TB/s BW |
| GPU features | -- | Tensor Cores V1 | FP8, Transformer Engine, NVLink 4.0 |
| Total GPUs | -- | 88 | 48 |

**Key differences from BR200:** 2x RAM on CPU nodes (512 vs 256), H100s have 2x the GPU memory of BR200 A100s (80 vs 40 GB), slower interconnect (Ethernet/IB), fewer nodes but better for memory-hungry CPU work and large-model training (H100).

---

## Network

### Slingshot-10 (BR200)

| Spec | Value |
|------|-------|
| Topology | Dragonfly (max 3 switch hops) |
| Switch | Rosetta ASIC, 64 x 200 Gb/s ports, 25.6 Tb/s bisection |
| NIC | Mellanox ConnectX-5 100G (2 per GPU node: mlx5_0, mlx5_1) |
| Latency | ~1.8 us unloaded, <8.7 us p99 |
| Intra-group | Copper (up to 2.6m, fully connected) |
| Inter-group | Optical (up to 100m) |

Quartz uses HDR InfiniBand / Ethernet (slower than Slingshot).

### Bandwidth Hierarchy (BR200)

```
GPU HBM2e local:       1,350 GB/s
NVLink (intra-node):   93.5 GB/s per GPU pair
Slingshot (inter-node): ~12.5 GB/s per NIC (100 Gbps)
Lustre I/O:            ~1-10 GB/s (shared, variable)
```

**Keep communication within a single node** whenever possible. Inter-node is ~7.5x slower than intra-node GPU-to-GPU.
