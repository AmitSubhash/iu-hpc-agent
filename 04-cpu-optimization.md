# CPU Optimization for EPYC 7742 (Zen 2)

## NUMA Topology (Verified on Compute Node)

**Compute nodes are DUAL-SOCKET with 8 NUMA nodes** (login is single-socket, 4 NUMA):

```
Socket 0:                          Socket 1:
  Node 0: CPUs 0-15, 128-143        Node 4: CPUs 64-79, 192-207
  Node 1: CPUs 16-31, 144-159       Node 5: CPUs 80-95, 208-223
  Node 2: CPUs 32-47, 160-175       Node 6: CPUs 96-111, 224-239
  Node 3: CPUs 48-63, 176-191       Node 7: CPUs 112-127, 240-255

Each NUMA node: 16 physical cores, ~32 GB RAM
Physical cores: 0-127. SMT siblings: 128-255.
Total: 128 physical (256 with SMT), 8 x 32 GB = 256 GB
```

### Cache Hierarchy

| Level | Size | Latency |
|-------|------|---------|
| L1d/L1i | 32 KB/core | ~1.5 ns |
| L2 | 512 KB/core | ~5 ns |
| L3 | 16 MB/CCX (4 cores share) | ~17 ns same-CCX, ~35-40 ns cross-CCX |
| DRAM | ~32 GB/NUMA | ~120-140 ns local, ~190-220 ns remote |

**Key:** L3 is per-CCX (16 MB for 4 cores), NOT shared across the socket. Cross-CCX is 2x slower. NUMA distance: 10 (local), 12 (remote).

---

## SMT (Hyperthreading)

**Rule: Use `--hint=nomultithread` for scientific Python/BLAS workloads.**

| Workload | SMT Effect | Use SMT? |
|----------|------------|----------|
| Dense FP (GEMM, FFT, NumPy) | -10 to -20% | **OFF** |
| AVX2-heavy (SciPy, convolutions) | -10 to -20% | **OFF** |
| Memory-BW-bound (stencils) | ~0% or negative | **OFF** |
| Monte Carlo (large state/thread) | -5 to -10% | **OFF** |
| I/O-bound (file processing) | +20-40% | ON |
| Mixed integer + many light tasks | +10-25% | ON |

---

## NUMA Binding (20-50% Improvement)

```bash
# 8 workers, one per NUMA (16 cores x 32 GB each):
#SBATCH --ntasks=8 --cpus-per-task=16 --hint=nomultithread
srun --cpu-bind=cores python worker.py

# 2 workers, one per socket (64 cores x 128 GB each):
#SBATCH --ntasks=2 --cpus-per-task=64 --hint=nomultithread
srun --cpu-bind=sockets python worker.py

# Single process, pin to NUMA 0 (latency-sensitive):
numactl --cpunodebind=0 --membind=0 python script.py

# Interleave memory for bandwidth-bound (streaming):
numactl --interleave=all python script.py
```

---

## Thread Control (Prevent Oversubscription)

**4 Python processes x 16 BLAS threads = 64 threads on 16 cores = 2-5x slowdown.**

```bash
# Embarrassingly parallel (many processes, each single-threaded):
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Single-process BLAS-heavy (one big matrix multiply):
export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Hybrid (4 processes x 16 threads = 64 cores):
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
```

### Standard Env Vars for CPU Jobs

```bash
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MKL_DEBUG_CPU_TYPE=5
export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS
```

---

## MKL on AMD Fix

Intel MKL throttles on AMD. Either:
```bash
export MKL_DEBUG_CPU_TYPE=5    # Force Skylake-X codepath
```
Or use OpenBLAS:
```bash
conda install numpy "libblas=*=*openblas"
```

---

## Compiler Flags for EPYC 7742

```bash
CFLAGS="-O3 -march=znver2 -mtune=znver2 -mavx2 -mfma -ftree-vectorize"
# Aggressive: add -flto -ffast-math
# WARNING: NO AVX-512 on Zen 2. Do NOT use -mavx512f.
```

---

## Multiprocessing Decision Matrix

| Method | Best For | NUMA-Aware |
|--------|----------|-----------|
| `multiprocessing.Pool` | CPU-bound, single-node | Manual (numactl) |
| `joblib` (loky) | scikit-learn, embarrassingly parallel | Partial |
| GNU `parallel` | File-level CLI tasks | Yes (numactl/task) |
| `mpi4py` | Multi-node, tightly coupled | Yes (rank binding) |
| `concurrent.futures` (threads) | I/O-bound (GIL released) | No |

---

## Profiling

```bash
# IPC -- below 1.0 means memory-bound
perf stat -e instructions,cycles,cache-misses python script.py

# NUMA balance (remote access should be ~0%)
numastat -p $(pgrep python)

# Thread placement
ps -eo pid,tid,psr,comm | grep python
```
