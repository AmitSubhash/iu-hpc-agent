# Storage and Environment Management

## Storage Tiers

| Storage | Path | Quota | Purge | Backup | Best For |
|---------|------|-------|-------|--------|----------|
| **Home** | `/N/u/$USER/BigRed200/` | 100 GB, 800K files | No | No | Scripts, configs |
| **Slate** | `/N/slate/$USER/` | 800 GB (1.6 TB on request) | No | No | Data, checkpoints, envs |
| **Slate-Project** | `/N/slate/project/<name>/` | Free up to 15 TB | No | No | Shared group data |
| **Scratch** | `/N/scratch/$USER/` | 100 TB, 10M inodes | **30 days** | No | Large temp outputs |
| **SDA** | HPSS tape | ~42 PB | No | **Yes** (2 sites) | Long-term archival |

### Critical Rules

1. Home is shared across BR200 and Quartz (same NFS mount)
2. **Slate is NOT backed up**
3. **Scratch purges files >30 days without notification** (emergency purge at >80%)
4. SDA is the only storage with backup (2 tape copies, IUB + IUPUI). Access via SFTP, SCP, HSI, or [Globus](https://globus.iu.edu/) (endpoint: `IURT-Scholarly Data Archive`)
5. Home fills fast from caches -- symlink to Slate:

```bash
mkdir -p /N/slate/$USER/.cache/{pip,huggingface,conda}
ln -sf /N/slate/$USER/.cache/pip ~/.cache/pip
ln -sf /N/slate/$USER/.cache/huggingface ~/.cache/huggingface
ln -sf /N/slate/$USER/.cache/conda ~/.conda/pkgs
```

### Additional Storage Systems

| Storage | Type | Cost | Best For |
|---------|------|------|----------|
| **Geode-Project** | Persistent disk, replicated | $0.20/GB/yr | Long-term project data beyond Slate |
| **Research Database Complex** | MySQL, PostgreSQL, MongoDB, Oracle | Free (most cases) | Database-driven research |
| **RADaRS** | Secure enclave (Windows) | Free | Restricted/sensitive data analysis |

Request Geode-Project via RT Projects. RDC databases via UITS Enterprise DBA team.

### Quotas

```bash
quota -s                              # Home
lfs quota -hu $USER /N/slate/         # Slate
lfs quota -hu $USER /N/scratch/       # Scratch
```

### Project Directory Convention

```bash
export PROJ="${PROJ:-/N/slate/$USER/<project_name>}"
export LOGS=$PROJ/logs  REPOS=$PROJ/repos  DATA=$PROJ/data  CKPT=$PROJ/checkpoints
export ENVS="${ENVS:-/N/slate/$USER/envs}"
mkdir -p "$LOGS" "$REPOS" "$DATA" "$CKPT" "$ENVS"
```

---

## I/O Optimization

### Local tmpfs (RAM-Backed, NOT SSD)

`/tmp` and `/dev/shm` on compute nodes are 126 GB shared tmpfs. Budget ~100 GB usable.

```bash
LOCAL=/tmp/${SLURM_JOB_ID}/data
mkdir -p ${LOCAL}
trap "rm -rf /tmp/${SLURM_JOB_ID}" EXIT
cp -r /N/slate/$USER/project/data/ ${LOCAL}/
python train.py --data-dir ${LOCAL}
```

### Lustre Stripe Tuning

```bash
lfs setstripe -c 4 /N/slate/$USER/project/checkpoints/   # multi-GB files
```

### Small File Avoidance

Lustre excels at large sequential I/O, not metadata-heavy ops. For many small files, use HDF5, WebDataset (tar streaming for PyTorch), or LMDB.

### DataLoader

```python
DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True,
           prefetch_factor=2, persistent_workers=True)
```

---

## Conda on Slate

```bash
conda create --prefix /N/slate/$USER/envs/myenv python=3.11 -y
conda activate /N/slate/$USER/envs/myenv
conda env export --prefix /N/slate/$USER/envs/myenv --from-history > environment.yml
```

- Always `--prefix` for Slate (not default `~/.conda/envs/`)
- Use `mamba` or libmamba solver (10-50x faster)
- Pin CUDA: `pytorch=*=*cu126*`
- Install from login node, not compute (don't waste allocation)

---

## Module + Conda Stacking

```bash
module purge
module load PrgEnv-gnu    2>/dev/null || true
module load cudatoolkit/12.6 2>/dev/null || module load cudatoolkit/11.8 2>/dev/null
module load cudnn/9.10.1.4_cuda12 2>/dev/null || true
module load nccl/2.27.7-1 2>/dev/null || true
conda activate /N/slate/$USER/envs/<env_name>
```

---

## Containers (Apptainer 1.4.1)

```bash
apptainer pull /N/slate/$USER/containers/pytorch.sif docker://nvcr.io/nvidia/pytorch:24.01-py3

apptainer exec --nv --bind /N/slate/$USER/project:/project \
    /N/slate/$USER/containers/pytorch.sif python /project/train.py
```

Use for: reproducibility across clusters, complex deps, NGC optimized builds.

---

## Software Modules (BR200, Verified)

| Module | Versions |
|--------|----------|
| `cudatoolkit` | 11.4, 11.7, 11.8, 12.2, **12.6** |
| `cudnn` | 8.9.2, 9.5.1, **9.10.1.4_cuda12** |
| `nccl` | 2.27.7-1 |
| `python/gpu` | 3.10.10, 3.11.5, 3.12.5 |
| `python` (CPU) | 3.11.13, 3.12.11, 3.13.5 |
| `matlab` | 2025a |
| `apptainer` | 1.4.1 |
| `cray-mpich` | 8.1.23 (optimized for Slingshot) |
| `openblas` | 0.3.26 |
| CUDA path | `/N/soft/sles15sp6/cuda/gnu/12.6/` |

Compilers: `PrgEnv-gnu` (GCC, default), `PrgEnv-cray`, `PrgEnv-intel`, `PrgEnv-nvidia`.

MPI: `cray-mpich` (default, Slingshot-optimized). `openmpi` available but slower.
