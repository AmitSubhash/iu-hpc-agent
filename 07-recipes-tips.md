# Workflow Recipes and Tips

## Decision Flowchart: What Should I Submit?

```
Training neural net?
  Fits 1 GPU (<35 GB)?       -> -p gpu --gres=gpu:1 -t 12:00:00
  Multi-GPU needed?           -> -p gpu --gres=gpu:4
  Multi-node?                 -> -p gpu -N 4 --gres=gpu:4

Processing files?
  <100 files                  -> -p general --cpus-per-task=64
  100+ files                  -> -p general --array=0-N%50 --cpus-per-task=1

Hyperparameter sweep?         -> --array=0-49%8 --gres=gpu:1

Compiling / building envs?    -> -p general --cpus-per-task=32 -t 02:00:00

Quick test?
  GPU: salloc -p gpu-debug --gres=gpu:1 -t 00:30:00
  CPU: salloc -p debug -t 00:30:00

Memory >250 GB?               -> Quartz general (512 GB) or BR200 GPU (512 GB)
MATLAB?                       -> module load matlab/2025a; -p general --cpus-per-task=16
```

---

## ML Training with Auto-Retry

```bash
#!/bin/bash
#SBATCH -J ml_train
#SBATCH -A <account>
#SBATCH -p gpu
#SBATCH --gres=gpu:4 --cpus-per-task=32 --mem=400G
#SBATCH --hint=nomultithread -t 2-00:00:00
#SBATCH --signal=B:USR1@300 --requeue
#SBATCH -o logs/%x-%j.out -e logs/%x-%j.err

set -euo pipefail
module purge
module load cudatoolkit/12.6 nccl/2.27.7-1 2>/dev/null || true
conda activate /N/slate/$USER/envs/<env_name>

export NCCL_DEBUG=WARN NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8    # 32 cpus / 4 gpus

nvidia-smi dmon -s pucvmet -d 30 -o DT > logs/gpu_${SLURM_JOB_ID}.csv &

# Stage small datasets to local tmpfs
DATA_SRC="/N/slate/$USER/data/train"
if [ $(du -sm "$DATA_SRC" | cut -f1) -lt 80000 ]; then
    LOCAL=/tmp/${SLURM_JOB_ID}/data && mkdir -p ${LOCAL}
    trap "rm -rf /tmp/${SLURM_JOB_ID}" EXIT
    cp -r "$DATA_SRC" ${LOCAL}/
    DATA_DIR=${LOCAL}/train
else
    DATA_DIR="$DATA_SRC"
fi

torchrun --nproc_per_node=4 train.py \
    --data-dir ${DATA_DIR} --checkpoint-dir /N/slate/$USER/checkpoints/ \
    --auto-resume --workers-per-gpu 8
```

---

## Mass Parallel Processing (10K+ Tasks)

```bash
#!/bin/bash
#SBATCH -J mass_process
#SBATCH -A <account>
#SBATCH -p general
#SBATCH --array=0-499%100             # MaxArraySize=500
#SBATCH --cpus-per-task=4 --mem=16G --hint=nomultithread
#SBATCH -t 01:00:00
#SBATCH -o logs/batch-%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

TOTAL_FILES=10000
CHUNK=$((TOTAL_FILES / SLURM_ARRAY_TASK_COUNT))
START=$((SLURM_ARRAY_TASK_ID * CHUNK))
python process_chunk.py --start ${START} --count ${CHUNK}
```

---

## Post-Job Efficiency Check

```bash
sacct -j JOBID --format=JobID,Elapsed,MaxRSS,ReqMem,AllocCPUS,AllocGRES,State
# MaxRSS < 50% ReqMem? -> request less memory
# Elapsed < 50% TimeLimit? -> request less time (better backfill)
# GPU util < 30%? -> consider CPU
```

---

## Common Pitfalls

See `10-troubleshooting.md` for detailed diagnostic flowcharts. Quick reference:

| Pitfall | Fix |
|---------|-----|
| Forgetting `-A <account>` | Always in SBATCH header |
| Job stuck PENDING | `sshare -u $USER`; reduce resources; shorter `--time` |
| Host OOM (signal 9) | `sacct --format=MaxRSS`; increase `--mem` |
| CUDA OOM | Reduce batch, BF16, activation checkpointing, FSDP |
| Training diverges after resume | Save/restore RNG + LR scheduler + optimizer state |
| GPU util <30% | More `num_workers`, `pin_memory=True`, bigger batch |
| Home quota full | Symlink caches to Slate (see `05-storage-envs.md`) |
| Scratch data vanished | 30-day purge. Use Slate for anything persistent |

---

## Tips and Tricks

### Email Notifications

```bash
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=your_username@iu.edu
```
`TIME_LIMIT_80` warns at 80% walltime -- check if you need more time.

### scrontab (SLURM Cron)

```bash
scrontab -e
0 */6 * * * -A <account> -p general --mem=1G -t 00:05:00 ~/bin/check_quotas.sh
0 2 * * * -A <account> -p general --mem=4G -t 01:00:00 ~/bin/archive_results.sh
```

### SSH Config

See `00-quickstart.md` for SSH config setup. `ControlMaster` multiplexes connections -- first SSH 2s, subsequent instant.

### Data Transfer (Local Machine to Cluster)

**scp (small transfers, <10 GB):**
```bash
scp -r ./my_data/ br200:/N/slate/$USER/project/data/
```

**Globus (large transfers, >1 GB, preferred):**
1. Install [Globus Connect Personal](https://www.globus.org/globus-connect-personal) on your laptop
2. Open [globus.iu.edu](https://globus.iu.edu/), authenticate with IU credentials
3. Left panel: your laptop endpoint. Right panel: `IURT-Slate`
4. Navigate to your Slate directory, start transfer

| IU Endpoint | Accesses |
|-------------|----------|
| `IURT-Slate` | Slate and Slate-Project |
| `IURT-Scholarly Data Archive` | SDA (tape) |

Also connects to Google at IU Drive and Microsoft OneDrive at IU.

```bash
# CLI alternative
pip install globus-cli && globus login
globus transfer SOURCE:/path/ DEST:/path/ --recursive
```

### tmux on Login Node

```bash
ssh br200 && tmux new -s work   # survives WiFi drops, laptop sleep
# Detach: Ctrl+B, D. Reconnect: tmux attach -t work
```

### Cross-Cluster Pipelines

Home and Slate are shared between BR200 and Quartz -- no data copy needed. Preprocess on Quartz (512 GB RAM), train on BR200 (A100s), same Slate paths.

### GPU Nodes for Memory-Heavy CPU Work

GPU nodes have 512 GB vs CPU 256 GB. For >250 GB CPU work:
- **Quartz general** (512 GB, no GPU billing) -- preferred
- BR200 GPU node works but costs 16 billing units per unused GPU

### Remote Dev from Claude Code

```bash
ssh br200 "sbatch /N/slate/$USER/project/train.sh"
ssh br200 "squeue -u $USER"
ssh br200 "tail -50 /N/slate/$USER/project/logs/train-12345.out"
```

---

## Free Lunch Checklist

- [ ] `--hint=nomultithread` in every SBATCH (5-20% gain)
- [ ] `OMP_NUM_THREADS` = `--cpus-per-task` (prevents 2-5x oversubscription)
- [ ] `--time` tight at 1.5x measured (better backfill)
- [ ] `--mem` explicit, never `--mem=0`
- [ ] `float32` over `float64` where OK (2x memory BW)
- [ ] `MKL_DEBUG_CPU_TYPE=5` if using MKL on AMD
- [ ] Data on Slate, not Home (Lustre vs NFS)
- [ ] Conda envs on Slate (prevents Home quota death)
- [ ] `module purge` at script start
