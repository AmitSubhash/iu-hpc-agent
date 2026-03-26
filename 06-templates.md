# SBATCH Templates

Replace `<account>`, `<env_name>`, and `<project>` with your values.
`$USER` expands automatically at runtime.

## Smart Defaults (Copy and Adjust)

```bash
#!/bin/bash
#SBATCH -J my_job
#SBATCH -A <account>
#SBATCH -p general                    # general | gpu | gpu-debug | debug
#SBATCH -N 1
#SBATCH --cpus-per-task=64            # 1-128 CPU, 1-64 GPU
#SBATCH --mem=128G                    # CPU max 250G, GPU max 500G
#SBATCH -t 04:00:00                   # general max 4d, gpu max 2d
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@300
#SBATCH -o logs/%x-%j.out -e logs/%x-%j.err
## GPU: uncomment next line
## #SBATCH --gres=gpu:4

set -euo pipefail
mkdir -p logs

module purge
## GPU: uncomment next two lines
## module load cudatoolkit/12.6 2>/dev/null || true
## module load nccl/2.27.7-1 2>/dev/null || true

conda activate /N/slate/$USER/envs/<env_name>

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_DEBUG_CPU_TYPE=5
export OMP_PROC_BIND=close
export OMP_PLACES=cores

export PROJ="/N/slate/$USER/<project>"
mkdir -p "${PROJ}"/{logs,checkpoints,data}

# YOUR CODE HERE
python train.py
```

---

## GPU Training (Single Node, 4 GPUs)

```bash
#!/bin/bash
#SBATCH -J train_model
#SBATCH -A <account>
#SBATCH -p gpu -N 1
#SBATCH --gres=gpu:4 --cpus-per-task=32 --mem=400G
#SBATCH --hint=nomultithread
#SBATCH -t 12:00:00 --signal=B:USR1@300
#SBATCH -o logs/%x-%j.out -e logs/%x-%j.err

set -euo pipefail
module purge
module load PrgEnv-gnu cudatoolkit/12.6 nccl/2.27.7-1 2>/dev/null || true
conda activate /N/slate/$USER/envs/<env_name>

export PROJ="/N/slate/$USER/<project>"
export NCCL_DEBUG=WARN NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "${PROJ}/checkpoints" logs

nvidia-smi dmon -s pucvmet -d 30 -o DT > logs/gpu_${SLURM_JOB_ID}.csv &
torchrun --nproc_per_node=4 train.py \
    --data-dir "${PROJ}/data" --checkpoint-dir "${PROJ}/checkpoints" --auto-resume
```

---

## CPU Preprocessing

```bash
#!/bin/bash
#SBATCH -J preprocess
#SBATCH -A <account>
#SBATCH -p general -N 1
#SBATCH --cpus-per-task=64 --mem=128G --hint=nomultithread
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x-%j.out -e logs/%x-%j.err

set -euo pipefail
module purge
conda activate /N/slate/$USER/envs/<env_name>

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_DEBUG_CPU_TYPE=5
python preprocess.py --workers $SLURM_CPUS_PER_TASK --input /N/slate/$USER/data/raw/
```

---

## Job Array (Hyperparameter Sweep)

```bash
#!/bin/bash
#SBATCH -J hpsweep
#SBATCH -A <account>
#SBATCH -p gpu
#SBATCH --array=0-49%8
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/hpsweep-%A_%a.out -e logs/hpsweep-%A_%a.err

set -euo pipefail
module purge
module load cudatoolkit/12.6 2>/dev/null || true
conda activate /N/slate/$USER/envs/<env_name>

PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" configs/hparams.csv)
LR=$(echo "$PARAMS" | cut -d',' -f1)
BS=$(echo "$PARAMS" | cut -d',' -f2)
WD=$(echo "$PARAMS" | cut -d',' -f3)

python train.py --lr "$LR" --batch-size "$BS" --weight-decay "$WD" \
    --run-id "sweep_${SLURM_ARRAY_TASK_ID}"
```

---

## Job Array (File Processing)

```bash
#!/bin/bash
#SBATCH -J fileproc
#SBATCH -A <account>
#SBATCH -p general
#SBATCH --array=0-499%50              # MaxArraySize=500
#SBATCH --cpus-per-task=4 --mem=8G --hint=nomultithread
#SBATCH -t 00:30:00
#SBATCH -o logs/batch-%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

FILES=($(ls /N/slate/$USER/data/raw/*.nii.gz))
python preprocess.py --input "${FILES[$SLURM_ARRAY_TASK_ID]}" \
    --output /N/slate/$USER/data/processed/
```

---

## Multi-Node Distributed (4 Nodes, 16 GPUs)

```bash
#!/bin/bash
#SBATCH -J distributed
#SBATCH -A <account>
#SBATCH -p gpu -N 4
#SBATCH --gres=gpu:4 --cpus-per-task=16 --mem=200G
#SBATCH -t 1-00:00:00 --signal=B:USR1@300
#SBATCH -o logs/%x-%j.out -e logs/%x-%j.err

set -euo pipefail
module purge
module load PrgEnv-gnu cudatoolkit/12.6 nccl/2.27.7-1 2>/dev/null || true
conda activate /N/slate/$USER/envs/<env_name>

export NCCL_DEBUG=INFO NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=hsn
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29500}"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    bash -lc "torchrun \
        --nnodes=$SLURM_NNODES --nproc_per_node=4 \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        train.py --auto-resume --checkpoint-dir /N/slate/$USER/<project>/checkpoints/"
```

---

## Interactive GPU Session

```bash
# Quick debug (1h, starts fast)
salloc -A $SLURM_ACCOUNT -p gpu-debug --gres=gpu:1 --cpus-per-task=8 --mem=16G -t 00:30:00

# Extended (4h)
salloc -A $SLURM_ACCOUNT -p gpu-interactive --gres=gpu:1 --cpus-per-task=16 --mem=32G -t 04:00:00

# Then:
module purge && module load cudatoolkit/12.6
conda activate /N/slate/$USER/envs/<env_name>
python -c "import torch; print(torch.cuda.get_device_name())"
```

---

## Jupyter on BR200

```bash
# On BR200 (after salloc on gpu-interactive):
jupyter lab --no-browser --port=8888 --ip=0.0.0.0

# From local machine:
ssh -L 8888:COMPUTE_NODE:8888 br200
# Open http://localhost:8888
```

---

## Pipeline Submission

```bash
#!/bin/bash
set -euo pipefail
ACCOUNT="${SLURM_ACCOUNT:?Set SLURM_ACCOUNT in .bashrc}"
PROJECT="myproject"
LOGDIR="/N/slate/$USER/${PROJECT}/logs"
mkdir -p "$LOGDIR"

JOB1=$(sbatch --parsable \
    -J "${PROJECT}_preproc" -A "$ACCOUNT" -p general \
    --cpus-per-task=64 --mem=200G -t 12:00:00 \
    -o "${LOGDIR}/preproc-%j.out" preprocess.sh)
echo "Preprocess: $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 \
    -J "${PROJECT}_train" -A "$ACCOUNT" -p gpu \
    --gres=gpu:4 --cpus-per-task=32 --mem=200G -t 2-00:00:00 \
    --signal=B:USR1@300 -o "${LOGDIR}/train-%j.out" train.sh)
echo "Training: $JOB2"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 \
    -J "${PROJECT}_eval" -A "$ACCOUNT" -p gpu \
    --gres=gpu:1 --cpus-per-task=8 --mem=32G -t 02:00:00 \
    -o "${LOGDIR}/eval-%j.out" eval.sh)
echo "Evaluation: $JOB3"

echo "Pipeline: $JOB1 -> $JOB2 -> $JOB3"
```

---

## Memory-Intensive CPU (Quartz)

```bash
#!/bin/bash
#SBATCH -J big_memory
#SBATCH -A <account>
#SBATCH -p general -N 1
#SBATCH --cpus-per-task=128 --mem=480G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/%x-%j.out -e logs/%x-%j.err

set -euo pipefail
module purge
conda activate /N/slate/$USER/envs/<env_name>
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_DEBUG_CPU_TYPE=5
python memory_intensive_analysis.py
```
