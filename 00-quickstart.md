# New User Quickstart

## 1. Get Access

1. Go to [projects.rt.iu.edu](https://projects.rt.iu.edu/)
2. **Students:** Search PI "lamhuber" to self-enroll in "HPC for Students" (no advisor needed)
3. **Researchers:** Create a project with your PI's username
4. Request allocations for BR200, Quartz, and Slate storage
5. Wait for approval (typically a few days)

## 2. First Connection

**Option A: SSH (recommended for experienced users)**
```bash
ssh YOUR_USERNAME@bigred200.uits.iu.edu   # Big Red 200
ssh YOUR_USERNAME@quartz.uits.iu.edu      # Quartz
# Both use IU credentials + Duo MFA
```

**Option B: Research Desktop (RED) -- GUI alternative, no command line needed**
Install the [ThinLinc client](https://www.cendio.com/thinlinc/download), connect to Quartz. Provides a graphical desktop with access to BR200, Quartz, Slate, and all storage. Great for users new to HPC or preferring visual interfaces.

**SSH config (recommended):** Add to `~/.ssh/config` on your local machine:
```
Host br200
    HostName bigred200.uits.iu.edu
    User YOUR_USERNAME
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 4h
    ServerAliveInterval 60

Host quartz
    HostName quartz.uits.iu.edu
    User YOUR_USERNAME
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 4h
```
Then: `mkdir -p ~/.ssh/sockets && ssh br200`

## 3. Set Up Your Environment

Add to `~/.bashrc` on BR200:
```bash
# === IU HPC Configuration ===
export SLURM_ACCOUNT="your_account_id"      # from projects.rt.iu.edu
export PROJ="/N/slate/$USER/your_project"
export ENVS="/N/slate/$USER/envs"

# Prevent home quota exhaustion (home = 100 GB only)
export PIP_CACHE_DIR="/N/slate/$USER/.cache/pip"
export HF_HOME="/N/slate/$USER/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$ENVS" 2>/dev/null
```

Then: `source ~/.bashrc`

## 4. Create Your First Conda Environment

```bash
# First time only: initialize conda
module load python/gpu/3.11.5
conda init bash
source ~/.bashrc

# Create env on Slate (NOT home)
module purge
conda create --prefix /N/slate/$USER/envs/myenv python=3.11 -y
conda activate /N/slate/$USER/envs/myenv
pip install torch numpy scipy   # your packages
```

## 5. Verify Your Setup

```bash
# Check account and allocations
sacctmgr show assoc where user=$USER format=Account,QOS

# Check storage
quota -s                              # Home (100 GB limit)
lfs quota -hu $USER /N/slate/         # Slate (800 GB default)

# Check partitions
sinfo -o "%20P %10a %10l %6D"

# Quick GPU test (starts within minutes)
salloc -A $SLURM_ACCOUNT -p gpu-debug --gres=gpu:1 --cpus-per-task=4 --mem=8G -t 00:15:00
# Once allocated:
module load cudatoolkit/12.6
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name()}')"
exit   # release allocation
```

## 6. Submit Your First Job

Create `test_job.sh`:
```bash
#!/bin/bash
#SBATCH -J test_job
#SBATCH -A <account>
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o test-%j.out -e test-%j.err

set -euo pipefail
module purge
module load cudatoolkit/12.6
conda activate /N/slate/$USER/envs/myenv

echo "Running on $(hostname)"
nvidia-smi
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}')"
echo "Success!"
```

```bash
sbatch test_job.sh            # submit
squeue -u $USER               # check status
cat test-*.out                 # read output
```

## 7. Common First-Day Mistakes

| Mistake | Fix |
|---------|-----|
| Forgetting `-A` flag | Always include `#SBATCH -A <account>` |
| Conda envs in home dir | Use `--prefix /N/slate/$USER/envs/name` |
| Requesting `--mem=0` | Specify explicit memory (e.g., `--mem=32G`) |
| Not loading modules | `module purge && module load ...` in every script |
| Running compute on login node | Use `salloc` (interactive) or `sbatch` (batch) |
| Storing data in home | Home is 100 GB NFS. Use Slate for everything large |
| Requesting max walltime | Shorter = faster scheduling. Use 1.5x estimated |

## Next Steps

- Estimate resources for your workload: `09-resource-estimator.md`
- Copy a template for your job type: `06-templates.md`
- Understand partitions and billing: `02-slurm.md`
- Job failed? `10-troubleshooting.md`
