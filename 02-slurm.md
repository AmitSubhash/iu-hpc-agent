# SLURM Configuration and Optimization

## Partitions

### Big Red 200

| Partition | Cores | RAM (usable) | GPUs | Max Time | Nodes |
|-----------|-------|-------------|------|----------|-------|
| `general` | 128 | 250 GB | -- | 4 days | 638 |
| `debug` | 128 | 250 GB | -- | 1 hour | 2 |
| `interactive` | 128 | 250 GB | -- | ~4 hours | subset |
| `gpu` | 64 | 500 GB | 4x A100 | 2 days | 62 |
| `gpu-debug` | 64 | 500 GB | 4x A100 | 1 hour | 2 |
| `gpu-interactive` | 64 | 500 GB | 4x A100 | 4 hours | 2 |

### Quartz

| Partition | Cores | RAM | GPUs | Max Time | Nodes |
|-----------|-------|-----|------|----------|-------|
| `general` | 128 | 512 GB | -- | 4 days | 92 |
| `gpu` (V100) | varies | 768 GB | 4x V100-32GB | 2 days | 22 |
| `gpu` (H100) | varies | TBD | 4x H100 | 2 days | 12 |

### Defaults

| Partition | DefaultTime | Default Memory |
|-----------|-------------|---------------|
| `general`, `debug` | 1 hour | 1,920 MB per CPU |
| `gpu`, `gpu-debug` | 1 hour | 125,440 MB per GPU (~122.5 GB) |
| `gpu-interactive` | 30 min | 125,440 MB per GPU |

**If you omit `--time`, you get 1 hour. If you omit `--mem` on GPU, each GPU gets 122.5 GB.**

---

## TRES Billing Weights (Verified 2026-03-19)

**CPU partitions:** `CPU=1.0, Mem=0.512G`
Full CPU node (128c, 250 GB) = **256 billing units/hr**

**GPU partitions:** `CPU=1.0, Mem=0.128G, GRES/gpu=16.0`
Full GPU node (64c, 500 GB, 4 GPUs) = **192 billing units/hr**

**Cost insights:**
- 1 GPU = 16x a CPU core in fair-share
- Memory is cheap on GPU nodes (0.128/GB vs 0.512/GB on CPU)
- Under-using GPUs wastes billing -- profile utilization first

---

## QOS (Verified 2026-03-19)

| QOS | Priority | MaxWall | MaxNodes/User | MaxSubmit |
|-----|----------|---------|---------------|-----------|
| `allocated` | 0 | 4 days | 200 | 1000 |
| `allocated-gpu` | 0 | 2 days | 36 | 1000 |
| `highprio` | 1 | 4 days | -- | 500 |
| `highprio-gpu` | 1 | 2 days | 12 | 1000 |
| `gpu-interactive` | 0 | 4 hours | 1 | 1 |
| `debug` | 0 | 1 hour | 2 | 4 |

**Check your QOS:** `sacctmgr show assoc where user=$USER format=Account,QOS`

---

## Scheduler (Verified 2026-03-19)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| PriorityWeight (Age/FairShare/QOS) | 100000 each | All three matter equally |
| PriorityDecayHalfLife | 4 days | Heavy usage 4 days ago is 50% forgotten |
| PriorityMaxAge | 12 hours | Waiting beyond 12h gives no more priority |
| bf_interval | 5 min | Backfill runs every 5 min |
| bf_max_job_user | 10 | Only 10 of YOUR jobs tested per backfill cycle |
| MaxArraySize | 500 | Array indices 0-499 max |

---

## Backfill Exploitation

**Accurate, short time limits are the #1 factor.**

| Strategy | Impact |
|----------|--------|
| Tight walltime (1.5x measured) | Highest |
| `--time-min` (give scheduler a range) | High |
| Fewer resources (1 GPU beats 4 for scheduling) | High |
| `gpu-debug` for tests (1h, nearly instant) | High |
| Explicit `--mem` (not `--mem=0`) | Medium |
| Decompose long jobs (8x 6h > 1x 48h) | Medium |
| Off-peak submission (weekends, evenings) | Medium |

---

## Job Arrays

```bash
#SBATCH --array=0-99          # 100 tasks
#SBATCH --array=0-99%10       # max 10 concurrent
#SBATCH --array=1,5,10,20     # specific indices
#SBATCH --array=0-100:5       # step: 0,5,10,...,100
```

Variables: `$SLURM_ARRAY_TASK_ID`, `$SLURM_ARRAY_JOB_ID`, `$SLURM_ARRAY_TASK_COUNT`.

---

## Dependency Chains

| Type | Triggers When |
|------|---------------|
| `afterok:JOBID` | Previous succeeded (exit 0) |
| `afterany:JOBID` | Previous ended (any exit) |
| `afternotok:JOBID` | Previous failed |
| `aftercorr:JOBID` | Corresponding array task OK |
| `singleton` | No other same-name job running |

Combine: `afterok:111,afterok:222` (AND), `afterok:111?afterok:222` (OR).

---

## Resource Right-Sizing

After every job:
```bash
sacct -j JOBID --format=JobID,Elapsed,MaxRSS,ReqMem,NCPUS,AllocGRES,State
```

| Resource | Rule |
|----------|------|
| GPUs | 1 unless multi-GPU scaling verified |
| Memory | 1.5-2x measured MaxRSS |
| CPUs | `num_workers + 1` |
| Walltime | 1.5x measured runtime |

**Over-requesting hurts:** fair-share penalizes allocated (not just used) resources.

---

## Common Commands

```bash
squeue -u $USER                            # my jobs
squeue -u $USER --start                    # estimated start times
scancel <job_id>                           # cancel
sinfo -p gpu -o "%20P %5a %10l %6D %6t"   # partition load
sacct -j <id> --format=JobID,State,ExitCode,Elapsed
sshare -u $USER                            # fair-share standing
scontrol show job <job_id>                 # job details
```
