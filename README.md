# IU HPC Agent Skill

A Claude Code skill that turns AI into an expert assistant for Indiana University's Big Red 200 and Quartz HPC systems. Verified against on-cluster data (March 2026).

## What It Does

When a user asks anything about IU's HPC systems, Claude reads the relevant skill files and provides accurate, actionable answers with copy-paste commands and templates. No more digging through scattered KB articles.

**Example interactions:**
- "I need to train a 7B parameter model" -- estimates GPU memory, walltime, fair-share cost, and generates a ready-to-submit SBATCH script
- "My job failed with OOM" -- walks through the diagnostic flowchart (host vs GPU OOM, exit codes, fixes)
- "How do I get started on Big Red 200?" -- provides the full onboarding path: access, SSH, conda, first job
- "I have 10,000 files to process" -- recommends job arrays with the right partition, throttling, and thread settings

## Architecture

```
SKILL.md (89 lines)          <-- Always loaded. Routes to the right file.
  |
  +-- 00-quickstart.md        New user onboarding (access, SSH, conda, first job)
  +-- 01-hardware.md          BR200 + Quartz specs, GPU topology, network
  +-- 02-slurm.md             Partitions, billing, QOS, scheduler, backfill
  +-- 03-gpu-optimization.md  DDP, FSDP, mixed precision, checkpointing, NCCL
  +-- 04-cpu-optimization.md  NUMA topology, SMT, BLAS threading, compiler flags
  +-- 05-storage-envs.md      Storage tiers, I/O tuning, conda, modules, containers
  +-- 06-templates.md         10 copy-paste SBATCH templates
  +-- 07-recipes-tips.md      Workflows, data transfer, tips, efficiency checklist
  +-- 08-access-support.md    Allocations, ACCESS, Jetstream2, training, support
  +-- 09-resource-estimator.md GPU memory formulas, walltime, sizing cheat sheet
  +-- 10-troubleshooting.md   Diagnostic flowcharts for every failure mode
```

**Token-efficient by design:** Claude loads only the 89-line router by default, then reads 1-2 topic files on demand. A typical query uses ~250 lines of context instead of dumping 2000+.

## What's Covered

| Category | Highlights |
|----------|-----------|
| **Hardware** | BR200 (640 CPU + 66 GPU nodes, A100-40GB, NVLink mesh), Quartz (92 CPU + 22 V100 + 12 H100 nodes) |
| **SLURM** | All partitions, billing weights (1 GPU = 16x CPU), QOS limits, scheduler config, backfill strategies |
| **GPU Training** | DDP/FSDP decision tree, multi-node launch, BF16, activation checkpointing, gradient accumulation, PyTorch checklist |
| **CPU Optimization** | Zen 2 NUMA topology (8 nodes/compute), SMT guidance, BLAS thread control, MKL-on-AMD fix, compiler flags |
| **Storage** | 5 tiers (Home/Slate/Scratch/SDA/Geode-Project), Lustre tuning, local tmpfs staging, small-file avoidance, Globus endpoints |
| **Templates** | GPU training, CPU preprocessing, HP sweep arrays, multi-node distributed, interactive, Jupyter, pipeline submission, memory-intensive |
| **Resource Estimation** | GPU memory formulas (16B/param FP32), model size tables (10M to 70B), walltime benchmarks, fair-share cost calculator |
| **Troubleshooting** | Exit code decoder, OOM (host + GPU), PENDING diagnosis, NCCL errors, filesystem issues, CUDA mismatches, low GPU utilization |
| **Onboarding** | 7-step quickstart from zero to running GPU job, common first-day mistakes, Research Desktop (RED) GUI option |
| **Access & Training** | RT Projects, ACCESS allocations, Jetstream2, 6+ training courses, Wednesday office hours |

## Installation

Copy the skill directory into your Claude Code skills path:

```bash
# For a single user
cp -r iu-hpc/ ~/.claude/skills/iu-hpc/

# Or clone the repo
git clone https://github.com/AmitSubhash/iu-hpc-agent.git ~/.claude/skills/iu-hpc/
```

## Setup

Add 3 lines to your `~/.bashrc` on BR200/Quartz:

```bash
export SLURM_ACCOUNT="your_account_id"   # from projects.rt.iu.edu
export PROJ="/N/slate/$USER/your_project"
export ENVS="/N/slate/$USER/envs"
```

All templates use `<account>` for SBATCH `-A` and `$USER` for paths. Replace as needed.

## Verification

All hardware specs, SLURM configuration, and billing weights were verified by SSHing into BR200 compute nodes (March 2026). To re-verify:

```bash
ssh br200
sinfo -o "%20P %10a %10l %6D %8c %10m %G"         # partitions
sacctmgr show qos format=Name,Priority,MaxWall      # QOS
scontrol show config | grep -E "Priority|Backfill"   # scheduler
```

## Contributing

- Hardware data should be verified on-cluster before updating
- Templates must include `set -euo pipefail`, explicit `--mem`, `module purge`, and thread control vars
- Keep each file under 300 lines
- No hardcoded usernames, account IDs, or personal paths -- use `$USER` and `<account>`

## Sources

- [About Big Red 200](https://kb.iu.edu/d/brcc) | [About Quartz](https://servicenow.iu.edu/kb?id=kb_article_view&sysparm_article=KB0023985)
- [Run GPU Jobs](https://kb.iu.edu/d/avjk) | [SLURM at IU](https://kb.iu.edu/d/awrz)
- [Slate Storage](https://kb.iu.edu/d/aqnk) | [Home Quotas](https://kb.iu.edu/d/bhrl)
- [ACCESS Allocations](https://allocations.access-ci.org/) | [Jetstream2](https://docs.jetstream-cloud.org/)
- [RT Projects](https://projects.rt.iu.edu/) | [IT Training](https://ittraining.iu.edu)
- [IU Globus](https://globus.iu.edu/) | [Research Data Commons](https://researchdata.iu.edu)
