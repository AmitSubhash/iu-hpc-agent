# Access, Allocations, and Support

## IU's Open-Access Model

IU does NOT use SU billing. No balance that decrements to zero.
- Allocations grant **access** to partitions
- Scheduling uses SLURM fair-share priority
- Heavy usage lowers priority but never hard-blocks you

## RT Projects Portal

All access at [projects.rt.iu.edu](https://projects.rt.iu.edu/).

| Type | Created By | Expires |
|------|------------|---------|
| Research | Faculty/Staff (PI) | June 30 annually |
| Class | Instructors | End of semester |
| HPC for Students | Self-service (PI: lamhuber) | End of semester |

**How:** Create project -> request allocations (BR200, Quartz, Slate) -> RT processes (days).

Account creation: [one.iu.edu/launch-task/iu/account-creation](https://one.iu.edu/launch-task/iu/account-creation)

### Checking Usage

```bash
sshare -u $USER                   # fair-share
sacct -u $USER --starttime=2026-01-01 --format=JobID,JobName,Partition,Elapsed,MaxRSS,State
sacctmgr show assoc where user=$USER format=Account,Share,GrpTRESMins,MaxTRESMins,QOS
```

Web: `https://one.iu.edu/task/iu/hpc-user-dashboard`

### Cost

- BR200 and Quartz: **free** to IU researchers
- Slate-Project: free up to 15 TB
- Jetstream2: free via ACCESS allocations
- SDA (tape): free

---

## ACCESS Allocations

| Tier | Credits | Proposal | Timeline |
|------|---------|----------|----------|
| Explore | 400K | 1 page | ~2 weeks |
| Discover | 1.5M | 1 page | ~2 weeks |
| Accelerate | 3M | 3 pages | ~2 weeks |
| Maximize | No cap | 10+5 pages | Semi-annual |

PI eligibility: US-based researcher/educator, graduate+. Grad students need advisor co-PI for Explore/Discover. NSF GRFP holders can lead without advisor.

---

## Jetstream2 (NSF Cloud)

| Attribute | Value |
|-----------|-------|
| Location | IU Bloomington |
| Hardware | AMD Milan 7713, 512 GB/node |
| GPUs | 90 nodes x 4x A100 (360 total) |
| Access | Via ACCESS allocations |

| GPU Flavor | vCPUs | RAM | GPU | SU/hr |
|------------|-------|-----|-----|-------|
| g3.medium | 8 | 30 GB | 25% A100 | 16 |
| g3.large | 16 | 60 GB | 50% A100 | 32 |
| g3.xl | 32 | 120 GB | 1x A100 | 64 |

Trial: 90 days, 1 small CPU VM, no proposal.

---

## On-Cluster Verification

```bash
sinfo -o "%20P %10a %10l %6D %8c %10m %25f %G"         # partitions
sacctmgr show qos format=Name,Priority,MaxWall,MaxTRESPerUser  # QOS
sprio -w                                                  # priority weights
module spider python                                      # available Python
quota -s && lfs quota -hu $USER /N/slate/ /N/scratch/     # storage
```

---

## Training Resources

IU offers free HPC training at [ittraining.iu.edu/explore-topics/research-computing](https://ittraining.iu.edu/explore-topics/research-computing/index.html):

| Course | Covers |
|--------|--------|
| Intro to HPC | Login, job creation, RED, basics |
| Intro to Deep Learning | NeuralNets, RandomForests on BR200 |
| LLMs on HPC | Multi-GPU training, fine-tuning |
| Research Programming with GPUs | CUDA, GPU optimization |
| Storage 101/201 | Slate, Geode, SDA, data management |
| HPC for Biologists | Bio-specific HPC workflows |

Plus: **Supercomputing for Everyone Series (SC4ES)** workshops and Wednesdays office hours.

---

## Support

| Channel | Contact |
|---------|---------|
| Help Form | [projects.rt.iu.edu/help/?queue=hps](https://projects.rt.iu.edu/help/?queue=hps) |
| Office Hours | Wednesdays 2-3pm ET, Zoom |
| Email | radl@iu.edu |
| Software Requests | [uits.iu.edu/.../hpc-software-request](https://uits.iu.edu/services/technology-for-research/support/hpc-software-request.html) |
| Knowledge Base | [kb.iu.edu](https://kb.iu.edu) |
| IT Training | [ittraining.iu.edu](https://ittraining.iu.edu) |
| Research Data Commons | [researchdata.iu.edu](https://researchdata.iu.edu) |

---

## Sources

- [About BR200](https://kb.iu.edu/d/brcc) | [Quartz](https://servicenow.iu.edu/kb?id=kb_article_view&sysparm_article=KB0023985)
- [Run GPU Jobs](https://kb.iu.edu/d/avjk) | [SLURM at IU](https://kb.iu.edu/d/awrz)
- [Slate](https://kb.iu.edu/d/aqnk) | [Home Quotas](https://kb.iu.edu/d/bhrl)
- [ACCESS](https://allocations.access-ci.org/) | [Jetstream2](https://docs.jetstream-cloud.org/)
- [SLURM Priority](https://slurm.schedmd.com/priority_multifactor.html) | [RT Projects](https://projects.rt.iu.edu/)
