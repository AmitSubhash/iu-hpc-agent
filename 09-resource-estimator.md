# Resource Estimation Guide

Use this file when users describe a workload and need help estimating resources
and generating an SBATCH script. Walk through the estimation, explain the reasoning,
then output a ready-to-submit script.

---

## Step 1: Classify the Workload

Ask or infer these from the user's description:

| Question | Why It Matters |
|----------|---------------|
| What are you running? (training, inference, preprocessing, simulation) | Determines partition and GPU need |
| Model size (parameter count) or dataset size? | Determines GPU memory and node count |
| How long does one epoch/iteration take? | Determines walltime |
| How many files/samples? | Determines if job array vs single job |
| Any framework specifics? (PyTorch DDP, FSDP, MCX, MATLAB) | Affects launch pattern |

---

## Step 2: Estimate GPU Memory

### Model Training (PyTorch)

Rough formula for GPU memory during training:

```
Memory = model_params + gradients + optimizer_state + activations

model_params   = num_params * bytes_per_param
gradients      = num_params * bytes_per_param
optimizer_state = num_params * optimizer_multiplier * bytes_per_param
activations    = depends on batch_size, seq_len, hidden_dim
```

| Precision | bytes_per_param | Optimizer (Adam) multiplier | Total per-param |
|-----------|----------------|---------------------------|-----------------|
| FP32 | 4 B | 2 (m + v) | 4 + 4 + 8 = **16 B/param** |
| BF16 mixed | 2 B model, 4 B master | 2 | 2 + 2 + (4 + 8) = **16 B/param** |
| BF16 pure | 2 B | 2 | 2 + 2 + 4 = **8 B/param** |

**Quick estimates (FP32 Adam, excluding activations):**

| Model Size | Memory (params+grad+optim) | Fits A100-40GB? | Strategy |
|-----------|---------------------------|-----------------|----------|
| 10M | ~160 MB | Yes (1 GPU) | DDP optional |
| 100M | ~1.6 GB | Yes (1 GPU) | DDP for speed |
| 500M | ~8 GB | Yes (1 GPU) | BF16 recommended |
| 1B | ~16 GB | Tight (1 GPU) | BF16 required |
| 3B | ~48 GB | No | FSDP across 4 GPUs |
| 7B | ~112 GB | No | FSDP across 4 GPUs + activation ckpt |
| 13B | ~208 GB | No | FSDP across 8+ GPUs (multi-node) |
| 70B | ~1.1 TB | No | Multi-node FSDP, 8+ nodes |

**Activation memory** adds 20-100% on top, depending on batch size and sequence length. Activation checkpointing trades ~30% speed for ~60% activation memory reduction.

### Model Inference

Inference is cheaper (no gradients, no optimizer):
```
Memory ~= num_params * bytes_per_param * 1.2  (1.2x for KV cache overhead)
```

| Model | FP16 Inference | Fits 1x A100-40GB? |
|-------|---------------|-------------------|
| 7B | ~14 GB | Yes |
| 13B | ~26 GB | Yes |
| 30B | ~60 GB | No, need 2 GPUs |
| 70B | ~140 GB | No, need 4 GPUs |

---

## Step 3: Estimate Walltime

### Training

```
time_per_epoch = (num_samples / batch_size) * time_per_step
total_time = time_per_epoch * num_epochs * 1.1  (10% overhead: checkpointing, logging)
```

**If user doesn't know time_per_step**, use these rough estimates for A100:

| Task | Throughput (A100, BF16) |
|------|------------------------|
| Image classification (ResNet-50) | ~3,000 img/s per GPU |
| Object detection (YOLO) | ~200-500 img/s per GPU |
| NLP fine-tune (BERT-base, seq=512) | ~300 samples/s per GPU |
| LLM fine-tune (7B, seq=2048) | ~2-5 samples/s per GPU |
| LLM fine-tune (70B, FSDP, 16 GPUs) | ~0.5-1 samples/s total |
| Diffusion model training | ~5-20 img/s per GPU |

**Recommendation:** Always request a `gpu-debug` session first (30 min) to measure actual throughput, then calculate walltime from that.

### Preprocessing (CPU)

| Task | Rate per Core |
|------|--------------|
| NIfTI/DICOM conversion | ~5-20 files/min |
| Image resize/augmentation | ~50-200 img/s |
| Text tokenization | ~10K-100K samples/s |
| Feature extraction (scikit-learn) | Varies widely |

Scale linearly with `--cpus-per-task` for embarrassingly parallel tasks.

---

## Step 4: Estimate Fair-Share Cost

```
CPU billing = cpus * 1.0 + mem_gb * 0.512
GPU billing = cpus * 1.0 + mem_gb * 0.128 + gpus * 16.0
Total cost = billing_units * hours
```

| Job Type | Typical Config | Billing/hr | 24h Cost |
|----------|---------------|-----------|----------|
| 1 GPU, 8 CPU, 32 GB | gpu | 8 + 4.1 + 16 = 28 | 672 |
| 4 GPU, 32 CPU, 200 GB | gpu | 32 + 25.6 + 64 = 122 | 2,918 |
| 64 CPU, 128 GB | general | 64 + 65.5 = 130 | 3,110 |
| Job array: 50 x (1 GPU, 4h) | gpu | 28 * 50 = 1,400 | 5,600 |

**IU doesn't charge money**, but billing units affect fair-share priority. Lower cost = higher priority for next jobs.

---

## Step 5: Generate the SBATCH Script

Use the estimation to select from templates in `06-templates.md`, adjusting:

1. **Partition**: `general` (CPU) or `gpu` (GPU)
2. **GPUs**: 0 (CPU), 1, or 4 based on model size
3. **CPUs**: `num_workers + 1` for GPU, or match parallelism for CPU
4. **Memory**: 1.5x estimated peak (check with `sacct --format=MaxRSS`)
5. **Time**: 1.5x estimated, or use `--time-min` for range
6. **Array**: If >100 independent tasks

### Sizing Cheat Sheet

| Workload Description | Partition | GPUs | CPUs | Memory | Time |
|---------------------|-----------|------|------|--------|------|
| Fine-tune BERT on 100K samples | gpu | 1 | 8 | 32G | 2-4h |
| Train ResNet-50 on ImageNet | gpu | 4 | 32 | 200G | 12-24h |
| Fine-tune 7B LLM (LoRA) | gpu | 1 | 8 | 64G | 4-8h |
| Fine-tune 7B LLM (full) | gpu | 4 | 32 | 400G | 1-2d |
| Preprocess 10K NIfTI files | general | 0 | 64 | 128G | 2-6h |
| Preprocess 10K NIfTI (array) | general | 0 | 4/task | 8G/task | 30m/task |
| HP sweep (50 configs, small model) | gpu | 1/task | 8/task | 32G/task | 4h/task |
| MCX photon simulation | gpu | 4 | 16 | 200G | 2-12h |
| MATLAB analysis (parpool) | general | 0 | 16 | 64G | 4-12h |
| Large matrix solve (>200 GB) | general (Quartz) | 0 | 128 | 480G | 6-24h |

---

## Example Walkthrough

**User says:** "I need to fine-tune Llama-2-7B on my 50K sample dataset with full parameter training"

**Estimation:**
1. **Model:** 7B params, FP32 Adam = ~112 GB (params + grad + optimizer)
2. **Plus activations:** ~30-50 GB at batch_size=1, seq_len=2048
3. **Total:** ~150 GB -- won't fit 1x A100-40GB
4. **Strategy:** BF16 mixed precision + FSDP across 4 GPUs
   - With BF16 FSDP: ~112 GB / 4 GPUs = ~28 GB/GPU + activations
   - Fits with activation checkpointing
5. **Throughput:** ~3 samples/s on 4x A100 (BF16, FSDP)
6. **Time:** 50K / 3 = ~16.7K seconds/epoch = ~4.6h/epoch
7. **3 epochs:** ~14h + 10% overhead = **~15h**
8. **Fair-share:** 4 GPU job * 15h = moderate cost

**Result:** `-p gpu --gres=gpu:4 --cpus-per-task=32 --mem=400G -t 18:00:00`

---

## Recommend: Profile First

Always suggest users run a **quick profiling job** before committing to a long run:

```bash
salloc -A $SLURM_ACCOUNT -p gpu-debug --gres=gpu:1 --cpus-per-task=8 --mem=32G -t 00:30:00

# Inside the session:
python train.py --max-steps 50 --profile
nvidia-smi dmon -d 5  # check GPU utilization

# After:
# - Note GPU util %, memory used, time per step
# - Use those numbers to estimate full run
```

This prevents wasting hours on a misconfigured job.
