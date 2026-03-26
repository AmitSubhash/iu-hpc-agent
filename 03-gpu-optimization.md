# GPU Optimization

## Multi-GPU Strategy

```
Model fits in 1 GPU (params + activations + optimizer < 35 GB)?
  YES -> Use 1 GPU
    Need larger effective batch? -> Gradient accumulation
    Need more throughput?       -> DDP with 4 GPUs (NVLink, single node)
  NO -> FSDP (FULL_SHARD) across 4 GPUs
    Still doesn't fit?          -> Multi-node FSDP (slower, Slingshot)
```

## DDP (Default Choice)

Each GPU holds full model replica. Gradient all-reduce over NVLink.
```bash
torchrun --nproc_per_node=4 train.py
```

## FSDP (Large Models)

Shards parameters + gradients + optimizer across GPUs.
~58% memory reduction vs DDP, ~25% speed penalty.

## Multi-Node Launch

```bash
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29500}"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    bash -lc "torchrun \
        --nnodes=$SLURM_NNODES --nproc_per_node=4 \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        train.py [args]"
```

---

## NCCL Configuration

```bash
export NCCL_DEBUG=WARN                  # INFO for debugging
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=hsn           # Slingshot HSN interface
export TORCH_DISTRIBUTED_DEBUG=DETAIL   # only when debugging
# Fallback if IB issues: export NCCL_IB_DISABLE=1
```

---

## Mixed Precision (BF16 on A100)

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)
# No loss scaling needed (unlike FP16). 2x memory savings, ~2x throughput.
```

## Activation Checkpointing

Trade ~30% speed for ~60% memory savings:
```python
from torch.utils.checkpoint import checkpoint
output = checkpoint(transformer_block, input, use_reentrant=False)
```

## Gradient Accumulation

```python
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

---

## PyTorch Performance Checklist

- `torch.backends.cudnn.benchmark = True`
- `pin_memory=True` in DataLoader
- `num_workers=4-8` per GPU (match `--cpus-per-task`)
- `torch.compile(model)` (PyTorch 2.0+)
- Create tensors on GPU: `torch.randn(size, device='cuda')`
- `channels_last` memory format for CNNs
- Tensor dims as multiples of 8 (Tensor Core alignment)
- `set_to_none=True` in `optimizer.zero_grad()`

## GPU Monitoring

```bash
nvidia-smi dmon -s pucvmet -d 30 -o DT > gpu_monitor_${SLURM_JOB_ID}.csv &
MONITOR_PID=$!
# ... training ...
kill $MONITOR_PID
```

---

## Checkpointing and Fault Tolerance

### SLURM Signal-Based

```bash
#SBATCH --signal=B:USR1@300   # SIGUSR1 300s before walltime
#SBATCH --requeue             # auto-requeue after checkpoint
```

### Signal Handler

```python
import signal, sys, os

def checkpoint_handler(signum, frame):
    print(f"Signal {signum}, saving checkpoint...")
    save_checkpoint(model, optimizer, scheduler, epoch, step,
                    path=os.path.join(os.environ.get("CKPT", "."), "interrupt_ckpt.pt"))
    sys.exit(0)

signal.signal(signal.SIGUSR1, checkpoint_handler)
signal.signal(signal.SIGTERM, checkpoint_handler)
```

### Save / Load

```python
def save_checkpoint(model, optimizer, scheduler, epoch, step, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch, "step": step,
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }, path)

def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    torch.random.set_rng_state(ckpt["rng_state"])
    torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    return ckpt["epoch"], ckpt["step"]
```

### Best Practices

- Save every N steps AND on SLURM signal
- Keep only last 3 checkpoints (save disk)
- Always save RNG states for exact reproducibility
- Write to Slate (`$CKPT`), NOT Scratch (30-day purge)

### Auto-Resubmission Chain

```bash
JOB=$(sbatch --parsable train_segment.sh)
for i in $(seq 2 8); do
    JOB=$(sbatch --parsable --dependency=afterany:$JOB train_segment.sh)
done
```

---

## NCCL Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Cuda failure 'out of memory'` | GPU OOM | Reduce batch, enable AMP |
| `Call to ibv_reg_mr failed` | Pinned memory limit | Check `ulimit -l` |
| Timeout in `init_process_group` | Wrong MASTER_ADDR/port | Use `scontrol show hostnames` |
| `No such device` | Wrong network interface | `NCCL_SOCKET_IFNAME=hsn` |
| Slow multi-node | TCP fallback | Verify `NCCL_IB_DISABLE` NOT set |

Debug: `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL TORCH_DISTRIBUTED_DEBUG=DETAIL`
