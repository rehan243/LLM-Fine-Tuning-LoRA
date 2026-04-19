# LoRA Fine-Tuning Best Practices

## Rank Selection
- Rank 8-16: Simple classification, sentiment
- Rank 16-32: General instruction following
- Rank 32-64: Domain-specific knowledge tasks
- Above 64: Diminishing returns for most use cases

## Target Modules
Always include both attention and MLP layers:
- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention)
- `gate_proj`, `up_proj`, `down_proj` (MLP)

## Training Tips
- Use bf16 mixed precision on A100/H100
- Learning rate: 1e-4 to 5e-4 with cosine scheduler
- Gradient accumulation for effective batch size 32+
- Merge weights back for inference (eliminates adapter overhead)