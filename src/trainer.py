"""LoRA trainer — the part that actually eats your GPU budget."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
from evaluate import load as load_metric

logger = logging.getLogger(__name__)


@dataclass
class TrainState:
    global_step: int = 0
    best_eval_loss: float = float("inf")


class LoRATrainer:
    """Fine-tune with LoRA. Assumes you already fought CUDA_VISIBLE_DEVICES."""

    def __init__(
        self,
        model_name: str,
        train_ds: Dataset,
        eval_ds: Dataset,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
        lr: float = 2e-4,
        epochs: int = 3,
        per_device_batch: int = 4,
        grad_accum: int = 8,
        max_seq_len: int = 2048,
        warmup_ratio: float = 0.03,
        output_dir: str = "checkpoints/lora",
        bf16: bool = True,
        seed: int = 42,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ) -> None:
        self.accelerator = Accelerator(gradient_accumulation_steps=grad_accum, mixed_precision="bf16" if bf16 else "no")
        set_seed(seed)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if bf16 else torch.float32,
            device_map=None,
            trust_remote_code=True,
        )
        target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        self.model = get_peft_model(base, peft_cfg)
        self.model.print_trainable_parameters()

        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.train_loader = DataLoader(train_ds, batch_size=per_device_batch, shuffle=True, collate_fn=collator)
        self.eval_loader = DataLoader(eval_ds, batch_size=per_device_batch, shuffle=False, collate_fn=collator)

        opt = AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        steps_per_epoch = math.ceil(len(self.train_loader) / grad_accum)
        total_steps = max(1, steps_per_epoch * epochs)
        warmup_steps = int(total_steps * warmup_ratio)
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

        self.model, self.opt, self.train_loader, self.eval_loader, self.sched = self.accelerator.prepare(
            self.model, opt, self.train_loader, self.eval_loader, sched
        )
        self.grad_accum = grad_accum
        self.epochs = epochs
        self.max_seq_len = max_seq_len
        self.state = TrainState()

        self._rouge = None
        try:
            self._rouge = load_metric("rouge")
        except Exception as exc:  # pragma: no cover - offline CI
            logger.warning("ROUGE metric unavailable (%s); eval will skip it.", exc)

        if wandb_project and self.accelerator.is_main_process:
            wandb.init(project=wandb_project, name=wandb_run_name, config={"model": model_name, "lora_r": lora_r})

    def _forward_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        batch = {k: v.to(self.accelerator.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        out = self.model(**batch)
        return out.loss

    def train(self) -> None:
        self.model.train()
        for epoch in range(self.epochs):
            self.accelerator.print(f"epoch {epoch + 1}/{self.epochs}")
            for step, batch in enumerate(self.train_loader):
                with self.accelerator.accumulate(self.model):
                    loss = self._forward_batch(batch)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.sched.step()
                    self.opt.zero_grad()
                self.state.global_step += 1
                if self.accelerator.is_main_process and self.state.global_step % 50 == 0:
                    wandb.log({"train/loss": loss.detach().float().item(), "step": self.state.global_step})
            metrics = self.evaluate()
            if self.accelerator.is_main_process:
                wandb.log({**metrics, "epoch": epoch})
                if metrics.get("eval/loss", float("inf")) < self.state.best_eval_loss:
                    self.state.best_eval_loss = metrics["eval/loss"]
                    self.save_adapter()

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        losses: list[float] = []
        preds: list[str] = []
        refs: list[str] = []
        rouge_batches = 0
        for batch in self.eval_loader:
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            labels = batch.get("labels")
            out = self.model(**batch)
            losses.append(out.loss.detach().float().item())
            # ROUGE: two batches max — full decode pass is a trap for the impatient
            if self._rouge is not None and labels is not None and rouge_batches < 2:
                gen = self.accelerator.unwrap_model(self.model).generate(
                    input_ids=batch["input_ids"][:, :48],
                    max_new_tokens=48,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                preds.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
                lab = labels.clone()
                lab[lab == -100] = self.tokenizer.pad_token_id or 0
                refs.extend(self.tokenizer.batch_decode(lab, skip_special_tokens=True))
                rouge_batches += 1
        mean_loss = float(sum(losses) / max(len(losses), 1))
        ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
        metrics: dict[str, float] = {"eval/loss": mean_loss, "eval/perplexity": ppl}
        if self._rouge and preds and refs:
            rouge = self._rouge.compute(predictions=preds, references=refs, use_stemmer=True)
            metrics["eval/rougeL"] = float(rouge.get("rougeL", {}).get("fmeasure", 0.0))
        self.model.train()
        return metrics

    def save_adapter(self) -> None:
        if not self.accelerator.is_main_process:
            return
        path = self.output_dir / f"adapter_step_{self.state.global_step}"
        self.accelerator.unwrap_model(self.model).save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Saved adapter to %s", path)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print("Import DatasetBuilder and wire datasets before running.")
