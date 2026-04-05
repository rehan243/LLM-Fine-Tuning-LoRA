"""Training Configuration Presets for LoRA Fine-Tuning - Rehan Malik"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    model_name: str
    lora_r: int
    lora_alpha: int
    learning_rate: float
    epochs: int
    batch_size: int
    grad_accum: int
    warmup_ratio: float
    bf16: bool = True
    gradient_checkpointing: bool = True

    @property
    def effective_batch_size(self):
        return self.batch_size * self.grad_accum


CONFIGS = {
    "llama2-7b-general": TrainingConfig(
        "meta-llama/Llama-2-7b-hf", 16, 32, 2e-4, 3, 4, 8, 0.03),
    "llama2-7b-code": TrainingConfig(
        "meta-llama/Llama-2-7b-hf", 32, 64, 1e-4, 2, 4, 8, 0.05),
    "llama2-13b-qlora": TrainingConfig(
        "meta-llama/Llama-2-13b-hf", 16, 32, 1e-4, 2, 2, 16, 0.03),
    "mistral-7b-instruct": TrainingConfig(
        "mistralai/Mistral-7B-v0.1", 16, 32, 2e-4, 3, 4, 8, 0.03),
    "llama3-8b-chat": TrainingConfig(
        "meta-llama/Meta-Llama-3-8B", 32, 64, 1e-4, 2, 4, 8, 0.05),
}


if __name__ == "__main__":
    for name, cfg in CONFIGS.items():
        print(f"{name}: r={cfg.lora_r}, lr={cfg.learning_rate}, "
              f"eff_batch={cfg.effective_batch_size}")
