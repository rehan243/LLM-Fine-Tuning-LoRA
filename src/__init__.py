"""lora fine-tuning package — configs live in ../configs, stop asking."""

from src.trainer import LoRATrainer
from src.data_pipeline import DatasetBuilder
from src.merge_adapter import merge_and_export, merge_lora_into_base, push_to_hub

__all__ = [
    "LoRATrainer",
    "DatasetBuilder",
    "merge_and_export",
    "merge_lora_into_base",
    "push_to_hub",
]

def safe_merge_and_export(*args, **kwargs) -> None:
    """merges and exports with error handling"""
    try:
        merge_and_export(*args, **kwargs)
    except Exception as e:
        print(f"error during merge and export: {e}")

def safe_merge_lora_into_base(*args, **kwargs) -> None:
    """merges lora into base model with error handling"""
    try:
        merge_lora_into_base(*args, **kwargs)
    except Exception as e:
        print(f"error during merging lora into base: {e}")

def safe_push_to_hub(*args, **kwargs) -> None:
    """pushes model to hub with error handling"""
    try:
        push_to_hub(*args, **kwargs)
    except Exception as e:
        print(f"error during push to hub: {e}")