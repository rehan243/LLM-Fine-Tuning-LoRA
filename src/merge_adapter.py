"""Merge LoRA back into base weights, then export something inference servers can load."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class MergeError(RuntimeError):
    """Raised when merge/export steps fail hard enough to cancel the release train."""


def _assert_adapter_files(adapter_dir: Path) -> None:
    if not adapter_dir.is_dir():
        raise MergeError(f"adapter path is not a directory: {adapter_dir}")
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.exists():
        raise MergeError(f"missing {cfg.name} — are you sure this is a PEFT export?")


def merge_and_export(
    base_id: str,
    adapter_dir: str | Path,
    out_dir: str | Path,
    dtype: str = "bfloat16",
) -> Path:
    """Library entrypoint so __init__ imports don't lie."""
    out = Path(out_dir)
    merge_lora_into_base(base_id, Path(adapter_dir), out, dtype=dtype)
    return out


def merge_lora_into_base(base_id: str, adapter_dir: Path, out_dir: Path, dtype: str = "bfloat16") -> None:
    _assert_adapter_files(adapter_dir)
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    tok.save_pretrained(out_dir)
    logger.info("Merged model saved to %s", out_dir)


def quantize_gguf(model_dir: Path, outfile: Path, method: str = "q4_k_m") -> None:
    """Shell out to llama.cpp convert — if missing, we fail loud."""
    cmd = [
        "python",
        str(Path("third_party/llama.cpp/convert_hf_to_gguf.py")),
        str(model_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        method,
    ]
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.error("GGUF conversion failed (%s). Install llama.cpp tooling or fix path.", exc)
        raise


def quantize_awq_stub(model_dir: Path, outfile: Path) -> None:
    """AWQ path is environment-specific; stub documents the hook."""
    logger.warning(
        "AWQ export not run in this environment — use autoawq or your vendor script on %s -> %s",
        model_dir,
        outfile,
    )


def push_to_hub(local_dir: Path, repo_id: str, private: bool = True, token: Optional[str] = None) -> None:
    if token:
        login(token=token)
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type="model")
    logger.info("Pushed %s to %s", local_dir, repo_id)


def disk_usage_gb(path: Path) -> float:
    """Rough size check before upload — Hub rejects comically huge folders sometimes."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                continue
    return total / (1024**3)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Merge LoRA and optionally quantize / push.")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--gguf", type=Path, default=None)
    p.add_argument("--awq-out", type=Path, default=None)
    p.add_argument("--push", type=str, default=None, help="hf.co repo id")
    p.add_argument("--public", action="store_true")
    args = p.parse_args(argv)

    merge_lora_into_base(args.base, args.adapter, args.out, dtype=args.dtype)
    sz = disk_usage_gb(args.out)
    logger.info("merged checkpoint on disk ~%.2f GiB", sz)
    if args.gguf:
        try:
            quantize_gguf(args.out, args.gguf)
        except Exception:
            logger.exception("GGUF step skipped/failed")
    if args.awq_out:
        quantize_awq_stub(args.out, args.awq_out)
    if args.push:
        push_to_hub(args.out, args.push, private=not args.public)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
