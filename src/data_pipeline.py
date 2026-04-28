"""Dataset plumbing — HF, JSONL, chat templates, the usual suspects."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class DataFormat(str, Enum):
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"


@dataclass
class DatasetBuilder:
    """Build tokenized train/eval splits. Handles the two formats people actually use."""

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 2048
    train_ratio: float = 0.95
    pad_to_multiple_of: int = 8

    def from_hf(
        self,
        path: str,
        split: str = "train",
        text_field: str = "text",
        streaming: bool = False,
    ) -> Dataset:
        try:
            return load_dataset(path, split=split, streaming=streaming)
        except Exception:
            # dataset scripts vary; fallback to data files
            return load_dataset("json", data_files=path, split=split, streaming=streaming)

    def from_jsonl(self, path: Union[str, Path]) -> Dataset:
        rows: list[dict[str, Any]] = []
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return Dataset.from_list(rows)

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # ancient tokenizer — stringify and hope for the best
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|{role}|>\n{content}\n")
        return "".join(parts)

    def format_row(self, row: dict[str, Any], fmt: DataFormat) -> str:
        if fmt == DataFormat.INSTRUCTION:
            instr = row.get("instruction") or row.get("input", "")
            out = row.get("output") or row.get("response", "")
            user_msg = instr.strip()
            asst_msg = out.strip()
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": asst_msg},
            ]
            return self._apply_chat_template(messages)
        # conversational
        msgs = row.get("messages")
        if not msgs and "conversations" in row:
            msgs = row["conversations"]
        if not isinstance(msgs, list):
            raise ValueError("conversation row needs `messages` list")
        return self._apply_chat_template(msgs)

    def tokenize_batch(self, examples: dict[str, list[Any]], fmt: DataFormat) -> dict[str, list[Any]]:
        """For Dataset.map(batched=True) callbacks outside of build()."""
        if fmt == DataFormat.INSTRUCTION:
            if "instruction" not in examples or "output" not in examples:
                raise ValueError("instruction tuning expects `instruction` and `output` keys")
            texts = [
                self.format_row({"instruction": i, "output": o}, fmt)
                for i, o in zip(examples["instruction"], examples["output"])
            ]
        else:
            if "messages" not in examples:
                raise ValueError("conversational format expects `messages` key")
            texts = [self.format_row({"messages": m}, fmt) for m in examples["messages"]]
        tok = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        tok["labels"] = [list(ids) for ids in tok["input_ids"]]
        return tok

    def build_streaming(
        self,
        path: str,
        fmt: DataFormat,
        row_parser: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> Any:
        """If you insist on streaming, you split downstream — we just tokenize lazily."""
        ds = self.from_hf(path, streaming=True)
        def _tok(ex: dict[str, Any]) -> dict[str, Any]:
            row = row_parser(ex) if row_parser else ex
            text = self.format_row(row, fmt)
            single = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False)
            single["labels"] = list(single["input_ids"])
            return single
        return ds.map(_tok)

    def build(
        self,
        raw: Dataset,
        fmt: DataFormat,
        message_key: Optional[str] = None,
    ) -> tuple[Dataset, Dataset]:
        if message_key:
            raw = raw.map(lambda x: {"messages": x[message_key]})

        def _map_fn(batch: dict[str, Any]) -> dict[str, Any]:
            if fmt == DataFormat.INSTRUCTION:
                texts = []
                for inst, out in zip(batch["instruction"], batch["output"]):
                    texts.append(self.format_row({"instruction": inst, "output": out}, fmt))
            else:
                texts = [self.format_row({"messages": m}, fmt) for m in batch["messages"]]
            tok = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            tok["labels"] = [list(ids) for ids in tok["input_ids"]]
            return tok

        if "instruction" in raw.column_names:
            cols = ["instruction", "output"]
        elif "messages" in raw.column_names:
            cols = ["messages"]
        else:
            raise ValueError(f"Unsupported columns: {raw.column_names}")

        tokenized = raw.map(_map_fn, batched=True, remove_columns=[c for c in raw.column_names if c in cols])
        split = tokenized.train_test_split(test_size=1.0 - self.train_ratio, seed=42)
        return split["train"], split["test"]
