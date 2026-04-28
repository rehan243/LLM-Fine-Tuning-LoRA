# tiny cli to list checkpoints newest first; works on a plain directory tree
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass
class Ckpt:
    path: Path
    mtime: float

    @property
    def mtime_dt(self) -> datetime:
        return datetime.fromtimestamp(self.mtime)


def find_checkpoints(root: Path, marker: str = "adapter_config.json") -> List[Ckpt]:
    out: List[Ckpt] = []
    for dirpath, _, files in os.walk(root):
        if marker in files:
            p = Path(dirpath)
            st = p.stat()
            out.append(Ckpt(path=p, mtime=st.st_mtime))
    out.sort(key=lambda c: c.mtime, reverse=True)
    return out


def format_lines(items: Iterable[Ckpt], limit: int) -> Iterable[str]:
    for i, ck in enumerate(items):
        if i >= limit:
            break
        yield f"{ck.mtime_dt.isoformat()}  {ck.path.as_posix()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    for line in format_lines(find_checkpoints(args.root), args.limit):
        print(line)


if __name__ == "__main__":
    main()
