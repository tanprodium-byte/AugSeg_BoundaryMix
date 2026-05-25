from __future__ import annotations

import csv
import json
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "mean") and not isinstance(x, (float, int)):
            try:
                if x.numel() != 1:
                    x = x.float().mean()
            except Exception:
                pass
        if hasattr(x, "item"):
            x = x.item()
        return float(x)
    except Exception:
        return float(default)


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "item"):
            x = x.item()
        return int(x)
    except Exception:
        return int(default)


def get_git_commit(repo_root: str | Path = ".") -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def get_or_create_run_id(save_path: str | Path, run_name: Optional[str] = None) -> str:
    save_dir = ensure_dir(save_path)
    run_id_file = save_dir / "run_id.txt"

    if run_id_file.exists():
        run_id = run_id_file.read_text().strip()
        if run_id:
            return run_id

    prefix = run_name or "run"
    prefix = str(prefix).replace("/", "_").replace(" ", "_")
    run_id = f"{prefix}_{now_string()}_{uuid.uuid4().hex[:8]}"
    run_id_file.write_text(run_id + "\n")
    return run_id


def append_csv_row(csv_path: str | Path, row: Dict[str, Any]) -> None:
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)

    # Convert Path / tensors / numpy scalars to plain serializable values.
    clean_row: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, Path):
            clean_row[k] = str(v)
        elif hasattr(v, "detach") or hasattr(v, "item"):
            clean_row[k] = safe_float(v)
        elif isinstance(v, (dict, list, tuple)):
            clean_row[k] = json.dumps(v, ensure_ascii=False)
        else:
            clean_row[k] = v

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(clean_row)


def trim_csv_rows_by_epoch(
    csv_path: str | Path,
    keep_epoch_lt: int,
    epoch_col_candidates: Iterable[str] = ("epoch", "meta/epoch"),
) -> None:
    """
    Keep only rows with epoch < keep_epoch_lt.

    Useful when resuming from checkpoint at start_epoch:
    if CSV contains partial rows from an interrupted epoch, remove them.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return

        epoch_col = None
        for c in epoch_col_candidates:
            if c in fieldnames:
                epoch_col = c
                break

        if epoch_col is None:
            return

        kept = []
        for row in reader:
            try:
                e = int(float(row.get(epoch_col, -1)))
            except Exception:
                e = -1
            if e < int(keep_epoch_lt):
                kept.append(row)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def default_log_paths(save_path: str | Path) -> Dict[str, Path]:
    save_dir = ensure_dir(save_path)
    return {
        "epoch_csv": save_dir / "epoch_metrics.csv",
        "iter_csv": save_dir / "iter_metrics.csv",
        "manifest": save_dir / "manifest.json",
        "run_id": save_dir / "run_id.txt",
    }
