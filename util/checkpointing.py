from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _atomic_torch_save(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_checkpoint_state(
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_miou: float,
    run_id: str,
    cfg: Dict[str, Any],
    args: Any,
    teacher_model: Optional[torch.nn.Module] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_to_save = unwrap_model(model)

    state: Dict[str, Any] = {
        "epoch": int(epoch),
        "model_state": model_to_save.state_dict(),
        "best_miou": float(best_miou),
        "run_id": str(run_id),
        "cfg": cfg,
        "args": vars(args) if hasattr(args, "__dict__") else args,
    }

    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()

    if teacher_model is not None:
        state["teacher_state"] = unwrap_model(teacher_model).state_dict()

    if extra:
        state["extra"] = extra

    return state


def save_checkpoint(
    *,
    save_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_miou: float,
    run_id: str,
    cfg: Dict[str, Any],
    args: Any,
    teacher_model: Optional[torch.nn.Module] = None,
    is_best: bool = False,
    save_latest: bool = True,
    save_best: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    save_dir = ensure_dir(save_path)
    paths: Dict[str, Path] = {}

    state = make_checkpoint_state(
        model=model,
        teacher_model=teacher_model,
        optimizer=optimizer,
        epoch=epoch,
        best_miou=best_miou,
        run_id=run_id,
        cfg=cfg,
        args=args,
        extra=extra,
    )

    if save_latest:
        latest = save_dir / "ckpt.pth"
        _atomic_torch_save(state, latest)
        paths["latest"] = latest

    if is_best and save_best:
        best = save_dir / "ckpt_best.pth"
        _atomic_torch_save(state, best)
        paths["best"] = best

    return paths


def _load_model_state(
    model: torch.nn.Module,
    state_dict: Dict[str, Any],
    *,
    strict: bool,
) -> None:
    model_to_load = unwrap_model(model)

    try:
        model_to_load.load_state_dict(state_dict, strict=strict)
        return
    except RuntimeError:
        pass

    if all(str(k).startswith("module.") for k in state_dict.keys()):
        stripped = {str(k)[7:]: v for k, v in state_dict.items()}
        model_to_load.load_state_dict(stripped, strict=strict)
        return

    wrapped_keys = {f"module.{k}": v for k, v in state_dict.items()}
    model.load_state_dict(wrapped_keys, strict=strict)


def load_checkpoint_if_available(
    *,
    save_path: str | Path,
    model: torch.nn.Module,
    teacher_model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    auto_resume: bool = True,
    filename: str = "ckpt.pth",
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Tuple[int, float, Optional[str], Optional[Path]]:
    """
    Returns:
        start_epoch, best_miou, run_id, ckpt_path

    start_epoch is the next epoch to run.
    """
    if not auto_resume:
        return 0, 0.0, None, None

    ckpt_path = Path(save_path) / filename
    if not ckpt_path.exists():
        return 0, 0.0, None, None

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    _load_model_state(model, ckpt["model_state"], strict=strict)

    if teacher_model is not None and "teacher_state" in ckpt:
        _load_model_state(teacher_model, ckpt["teacher_state"], strict=strict)

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    start_epoch = int(ckpt.get("epoch", 0))
    best_miou = float(ckpt.get("best_miou", 0.0))
    run_id = ckpt.get("run_id", None)

    return start_epoch, best_miou, run_id, ckpt_path


def copy_config_to_save_path(config_path: str | Path, save_path: str | Path) -> Optional[Path]:
    config_path = Path(config_path)
    if not config_path.exists():
        return None
    save_dir = ensure_dir(save_path)
    dst = save_dir / "config.yaml"
    shutil.copy2(config_path, dst)
    return dst
