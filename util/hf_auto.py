from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def _warn(msg: str) -> None:
    print(f"[HF_AUTO][WARN] {msg}", flush=True)


def _info(msg: str) -> None:
    print(f"[HF_AUTO] {msg}", flush=True)


def get_hf_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg.get("hf", {}) or {})


def hf_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(get_hf_cfg(cfg).get("enabled", False))


def _try_import_hf():
    try:
        from huggingface_hub import HfApi, hf_hub_download
        return HfApi, hf_hub_download
    except Exception as e:
        _warn(f"huggingface_hub is not available: {repr(e)}")
        return None, None


def _get_token(hf_cfg: Dict[str, Any]) -> Optional[str]:
    return (
        hf_cfg.get("token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _default_bundle_name(save_path: str | Path) -> str:
    return "latest.tar.gz"


def _default_path_in_repo(hf_cfg: Dict[str, Any], save_path: str | Path) -> str:
    bundle_name = hf_cfg.get("bundle_name") or _default_bundle_name(save_path)
    if bool(hf_cfg.get("keep_only_latest", True)):
        return hf_cfg.get("path_in_repo") or bundle_name
    return hf_cfg.get("path_in_repo") or bundle_name


def _add_if_exists(tar: tarfile.TarFile, path: Path, arcname: Optional[str] = None) -> None:
    if path.exists():
        tar.add(path, arcname=arcname or path.name)


def create_hf_bundle(
    *,
    cfg: Dict[str, Any],
    save_path: str | Path,
    config_path: str | Path,
    manifest: Dict[str, Any],
    bundle_path: str | Path,
) -> Path:
    save_dir = Path(save_path)
    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        manifest_path = tmp / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))

        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(manifest_path, arcname="manifest.json")

            config_path = Path(config_path)
            if config_path.exists():
                tar.add(config_path, arcname="config.yaml")

            for name in [
                "run_id.txt",
                "ckpt.pth",
                "ckpt_best.pth",
                "epoch_metrics.csv",
                "iter_metrics.csv",
                "manifest.json",
            ]:
                _add_if_exists(tar, save_dir / name, arcname=name)

            # Add text logs in save_path root.
            for txt in sorted(save_dir.glob("*.txt")):
                _add_if_exists(tar, txt, arcname=txt.name)

    return bundle_path


def maybe_upload_hf_bundle(
    *,
    cfg: Dict[str, Any],
    save_path: str | Path,
    config_path: str | Path,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    hf_cfg = get_hf_cfg(cfg)
    if not bool(hf_cfg.get("enabled", False)):
        return {"enabled": False, "uploaded": False, "reason": "hf.disabled"}

    if not bool(hf_cfg.get("auto_upload", True)):
        return {"enabled": True, "uploaded": False, "reason": "auto_upload.disabled"}

    repo_id = hf_cfg.get("repo_id")
    repo_type = hf_cfg.get("repo_type", "model")
    if not repo_id:
        _warn("hf.repo_id is missing; skip upload")
        return {"enabled": True, "uploaded": False, "reason": "missing.repo_id"}

    token = _get_token(hf_cfg)
    if not token:
        _warn("HF token not found; set HF_TOKEN to enable upload")
        return {"enabled": True, "uploaded": False, "reason": "missing.token"}

    HfApi, _ = _try_import_hf()
    if HfApi is None:
        return {"enabled": True, "uploaded": False, "reason": "missing.huggingface_hub"}

    save_dir = Path(save_path)
    bundle_name = hf_cfg.get("bundle_name") or _default_bundle_name(save_dir)
    path_in_repo = _default_path_in_repo(hf_cfg, save_dir)
    local_bundle = save_dir / "_hf_bundle" / bundle_name

    try:
        create_hf_bundle(
            cfg=cfg,
            save_path=save_dir,
            config_path=config_path,
            manifest=manifest,
            bundle_path=local_bundle,
        )

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(local_bundle),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
        )
        _info(f"Uploaded bundle to HF: {repo_id}/{path_in_repo}")

        if bool(hf_cfg.get("squash_after_upload", False)):
            try:
                api.super_squash_history(repo_id=repo_id, repo_type=repo_type)
                _info(f"Squashed HF repo history: {repo_id}")
            except Exception as e:
                _warn(f"HF history squash failed: {repr(e)}")

        return {
            "enabled": True,
            "uploaded": True,
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
            "local_bundle": str(local_bundle),
        }
    except Exception as e:
        _warn(f"HF upload failed: {repr(e)}")
        return {"enabled": True, "uploaded": False, "reason": repr(e)}


def maybe_download_hf_bundle(
    *,
    cfg: Dict[str, Any],
    save_path: str | Path,
    skip_if_ckpt_exists: bool = True,
) -> Dict[str, Any]:
    hf_cfg = get_hf_cfg(cfg)
    if not bool(hf_cfg.get("enabled", False)):
        return {"enabled": False, "downloaded": False, "reason": "hf.disabled"}

    if not bool(hf_cfg.get("auto_download", True)):
        return {"enabled": True, "downloaded": False, "reason": "auto_download.disabled"}

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    if skip_if_ckpt_exists and (save_dir / "ckpt.pth").exists():
        return {"enabled": True, "downloaded": False, "reason": "local.ckpt.exists"}

    repo_id = hf_cfg.get("repo_id")
    repo_type = hf_cfg.get("repo_type", "model")
    if not repo_id:
        _warn("hf.repo_id is missing; skip download")
        return {"enabled": True, "downloaded": False, "reason": "missing.repo_id"}

    token = _get_token(hf_cfg)
    if not token:
        _warn("HF token not found; set HF_TOKEN to enable download")
        return {"enabled": True, "downloaded": False, "reason": "missing.token"}

    _, hf_hub_download = _try_import_hf()
    if hf_hub_download is None:
        return {"enabled": True, "downloaded": False, "reason": "missing.huggingface_hub"}

    path_in_repo = _default_path_in_repo(hf_cfg, save_dir)

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type=repo_type,
            token=token,
        )

        with tarfile.open(downloaded, "r:gz") as tar:
            tar.extractall(save_dir)

        _info(f"Downloaded and extracted HF bundle: {repo_id}/{path_in_repo}")
        return {
            "enabled": True,
            "downloaded": True,
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
            "downloaded_path": str(downloaded),
        }
    except Exception as e:
        _warn(f"HF download failed: {repr(e)}")
        return {"enabled": True, "downloaded": False, "reason": repr(e)}
