#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"

cd "${ROOT}"

"${PYTHON_BIN}" -m py_compile train_semi.py util/*.py tools/*.py

scripts=(
  tools/run_boundary_mix_v1_a0.sh
  tools/run_boundary_mix_v1_a1.sh
  tools/run_boundary_mix_v1_a2.sh
  tools/run_boundary_mix_v1_a3.sh
  tools/run_boundary_mix_v1_a4.sh
  tools/run_boundary_mix_v1_a5.sh
)

for script in "${scripts[@]}"; do
  bash -n "${script}"
done

"${PYTHON_BIN}" - <<'PYTABLE'
from pathlib import Path
import yaml

root = Path.cwd().resolve()
expected_hf = {
    "enabled": True,
    "repo_id": "tanprodium/augseg-checkpoints",
    "repo_type": "model",
    "auto_download": True,
    "auto_upload": True,
    "upload_every_epoch": True,
    "keep_only_latest": True,
    "bundle_name": "latest.tar.gz",
    "squash_after_upload": False,
}
rows = [
    {
        "slug": "a0_baseline",
        "run_name": "boundary_mix_v1_a0_baseline",
        "enabled": False,
        "gamma_in": None,
        "gamma_out": None,
        "use_confidence": None,
    },
    {
        "slug": "a1_neutral_control",
        "run_name": "boundary_mix_v1_a1_neutral_control",
        "enabled": True,
        "gamma_in": 1.0,
        "gamma_out": 1.0,
        "use_confidence": False,
    },
    {
        "slug": "a2_boundary_ignore",
        "run_name": "boundary_mix_v1_a2_boundary_ignore",
        "enabled": True,
        "gamma_in": 0.0,
        "gamma_out": 0.0,
        "use_confidence": False,
    },
    {
        "slug": "a3_boundary_reweight_common",
        "run_name": "boundary_mix_v1_a3_boundary_reweight_common",
        "enabled": True,
        "gamma_in": 0.5,
        "gamma_out": 0.5,
        "use_confidence": False,
    },
    {
        "slug": "a4_inner_outer_reweight",
        "run_name": "boundary_mix_v1_a4_inner_outer_reweight",
        "enabled": True,
        "gamma_in": 0.7,
        "gamma_out": 0.3,
        "use_confidence": False,
    },
    {
        "slug": "a5_boundary_confidence",
        "run_name": "boundary_mix_v1_a5_boundary_confidence",
        "enabled": True,
        "gamma_in": 0.7,
        "gamma_out": 0.3,
        "use_confidence": True,
    },
]

def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)

def require(label, actual, expected):
    if actual != expected:
        raise SystemExit(f"{label}: expected {expected!r}, got {actual!r}")

print(
    f"{'experiment':42} {'enabled':7} {'gamma_in':8} {'gamma_out':9} "
    f"{'conf':5} {'save_path':42} {'hf.path_in_repo'}"
)
for row in rows:
    slug = row["slug"]
    path = root / "exps/boundary_mix_v1/voc_semi662" / slug / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bm = cfg.get("boundary_mix", {})
    hf = cfg.get("hf", {})
    run_name = cfg.get("run", {}).get("name", path.parent.name)
    snapshot_dir = cfg.get("saver", {}).get("snapshot_dir", "")
    save_path = (path.parent / snapshot_dir).resolve()
    save_display = save_path.relative_to(root)
    hf_path = hf.get("path_in_repo", "")

    require(f"{slug}.run.name", run_name, row["run_name"])
    require(f"{slug}.boundary_mix.enabled", bm.get("enabled"), row["enabled"])
    require(f"{slug}.boundary_mix.debug", bm.get("debug"), False)
    require(f"{slug}.boundary_mix.vis_debug", bm.get("vis_debug"), False)
    require(f"{slug}.save_path", str(save_display), f"exp_boundary_mix_v1/{slug}")
    require(f"{slug}.hf.path_in_repo", hf_path, f"boundary_mix_v1/voc_semi662/{slug}/latest.tar.gz")
    for key, expected in expected_hf.items():
        require(f"{slug}.hf.{key}", hf.get(key), expected)

    if row["enabled"]:
        require(f"{slug}.boundary_mix.gamma_in", bm.get("gamma_in"), row["gamma_in"])
        require(f"{slug}.boundary_mix.gamma_out", bm.get("gamma_out"), row["gamma_out"])
        require(f"{slug}.boundary_mix.use_confidence", bm.get("use_confidence"), row["use_confidence"])
    else:
        for inactive_key in ("gamma_in", "gamma_out", "use_confidence"):
            if inactive_key in bm:
                raise SystemExit(f"{slug}.boundary_mix.{inactive_key}: should be unset when BoundaryMix is disabled")

    print(
        f"{run_name:42} {fmt(bm.get('enabled')):7} {fmt(bm.get('gamma_in')):8} "
        f"{fmt(bm.get('gamma_out')):9} {fmt(bm.get('use_confidence')):5} "
        f"{str(save_display):42} {hf_path}"
    )
PYTABLE
