import argparse
import os
from pathlib import Path

import numpy as np
import torch

from util.boundary_mix import boundary_mix_debug_tensors


def _to_numpy_2d(tensor):
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def _save_gray(path, tensor):
    from PIL import Image

    arr = _to_numpy_2d(tensor)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if arr.size and arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _save_image(path, image, mean=None, std=None):
    from PIL import Image

    img = image.detach().float().cpu()
    if img.dim() == 4:
        img = img[0]
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=img.dtype).view(-1, 1, 1)
        std_t = torch.tensor(std, dtype=img.dtype).view(-1, 1, 1)
        img = img * std_t + mean_t
    img = img.clamp(0.0, 1.0)
    arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _save_label(path, label):
    from PIL import Image

    arr = label.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.astype(np.uint8, copy=False)
    Image.fromarray(arr, mode="L").save(path)


def save_boundary_mix_debug(
    out_dir,
    *,
    mix_mask,
    confidence,
    mixed_image=None,
    mixed_label=None,
    prediction=None,
    mean=None,
    std=None,
    prefix="boundary_mix",
    kernel_size=5,
    gamma_in=0.7,
    gamma_out=0.3,
    use_confidence=True,
    normalize_weight=True,
    max_items=1,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tensors = boundary_mix_debug_tensors(
        mix_mask,
        confidence,
        kernel_size=kernel_size,
        gamma_in=gamma_in,
        gamma_out=gamma_out,
        use_confidence=use_confidence,
        normalize_weight=normalize_weight,
    )

    batch = int(tensors["mask"].shape[0])
    n_items = min(batch, int(max_items))
    saved = []
    for idx in range(n_items):
        stem = f"{prefix}_b{idx:02d}"
        files = {
            "mask": out_path / f"{stem}_mask_M.png",
            "b_in": out_path / f"{stem}_B_in.png",
            "b_out": out_path / f"{stem}_B_out.png",
            "confidence": out_path / f"{stem}_confidence_R.png",
            "weight": out_path / f"{stem}_weight_W.png",
        }
        _save_gray(files["mask"], tensors["mask"][idx])
        _save_gray(files["b_in"], tensors["inner"][idx])
        _save_gray(files["b_out"], tensors["outer"][idx])
        _save_gray(files["confidence"], tensors["confidence"][idx])
        _save_gray(files["weight"], tensors["weight"][idx])
        if mixed_image is not None:
            files["mixed_image"] = out_path / f"{stem}_mixed_image.png"
            _save_image(files["mixed_image"], mixed_image[idx], mean=mean, std=std)
        if mixed_label is not None:
            files["mixed_label"] = out_path / f"{stem}_mixed_label.png"
            _save_label(files["mixed_label"], mixed_label[idx])
        if prediction is not None:
            files["prediction"] = out_path / f"{stem}_prediction.png"
            pred = prediction[idx]
            if pred.dim() == 3:
                pred = pred.argmax(dim=0)
            _save_label(files["prediction"], pred)
        saved.extend(str(path) for path in files.values())
    return saved


def main():
    parser = argparse.ArgumentParser(description="Write synthetic BoundaryMix debug PNGs.")
    parser.add_argument("--out-dir", default="exp_boundary_debug_vis")
    parser.add_argument("--height", type=int, default=129)
    parser.add_argument("--width", type=int, default=129)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--gamma-in", type=float, default=0.7)
    parser.add_argument("--gamma-out", type=float, default=0.3)
    parser.add_argument("--no-confidence", action="store_true")
    args = parser.parse_args()

    h, w = args.height, args.width
    mask = torch.zeros(1, h, w)
    mask[:, h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 1.0
    yy = torch.linspace(0, 1, h).view(1, h, 1)
    xx = torch.linspace(0, 1, w).view(1, 1, w)
    confidence = (0.25 + 0.75 * (1.0 - (xx - yy).abs())).clamp(0.0, 1.0)

    files = save_boundary_mix_debug(
        args.out_dir,
        mix_mask=mask,
        confidence=confidence,
        prefix="synthetic",
        kernel_size=args.kernel_size,
        gamma_in=args.gamma_in,
        gamma_out=args.gamma_out,
        use_confidence=not args.no_confidence,
    )
    print(f"Saved {len(files)} files to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
