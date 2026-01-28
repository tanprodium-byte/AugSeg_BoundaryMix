import argparse, os, shutil
from pathlib import Path

def norm_id(s: str) -> str:
    s = s.strip()
    s = os.path.basename(s)
    s = os.path.splitext(s)[0]
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_root", required=True)
    ap.add_argument("--sbd_root", required=True, help=".../benchmark_RELEASE/dataset")
    ap.add_argument("--lists", nargs="+", required=True, help="labeled.txt unlabeled.txt ...")
    args = ap.parse_args()

    voc_root = Path(args.voc_root)
    sbd_root = Path(args.sbd_root)
    out_dir = voc_root / "SegmentationClassAug"
    out_dir.mkdir(parents=True, exist_ok=True)

    # lazy import: chỉ cần khi phải đọc .mat
    from PIL import Image
    import numpy as np
    from scipy.io import loadmat

    ids = []
    for lf in args.lists:
        for line in Path(lf).read_text().splitlines():
            line = line.strip()
            if line:
                ids.append(norm_id(line))
    ids = sorted(set(ids))

    voc_dir = voc_root / "SegmentationClass"
    sbd_cls_dir = sbd_root / "cls"

    n_voc = n_sbd = n_missing = 0
    for img_id in ids:
        dst = out_dir / f"{img_id}.png"
        src_voc = voc_dir / f"{img_id}.png"
        if src_voc.exists():
            shutil.copyfile(src_voc, dst)
            n_voc += 1
            continue

        src_mat = sbd_cls_dir / f"{img_id}.mat"
        if src_mat.exists():
            mat = loadmat(src_mat)
            arr = mat["GTcls"][0]["Segmentation"][0]
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)
            Image.fromarray(arr).save(dst)
            n_sbd += 1
        else:
            n_missing += 1
            print(f"[MISSING] no VOC png, no SBD mat for id={img_id}")

    print(f"Total ids={len(ids)} | from VOC png={n_voc} | from SBD mat={n_sbd} | missing={n_missing}")
    print(f"Output: {out_dir}")

if __name__ == "__main__":
    main()
