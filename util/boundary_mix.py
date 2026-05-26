import numpy as np
import torch
import torch.nn.functional as F


def _as_bchw_mask(mask):
    if mask.dim() == 3:
        return mask.unsqueeze(1).float()
    if mask.dim() == 4 and mask.size(1) == 1:
        return mask.float()
    raise ValueError("mask must have shape [B,H,W] or [B,1,H,W]")


def erode_mask(mask, kernel_size=5):
    mask = _as_bchw_mask(mask)
    pad = kernel_size // 2
    return -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)


def dilate_mask(mask, kernel_size=5):
    mask = _as_bchw_mask(mask)
    pad = kernel_size // 2
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)


def get_inner_outer_boundary(mask, kernel_size=5):
    mask = _as_bchw_mask(mask).clamp(0.0, 1.0)
    eroded = erode_mask(mask, kernel_size=kernel_size).clamp(0.0, 1.0)
    dilated = dilate_mask(mask, kernel_size=kernel_size).clamp(0.0, 1.0)
    inner = (mask - eroded).clamp(0.0, 1.0)
    outer = (dilated - mask).clamp(0.0, 1.0)
    return inner, outer


def boundary_confidence_weight_map(
    mix_mask,
    confidence,
    *,
    kernel_size=5,
    gamma_in=0.7,
    gamma_out=0.3,
    use_confidence=True,
    normalize_weight=True,
    eps=1e-6,
):
    mask = _as_bchw_mask(mix_mask).clamp(0.0, 1.0)
    inner, outer = get_inner_outer_boundary(mask, kernel_size=kernel_size)

    if use_confidence:
        risk = _as_bchw_mask(confidence).to(device=mask.device, dtype=mask.dtype)
    else:
        risk = torch.ones_like(mask)

    source_interior = (mask - inner).clamp(0.0, 1.0)
    target = 1.0 - mask
    target_exterior = (target - outer).clamp(0.0, 1.0)

    weight = (
        source_interior
        + float(gamma_in) * inner
        + risk * target_exterior
        + float(gamma_out) * risk * outer
    )

    if normalize_weight:
        valid = weight > 0
        mean_weight = weight[valid].mean().clamp_min(eps) if valid.any() else weight.new_tensor(1.0)
        weight = weight / mean_weight

    return weight.squeeze(1)


def _masked_mean(values, mask):
    mask = mask.to(dtype=torch.bool)
    if not mask.any():
        return float("nan")
    return float(values[mask].detach().mean().item())


def boundary_mix_debug_tensors(
    mix_mask,
    confidence,
    *,
    kernel_size=5,
    gamma_in=0.7,
    gamma_out=0.3,
    use_confidence=True,
    normalize_weight=True,
):
    mask = _as_bchw_mask(mix_mask).clamp(0.0, 1.0)
    confidence = _as_bchw_mask(confidence).to(device=mask.device, dtype=mask.dtype)
    inner, outer = get_inner_outer_boundary(mask, kernel_size=kernel_size)
    weight = boundary_confidence_weight_map(
        mask,
        confidence,
        kernel_size=kernel_size,
        gamma_in=gamma_in,
        gamma_out=gamma_out,
        use_confidence=use_confidence,
        normalize_weight=normalize_weight,
    ).unsqueeze(1)
    target_exterior = ((1.0 - mask) - outer).clamp(0.0, 1.0)

    return {
        "mask": mask,
        "inner": inner,
        "outer": outer,
        "confidence": confidence,
        "weight": weight,
        "target_exterior": target_exterior,
    }


def boundary_mix_debug_stats(
    mix_mask,
    confidence,
    *,
    kernel_size=5,
    gamma_in=0.7,
    gamma_out=0.3,
    use_confidence=True,
    normalize_weight=True,
):
    tensors = boundary_mix_debug_tensors(
        mix_mask,
        confidence,
        kernel_size=kernel_size,
        gamma_in=gamma_in,
        gamma_out=gamma_out,
        use_confidence=use_confidence,
        normalize_weight=normalize_weight,
    )
    weight = tensors["weight"]
    inner = tensors["inner"]
    outer = tensors["outer"]
    confidence = tensors["confidence"]
    target_exterior = tensors["target_exterior"]

    return {
        "weight_mean": float(weight.detach().mean().item()),
        "weight_min": float(weight.detach().min().item()),
        "weight_max": float(weight.detach().max().item()),
        "b_in_pixels": int(inner.detach().sum().item()),
        "b_out_pixels": int(outer.detach().sum().item()),
        "target_exterior_conf_mean": _masked_mean(confidence, target_exterior > 0),
        "target_outer_boundary_conf_mean": _masked_mean(confidence, outer > 0),
    }


def weighted_cross_entropy_loss(
    predict,
    target,
    weight,
    *,
    ignore_index=255,
    eps=1e-6,
):
    ce = F.cross_entropy(predict, target, ignore_index=ignore_index, reduction="none")
    valid = (target != ignore_index).to(dtype=ce.dtype)
    weight = weight.to(device=ce.device, dtype=ce.dtype) * valid
    return (ce * weight).sum() / (weight.sum() + eps)


def cut_mix_label_adaptive_with_mask(
    unlabeled_image,
    unlabeled_mask,
    unlabeled_logits,
    labeled_image,
    labeled_mask,
    lst_confidences,
):
    assert len(lst_confidences) == len(unlabeled_image), "Ensure the confidence is properly obtained"
    assert labeled_image.shape == unlabeled_image.shape, "Ensure shape match between lb and unlb"
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    mix_source_mask = torch.zeros_like(unlabeled_mask, dtype=torch.float32)
    source_mask = torch.zeros_like(unlabeled_mask, dtype=torch.float32)
    labeled_logits = torch.ones_like(labeled_mask)

    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

    l_bbx1, l_bby1, l_bbx2, l_bby2 = _rand_bbox(unlabeled_image.size(), lam=np.random.beta(8, 2))
    u_bbx1, u_bby1, u_bbx2, u_bby2 = _rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))

    for i in range(0, mix_unlabeled_image.shape[0]):
        if np.random.random() > lst_confidences[i]:
            mix_unlabeled_image[i, :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = (
                labeled_image[u_rand_index[i], :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
            )

            mix_unlabeled_target[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = (
                labeled_mask[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
            )

            mix_unlabeled_logits[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = (
                labeled_logits[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
            )

            mix_source_mask[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = 1.0

    for i in range(0, unlabeled_image.shape[0]):
        unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = (
            mix_unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        )

        unlabeled_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = (
            mix_unlabeled_target[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        )

        unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = (
            mix_unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        )

        source_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = (
            mix_source_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        )

    return unlabeled_image, unlabeled_mask, unlabeled_logits, source_mask


def thresholded_boundary_mix_loss(
    predict,
    target,
    confidence,
    mix_mask,
    *,
    thresh=0.95,
    ignore_index=255,
    kernel_size=5,
    gamma_in=0.7,
    gamma_out=0.3,
    use_confidence=True,
    normalize_weight=True,
):
    valid = confidence.ge(thresh).bool() & target.ne(ignore_index).bool()
    target = target.clone()
    target[~valid] = ignore_index
    weight = boundary_confidence_weight_map(
        mix_mask,
        confidence,
        kernel_size=kernel_size,
        gamma_in=gamma_in,
        gamma_out=gamma_out,
        use_confidence=use_confidence,
        normalize_weight=normalize_weight,
    )
    loss = weighted_cross_entropy_loss(
        predict,
        target,
        weight,
        ignore_index=ignore_index,
    )
    return loss, valid.float().mean()


def _rand_bbox(size, lam=None):
    if len(size) == 4:
        width = size[2]
        height = size[3]
    elif len(size) == 3:
        width = size[1]
        height = size[2]
    else:
        raise ValueError("size must have 3 or 4 dimensions")
    batch = size[0]

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)

    cx = np.random.randint(size=[batch], low=int(width / 8), high=width)
    cy = np.random.randint(size=[batch], low=int(height / 8), high=height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2
