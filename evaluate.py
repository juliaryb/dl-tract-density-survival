"""Shared inference and metrics utilities for AE reconstruction evaluation."""
import logging
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from scipy.stats import sem
from config import Config
from model import Autoencoder
from utils import load_brain_mask

logger = logging.getLogger(__name__)


def _apply_norm(vol: torch.Tensor, normalisation: str, mean, std, vmin=None, vmax=None) -> torch.Tensor:
    if normalisation in ("log1p", "log1p_zscore"):
        vol = torch.log1p(vol)
    if normalisation in ("zscore", "log1p_zscore"):
        vol = (vol - mean) / (std + 1e-8)
    if normalisation == "minmax":
        vol = (vol - vmin) / (vmax - vmin + 1e-8)
    return vol


def _denormalize(recon: torch.Tensor, normalisation: str, mean, std, vmin=None, vmax=None) -> torch.Tensor:
    if normalisation == "zscore":
        return recon * std + mean
    if normalisation == "log1p":
        return torch.expm1(recon)
    if normalisation == "log1p_zscore":
        return torch.expm1(recon * std + mean)
    if normalisation == "minmax":
        return recon * (vmax - vmin) + vmin
    return recon  # "none"


# Derived from the training split inside the brain mask; the fallback is 
# the observed global maximum across the full dataset.
DEFAULT_DATA_RANGE = 927.4595


def _pointwise(inp: np.ndarray, rec: np.ndarray) -> dict:
    """R², MAE and MSE over a flat pair of arrays.
    """
    ss_res = ((inp - rec) ** 2).sum()
    ss_tot = ((inp - inp.mean()) ** 2).sum()
    return {
        "r2": float(1.0 - ss_res / (ss_tot + 1e-8)),
        "mae": float(np.abs(inp - rec).mean()),
        "mse": float(((inp - rec) ** 2).mean()),
    }


def subject_metrics(
    raw_in: torch.Tensor,
    raw_recon: torch.Tensor,
    data_range: float = DEFAULT_DATA_RANGE,
    mask: torch.Tensor | None = None,
) -> dict:
    """Reconstruction metrics for one volume pair.

    Returns R², MAE, MSE and SSIM over the whole volume and, when a
    brain mask is supplied, the same metrics restricted to the masked region of
    interest (suffix "_roi").

    SSIM uses the parameterisation of Wang et al. (2004) - Gaussian weighting with
    sigma=1.5 and population covariance - rather than the scikit-image defaults
    (uniform window, sample covariance), so that the reported values match the
    conventional definition.

    Because SSIM combines information across neighbouring voxels, a non-rectangular
    mask cannot be applied to the inputs; the full SSIM map is computed instead and
    then averaged over in-mask voxels only.
    """
    inp3d = raw_in.squeeze(0).numpy()
    rec3d = raw_recon.squeeze(0).numpy()

    _, ssim_map = structural_similarity(
        inp3d, rec3d,
        data_range=data_range,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        full=True,
    )

    out = _pointwise(inp3d.ravel(), rec3d.ravel())
    out["ssim"] = float(ssim_map.mean())

    if mask is not None:
        m = mask.squeeze(0).numpy().astype(bool)
        out.update({f"{k}_roi": v for k, v in _pointwise(inp3d[m], rec3d[m]).items()})
        out["ssim_roi"] = float(ssim_map[m].mean())

    return out


def evaluate_model(cfg: Config, raw_dataset, stats: dict, device: torch.device) -> list[dict] | None:
    """Load a checkpoint and return per-subject metric dicts, or None if missing."""
    ckpt = Path(cfg.checkpoints_dir) / cfg.model_name
    if not ckpt.exists():
        logger.warning("Checkpoint not found: %s — skipping", ckpt)
        return None

    model = Autoencoder(stats["padded_shape"], cfg.latent_dim).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    norm_mean = norm_std = norm_min = norm_max = None
    if cfg.normalisation == "zscore":
        norm_mean, norm_std = stats["norm_mean"], stats["norm_std"]
    elif cfg.normalisation == "log1p_zscore":
        norm_mean, norm_std = stats["log1p_norm_mean"], stats["log1p_norm_std"]
    elif cfg.normalisation == "minmax":
        norm_min, norm_max = stats["norm_min"], stats["norm_max"]

    # Brain mask for the ROI-restricted metrics, cropped/padded to match volumes.
    brain_mask = load_brain_mask(cfg.brain_mask, stats["bbox"], stats["padded_shape"])

    # Fixed across all subjects and models - see DEFAULT_DATA_RANGE.
    data_range = float(stats.get("norm_max") or DEFAULT_DATA_RANGE)
    logger.info("SSIM data_range=%.4f (fixed across subjects)", data_range)

    per_subject = []
    with torch.no_grad():
        for i in range(len(raw_dataset)):
            raw_vol = raw_dataset[i]
            norm_vol = _apply_norm(raw_vol.clone(), cfg.normalisation, norm_mean, norm_std, norm_min, norm_max)
            recon_norm = model(norm_vol.unsqueeze(0).to(device))[0].squeeze(0).cpu()
            recon_raw = torch.clamp(
                _denormalize(recon_norm, cfg.normalisation, norm_mean, norm_std, norm_min, norm_max), min=0.0
            )
            per_subject.append(
                subject_metrics(raw_vol, recon_raw, data_range=data_range, mask=brain_mask)
            )

    return per_subject


def summarize(per_subject: list[dict]) -> dict:
    """Aggregate per-subject metric dicts into mean ± standard error.

    Covers whatever metrics are present, including the "_roi" variants.
    """
    metrics = per_subject[0].keys() if per_subject else ()
    return {
        metric: {
            "mean": float(np.mean([s[metric] for s in per_subject])),
            "std_err":  float(sem( [s[metric] for s in per_subject])),
        }
        for metric in metrics
    }
