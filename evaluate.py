"""Shared inference and metrics utilities for AE reconstruction evaluation."""
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

from config import Config
from model import Autoencoder

logger = logging.getLogger(__name__)


def _apply_norm(vol: torch.Tensor, normalisation: str, mean, std) -> torch.Tensor:
    if normalisation in ("log1p", "log1p_zscore"):
        vol = torch.log1p(vol)
    if normalisation in ("zscore", "log1p_zscore"):
        vol = (vol - mean) / (std + 1e-8)
    return vol


def _denormalize(recon: torch.Tensor, normalisation: str, mean, std) -> torch.Tensor:
    if normalisation == "zscore":
        return recon * std + mean
    if normalisation == "log1p":
        return torch.expm1(recon)
    if normalisation == "log1p_zscore":
        return torch.expm1(recon * std + mean)
    return recon  # "none"


def subject_metrics(raw_in: torch.Tensor, raw_recon: torch.Tensor) -> dict:
    """R², MAE, Pearson r, SSIM over all voxels for one volume pair."""
    inp = raw_in.flatten().numpy()
    rec = raw_recon.flatten().numpy()

    ss_res = ((inp - rec) ** 2).sum()
    ss_tot = ((inp - inp.mean()) ** 2).sum()
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
    mae = float(np.abs(inp - rec).mean())
    mse = float(((inp - rec) ** 2).mean())
    r = float(pearsonr(inp, rec)[0])

    data_range = float(raw_in.max())
    ssim = float(structural_similarity(
        raw_in.squeeze(0).numpy(),
        raw_recon.squeeze(0).numpy(),
        data_range=data_range if data_range > 0 else 1.0,
    ))

    return {"r2": r2, "mae": mae, "mse": mse, "pearson_r": r, "ssim": ssim}


def evaluate_model(cfg: Config, raw_dataset, stats: dict, device: torch.device) -> list[dict] | None:
    """Load a checkpoint and return per-subject metric dicts, or None if missing."""
    ckpt = Path(cfg.checkpoints_dir) / cfg.model_name
    if not ckpt.exists():
        logger.warning("Checkpoint not found: %s — skipping", ckpt)
        return None

    model = Autoencoder(stats["padded_shape"], cfg.latent_dim).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    norm_mean = norm_std = None
    if cfg.normalisation == "zscore":
        norm_mean, norm_std = stats["norm_mean"], stats["norm_std"]
    elif cfg.normalisation == "log1p_zscore":
        norm_mean, norm_std = stats["log1p_norm_mean"], stats["log1p_norm_std"]

    per_subject = []
    with torch.no_grad():
        for i in range(len(raw_dataset)):
            raw_vol = raw_dataset[i]
            norm_vol = _apply_norm(raw_vol.clone(), cfg.normalisation, norm_mean, norm_std)
            recon_norm = model(norm_vol.unsqueeze(0).to(device))[0].squeeze(0).cpu()
            recon_raw = torch.clamp(
                _denormalize(recon_norm, cfg.normalisation, norm_mean, norm_std), min=0.0
            )
            per_subject.append(subject_metrics(raw_vol, recon_raw))

    return per_subject


def summarize(per_subject: list[dict]) -> dict:
    """Aggregate per-subject metric dicts into mean ± std."""
    return {
        metric: {
            "mean": float(np.mean([s[metric] for s in per_subject])),
            "std":  float(np.std( [s[metric] for s in per_subject])),
        }
        for metric in ("r2", "mae", "mse", "pearson_r", "ssim")
    }
