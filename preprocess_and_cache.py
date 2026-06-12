"""
One-time preprocessing: load NIfTI -> clamp -> crop -> pad -> save as float16 .pt

Run this once on the machine that holds the raw data before training jobs.
Tensors are cached WITHOUT normalisation so one cache works for all normalisation
experiments; normalisation is applied on-the-fly in CachedTDMapDataset.

Also computes and saves both raw and log1p normalisation stats (training set only,
no data leakage) into preprocessing_stats.json.
log1p stats differ from raw stats because log1p compresses the right-skewed
distribution, giving a completely different mean and std.

Usage:
    python preprocess_and_cache.py
"""

import json
import logging
from pathlib import Path

import nibabel as nib
import torch
import torch.nn.functional as F

from config import Config
from utils import (
    CachedTDMapDataset,
    compute_normalisation_stats,
    load_preprocessing_stats,
    load_splits,
    regrid_map,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _cache_volume(
    sid: str,
    cohort: str,
    cfg: Config,
    bbox: dict,
    padded_shape: tuple,
    cache_dir: Path,
) -> bool:
    """Preprocess and cache one volume. Returns True if newly written."""
    out_path = cache_dir / f"{sid}.pt"
    if out_path.exists():
        return False

    try:
        nii_path = regrid_map(
            subject=sid,
            template=cfg.template,
            root=cfg.gbm_root,
            data_dir=cfg.cohort_dirs[cohort],
            tissue=cfg.tissue,
            tdmap=cfg.tdmap,
            voxel_res=cfg.voxel_res,
            nthreads=cfg.nthreads,
        )
    except FileNotFoundError:
        logger.warning("Missing map for %s — skipping", sid)
        return False

    vol = torch.from_numpy(nib.load(nii_path).get_fdata()).float()
    vol = torch.clamp(vol, min=0.0).unsqueeze(0)  # (1, D, H, W)

    b = bbox
    vol = vol[:, b["z_min"]:b["z_max"], b["y_min"]:b["y_max"], b["x_min"]:b["x_max"]]

    pd = padded_shape[0] - vol.shape[1]
    ph = padded_shape[1] - vol.shape[2]
    pw = padded_shape[2] - vol.shape[3]
    vol = F.pad(vol, (0, pw, 0, ph, 0, pd))

    # float16 halves disk usage (~3MB vs ~6MB per volume)
    # CachedTDMapDataset casts back to float32 on load (near-zero cost)
    torch.save(vol.half(), out_path)
    return True


def main() -> None:
    cfg = Config()

    stats_path   = Path(cfg.jsons_dir) / "preprocessing_stats.json"
    stats        = load_preprocessing_stats(str(stats_path))
    bbox         = stats["bbox"]
    padded_shape = stats["padded_shape"]

    splits    = load_splits(cfg.splits_dir)
    train_ids = splits["train"]
    all_ids   = train_ids + splits["val"] + splits["test"]

    with open(Path(cfg.jsons_dir) / "id_to_cohort.json") as f:
        id_to_cohort = json.load(f)

    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Cache directory: %s", cache_dir)

    # ------------------------------------------------------------------
    # Step 1: cache all volumes
    # ------------------------------------------------------------------
    newly_cached = 0
    for i, sid in enumerate(all_ids):
        cohort = id_to_cohort[sid]
        if _cache_volume(sid, cohort, cfg, bbox, padded_shape, cache_dir):
            newly_cached += 1
        if (i + 1) % 100 == 0 or i == 0:
            logger.info("%d/%d  (%d new, %d already cached)",
                        i + 1, len(all_ids), newly_cached, i + 1 - newly_cached)

    logger.info("Caching done: %d new, %d already existed.", newly_cached, len(all_ids) - newly_cached)

    t = torch.load(cache_dir / f"{all_ids[0]}.pt", weights_only=True)
    logger.info("Spot-check %s: shape=%s dtype=%s", all_ids[0], tuple(t.shape), t.dtype)

    # ------------------------------------------------------------------
    # Step 2: compute normalisation stats from training set only
    # Reuses compute_normalisation_stats from utils.py.
    # CachedTDMapDataset with normalisation="none" yields raw values;
    # with normalisation="log1p" it yields log1p-transformed values.
    # The resulting stats are stored under separate keys so train_sweep.py
    # can select the right ones per normalisation type.
    # ------------------------------------------------------------------
    logger.info("Computing raw stats (used for 'zscore') ...")
    raw_ds    = CachedTDMapDataset(train_ids, cache_dir=str(cache_dir), normalisation="none")
    raw_stats = compute_normalisation_stats(raw_ds)

    logger.info("Computing log1p stats (used for 'log1p_zscore') ...")
    log1p_ds    = CachedTDMapDataset(train_ids, cache_dir=str(cache_dir), normalisation="log1p")
    log1p_stats = compute_normalisation_stats(log1p_ds)

    stats["norm_mean"]       = raw_stats["mean"]
    stats["norm_std"]        = raw_stats["std"]
    stats["log1p_norm_mean"] = log1p_stats["mean"]
    stats["log1p_norm_std"]  = log1p_stats["std"]
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info(
        "Saved to %s — raw: mean=%.4f std=%.4f | log1p: mean=%.4f std=%.4f",
        stats_path,
        raw_stats["mean"], raw_stats["std"],
        log1p_stats["mean"], log1p_stats["std"],
    )


if __name__ == "__main__":
    main()
