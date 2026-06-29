"""
Cox proportional hazards analysis on AE latent codes across latent dimensions.

For each latent dim:
  1. Encode all subjects with the trained AE (cached to disk on first run).
  2. Fit three Cox models on train+val subjects.
  3. Evaluate concordance (C-index) on held-out test subjects.

Models:
  clinical          — age, sex
  latent            — z1 ... zN
  clinical+latent   — age, sex, z1 ... zN
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter

from config import Config
from model import Autoencoder
from train import encode_dataset
from utils import (
    build_dataset_from_ids, load_preprocessing_stats, load_splits, unpad_ucsf_ids,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LATENT_DIMS      = [2, 4, 6, 8, 12, 16, 32, 64, 128]
DURATION_COL     = "OS (days) - corrected"
EVENT_COL        = "status"
CLINICAL_COLS    = ["age", "sex"]


def _get_or_encode(cfg: Config, all_ids: list[str], id_to_cohort: dict,
                   stats: dict, device: torch.device) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """Return (codes, ids) for all subjects, encoding and caching if needed."""
    codes_dir = Path(cfg.latents_dir) / cfg.run_tag
    codes_path = codes_dir / "codes.npy"
    ids_path   = codes_dir / "ids.npy"

    if codes_path.exists() and ids_path.exists():
        logger.info("Loading cached codes from %s", codes_dir)
        return np.load(codes_path), np.load(ids_path, allow_pickle=True).tolist()

    ckpt = Path(cfg.checkpoints_dir) / cfg.model_name
    if not ckpt.exists():
        logger.warning("Checkpoint not found: %s — skipping", ckpt)
        return None, None

    norm_mean = norm_std = None
    if cfg.normalisation == "zscore":
        norm_mean, norm_std = stats["norm_mean"], stats["norm_std"]

    cache_dir = cfg.cache_dir if Path(cfg.cache_dir).exists() else None
    dataset = build_dataset_from_ids(
        all_ids, id_to_cohort, cfg,
        bbox=stats["bbox"], padded_shape=stats["padded_shape"],
        norm_mean=norm_mean, norm_std=norm_std,
        normalisation=cfg.normalisation,
        cache_dir=cache_dir,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    model = Autoencoder(stats["padded_shape"], cfg.latent_dim).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    codes = encode_dataset(model, loader, device).numpy()

    codes_dir.mkdir(parents=True, exist_ok=True)
    np.save(codes_path, codes)
    np.save(ids_path, np.array(all_ids))
    logger.info("Encoded %s -> %s", codes.shape, codes_dir)

    return codes, all_ids


def _build_df(codes: np.ndarray, ids: list[str],
              clinical: pd.DataFrame, latent_dim: int) -> tuple[pd.DataFrame, list[str]]:
    """Merge latent codes with clinical data. Returns (merged_df, z_col_names)."""
    z_cols = [f"z{i+1}" for i in range(latent_dim)]
    codes_df = pd.DataFrame(codes, columns=z_cols)
    codes_df["id"] = [unpad_ucsf_ids(sid) if "UCSF" in sid else sid for sid in ids]
    df = codes_df.merge(clinical, left_on="id", right_on="ID", how="inner")
    df = df.dropna(subset=[DURATION_COL, EVENT_COL] + CLINICAL_COLS + z_cols)
    return df, z_cols


def main():
    cfg    = Config()
    device = cfg.device

    stats  = load_preprocessing_stats(str(Path(cfg.jsons_dir) / "preprocessing_stats.json"))
    splits = load_splits(cfg.splits_dir)

    train_val_ids = splits["train"] + splits["val"]
    test_ids      = splits["test"]
    all_ids       = train_val_ids + test_ids

    # unpadded set for df filtering after merge
    train_val_set = {unpad_ucsf_ids(s) if "UCSF" in s else s for s in train_val_ids}

    with open(Path(cfg.jsons_dir) / "id_to_cohort.json") as f:
        id_to_cohort = json.load(f)

    clinical = pd.read_csv(cfg.clinical_csv)
    out_dir = Path(cfg.jsons_dir)

    summaries_dir = out_dir / "cox_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for dim in LATENT_DIMS:
        run_cfg = Config()
        run_cfg.latent_dim       = dim
        run_cfg.normalisation    = "zscore"
        run_cfg.use_lr_scheduler = True

        logger.info("=== latent_dim=%d ===", dim)
        codes, ids = _get_or_encode(run_cfg, all_ids, id_to_cohort, stats, device)
        if codes is None:
            continue

        df, z_cols = _build_df(codes, ids, clinical, dim)
        train_val_df = df[ df["id"].isin(train_val_set)]
        test_df      = df[~df["id"].isin(train_val_set)]
        logger.info("  train+val n=%d  test n=%d", len(train_val_df), len(test_df))

        models = {
            "clinical":        CLINICAL_COLS,
            "latent":          z_cols,
            "clinical+latent": CLINICAL_COLS + z_cols,
        }

        summaries = []
        dim_results = {}
        for name, covariates in models.items():
            cph = CoxPHFitter()
            cph.fit(train_val_df[[DURATION_COL, EVENT_COL] + covariates],
                    duration_col=DURATION_COL, event_col=EVENT_COL)
            summary = cph.summary.copy()
            summary.insert(0, "model", name)
            summaries.append(summary)
            logger.info("  %-20s  C=%.4f", name, c)
            dim_results[name] = round(c, 4)

        results[f"latent{dim}"] = dim_results
        pd.concat(summaries).to_csv(summaries_dir / f"latent{dim}.csv")

    # # Table
    # print(f"\n{'':12} {'clinical':>10} {'latent':>10} {'clin+latent':>12}")
    # print("-" * 46)
    # for tag, m in results.items():
    #     print(f"{tag:<12} {m['clinical']:>10.4f} {m['latent']:>10.4f} {m['clinical+latent']:>12.4f}")

    # (out_dir / "cox_comparison.json").write_text(json.dumps(results, indent=2))

    # # Plot
    # dims = [int(k.replace("latent", "")) for k in results]
    # fig, ax = plt.subplots(figsize=(6, 4))
    # for name in ("clinical", "latent", "clinical+latent"):
    #     ax.plot(dims, [results[f"latent{d}"][name] for d in dims], marker="o", label=name)
    # ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="random")
    # ax.set_xscale("log", base=2)
    # ax.set_xticks(dims)
    # ax.set_xticklabels(dims)
    # ax.set_xlabel("Latent dimension")
    # ax.set_ylabel("C-index (test set)")
    # ax.set_title("Cox concordance vs latent dimension")
    # ax.legend()
    # fig.tight_layout()
    # fig_path = out_dir / "cox_comparison.png"
    # fig.savefig(fig_path, dpi=150)
    # plt.close(fig)
    # logger.info("Saved -> %s and %s", out_dir / "cox_comparison.json", fig_path)


if __name__ == "__main__":
    main()
