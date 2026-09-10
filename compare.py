"""Reconstruction quality comparisons for AE sweep experiments.

Usage:
  python compare.py --mode baselines    # 8 norm x scheduler variants at latent_dim=2
  python compare.py --mode latent_dims  # 7 latent dims at zscore + cosine
  python compare.py --mode all          # both (default)
"""
import argparse
import json
import logging
from pathlib import Path


import matplotlib.pyplot as plt

from config import Config
from evaluate import evaluate_model, summarize
from utils import build_dataset_from_ids, load_preprocessing_stats, load_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPLIT = "test"

_BASELINE_EXPERIMENTS = [
    {"normalisation": "none",         "use_lr_scheduler": True},
    {"normalisation": "none",         "use_lr_scheduler": False},
    {"normalisation": "zscore",       "use_lr_scheduler": True},
    {"normalisation": "zscore",       "use_lr_scheduler": False},
    {"normalisation": "log1p",        "use_lr_scheduler": True},
    {"normalisation": "log1p",        "use_lr_scheduler": False},
    {"normalisation": "log1p_zscore", "use_lr_scheduler": True},
    {"normalisation": "log1p_zscore", "use_lr_scheduler": False},
    {"normalisation": "minmax",       "use_lr_scheduler": True},
]

_LATENT_DIMS = [2, 4, 6, 8, 12, 16, 32, 64, 128]


def _print_table(results: dict) -> None:
    header = (
        f"\n{'Run':<35} {'R²':>7} {'±':>1} {'std_err':>6}"
        f"  {'MAE':>8}  {'Pearson r':>9}  {'SSIM':>7}"
    )
    print(header)
    print("-" * len(header))
    for tag, m in sorted(results.items(), key=lambda x: -x[1]["r2"]["mean"]):
        print(
            f"{tag:<35} {m['r2']['mean']:>7.4f} ± {m['r2']['std_err']:>6.4f}"
            f"  {m['mae']['mean']:>8.4f}  {m['pearson_r']['mean']:>9.4f}"
            f"  {m['ssim']['mean']:>7.4f}"
        )


def compare_baselines(raw_dataset, stats: dict, device, out_dir: Path) -> None:
    results = {}
    for exp in _BASELINE_EXPERIMENTS:
        cfg = Config()
        cfg.latent_dim = 2
        cfg.normalisation = exp["normalisation"]
        cfg.use_lr_scheduler = exp["use_lr_scheduler"]
        logger.info("--- %s ---", cfg.run_tag)
        per_subject = evaluate_model(cfg, raw_dataset, stats, device)
        if per_subject is not None:
            results[cfg.run_tag] = summarize(per_subject)

    _print_table(results)
    out = out_dir / "comparison_baselines.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info("Saved -> %s", out)


def compare_latent_dims(raw_dataset, stats: dict, device, out_dir: Path) -> None:
    results = {}
    for dim in _LATENT_DIMS:
        cfg = Config()
        cfg.latent_dim = dim
        cfg.normalisation = "zscore"
        cfg.use_lr_scheduler = True
        logger.info("--- %s ---", cfg.run_tag)
        per_subject = evaluate_model(cfg, raw_dataset, stats, device)
        if per_subject is not None:
            results[cfg.run_tag] = summarize(per_subject)

    _print_table(results)
    out = out_dir / "comparison_latent_dims.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info("Saved -> %s", out)

    dims = [d for d in _LATENT_DIMS if f"latent{d}_zscore_cosinelr" in results]
    means = [results[f"latent{d}_zscore_cosinelr"]["r2"]["mean"] for d in dims]
    stds  = [results[f"latent{d}_zscore_cosinelr"]["r2"]["std_err"]  for d in dims]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(dims, means, yerr=stds, marker="o", capsize=4)
    # ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels(dims)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("R² (test set, mean ± std_err)")
    ax.set_title("Reconstruction quality vs latent dimension")
    fig.tight_layout()
    fig_path = out_dir / "latent_dims_reconstruction.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot -> %s", fig_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baselines", "latent_dims", "all"], default="all")
    args = parser.parse_args()

    base_cfg = Config()
    device = base_cfg.device
    out_dir = Path(base_cfg.jsons_dir)

    stats = load_preprocessing_stats(str(out_dir / "preprocessing_stats.json"))
    splits = load_splits(base_cfg.splits_dir)

    with open(out_dir / "id_to_cohort.json") as f:
        id_to_cohort = json.load(f)

    cache_dir = base_cfg.cache_dir if Path(base_cfg.cache_dir).exists() else None
    raw_cfg = Config()
    raw_cfg.normalisation = "none"
    raw_dataset = build_dataset_from_ids(
        splits[SPLIT], id_to_cohort, raw_cfg,
        bbox=stats["bbox"], padded_shape=stats["padded_shape"],
        norm_mean=None, norm_std=None, normalisation="none",
        cache_dir=cache_dir,
    )
    logger.info("Evaluating on %d %s subjects", len(raw_dataset), SPLIT)

    if args.mode in ("baselines", "all"):
        compare_baselines(raw_dataset, stats, device, out_dir)
    if args.mode in ("latent_dims", "all"):
        compare_latent_dims(raw_dataset, stats, device, out_dir)


if __name__ == "__main__":
    main()
