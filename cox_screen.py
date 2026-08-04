"""
Per-latent-component Cox screening across all latent dimensions.

For each latent dim, for each individual z in that dim's latent vector, fits
"clinical" (age, sex) vs "clinical+z" via the same stratified k-fold CV as
cox.py, and checks whether any single component beats the clinical baseline.

This is a much larger search than cox.py's whole-dim comparison (sum of all
z's across all dims -- 272 candidates for the default LATENT_DIMS) -- treat any
standout candidate as a lead to check independently, not a confirmed finding
(see the dim=4 "z3" false lead: best-of-4 in the whole-dim model, but no
improvement over clinical when tested on its own).

Outputs (kept separate from cox.py's jsons/cox_results/):
  jsons/cox_screen/latent{dim}_screen_folds.csv   -- one row per (fold, candidate)
  jsons/cox_screen/latent{dim}_screen_summary.csv -- one row per candidate
  jsons/cox_screen/all_dims_screen_summary.csv    -- all dims concatenated, for
                                                      ranking across everything
"""
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import Config
from cox import (
    CLINICAL_COLS, LATENT_DIMS, N_SPLITS,
    _build_df, _fit_fold_model, _get_or_encode, _strat_key,
)
from utils import load_preprocessing_stats, load_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _cv_screen_components(df: pd.DataFrame, z_cols: list[str], n_splits: int = N_SPLITS,
                           seed: int = Config.random_seed) -> pd.DataFrame:
    """Stratified k-fold CV: 'clinical' baseline vs 'clinical+z' for every z.

    The clinical baseline is fit once per fold (it doesn't depend on z), and
    every candidate in a fold uses the same train/test split as the baseline
    and every other candidate, for a fair paired comparison.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strat_key = _strat_key(df)

    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(df, strat_key)):
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

        base = _fit_fold_model(df_train, df_test, CLINICAL_COLS)
        rows.append({
            "fold": fold_id, "candidate": "clinical",
            "train_c": base["train_c"], "test_c": base["test_c"],
            "hr": None, "p": None,
        })

        for z in z_cols:
            result = _fit_fold_model(df_train, df_test, CLINICAL_COLS + [z])
            z_row = result["covariates"].loc[z]
            rows.append({
                "fold": fold_id, "candidate": z,
                "train_c": result["train_c"], "test_c": result["test_c"],
                "hr": z_row["hr"], "p": z_row["p"],
            })

    return pd.DataFrame(rows)


def _summarize_screen(screen_folds: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate: mean/std C-index, mean HR/p, fraction of folds
    significant, and delta_test_c_vs_clinical (paired, since every candidate
    was evaluated on the same folds as the clinical baseline)."""
    baseline_test_c = screen_folds.loc[screen_folds["candidate"] == "clinical", "test_c"].mean()

    valid_p = screen_folds.dropna(subset=["p"])
    frac_sig = (
        valid_p.assign(sig=valid_p["p"] < 0.05)
        .groupby("candidate")["sig"].mean()
        .rename("frac_folds_p_lt_05")
    )

    summary = screen_folds.groupby("candidate").agg(
        train_c_mean=("train_c", "mean"), train_c_std=("train_c", "std"),
        test_c_mean=("test_c", "mean"),   test_c_std=("test_c", "std"),
        mean_hr=("hr", "mean"), mean_p=("p", "mean"),
    ).join(frac_sig).reset_index()

    summary["delta_test_c_vs_clinical"] = summary["test_c_mean"] - baseline_test_c
    return summary.sort_values("delta_test_c_vs_clinical", ascending=False)


def main():
    cfg    = Config()
    device = cfg.device

    stats  = load_preprocessing_stats(str(Path(cfg.jsons_dir) / "preprocessing_stats.json"))
    splits = load_splits(cfg.splits_dir)
    all_ids = splits["train"] + splits["val"] + splits["test"]

    with open(Path(cfg.jsons_dir) / "id_to_cohort.json") as f:
        id_to_cohort = json.load(f)

    clinical = pd.read_csv(cfg.clinical_csv)
    out_dir = Path(cfg.jsons_dir) / "cox_screen"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for dim in LATENT_DIMS:
        run_cfg = Config()
        run_cfg.latent_dim       = dim
        run_cfg.normalisation    = "zscore"
        run_cfg.use_lr_scheduler = True

        logger.info("=== screening latent_dim=%d (%d candidates) ===", dim, dim)
        codes, ids = _get_or_encode(run_cfg, all_ids, id_to_cohort, stats, device)
        if codes is None:
            continue

        df, z_cols = _build_df(codes, ids, clinical, dim)
        screen_folds = _cv_screen_components(df, z_cols)
        screen_summary = _summarize_screen(screen_folds)

        screen_folds.to_csv(out_dir / f"latent{dim}_screen_folds.csv", index=False)
        screen_summary.to_csv(out_dir / f"latent{dim}_screen_summary.csv", index=False)

        top = screen_summary[screen_summary["candidate"] != "clinical"].iloc[0]
        logger.info(
            "  best candidate: %-6s  delta_test_c=%+.4f  frac_folds_p_lt_05=%.2f",
            top["candidate"], top["delta_test_c_vs_clinical"], top["frac_folds_p_lt_05"],
        )

        screen_summary.insert(0, "dim", dim)
        all_summaries.append(screen_summary)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined = combined[combined["candidate"] != "clinical"].sort_values(
        "delta_test_c_vs_clinical", ascending=False
    )
    combined.to_csv(out_dir / "all_dims_screen_summary.csv", index=False)
    logger.info("Saved -> %s", out_dir / "all_dims_screen_summary.csv")


if __name__ == "__main__":
    main()
