"""
Cox proportional hazards analysis on AE latent codes across latent dimensions.

For each latent dim:
  1. Encode all subjects with the trained AE (cached to disk on first run).
  2. Merge with clinical data (all subjects, regardless of AE train/val/test split).
  3. Compare a "clinical" (age, sex) and a "clinical+latent" (age, sex, z1..zN)
     Cox model via stratified k-fold cross-validation.

Per-fold, per-model outputs (nothing is averaged away):
  jsons/cox_results/latent{dim}_model_folds.csv       — one row per (fold, model)
  jsons/cox_results/latent{dim}_covariate_folds.csv   — one row per (fold, model, covariate)
  jsons/cox_results/latent{dim}_covariate_summary.csv — one row per (model, covariate)
  jsons/cox_results/model_comparison.json             — mean/std C-index per model, all dims
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold

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
N_SPLITS         = 5


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


def _strat_key(df: pd.DataFrame) -> pd.Series:
    """Stratification key combining cohort and censoring status."""
    return df["cohort"].astype(str) + "_" + df[EVENT_COL].astype(str)


def _fit_fold_model(df_train: pd.DataFrame, df_test: pd.DataFrame,
                     covariates: list[str], penalizer: float = 0.0,
                     l1_ratio: float = 0.0) -> dict:
    """Fit one Cox model on df_train, evaluate on df_test.

    Returns model-level stats (train_c, test_c, n_train, n_test,
    n_events_train, n_events_test, converged) plus a "covariates" DataFrame
    (coef, hr, hr_ci_low, hr_ci_high, p) indexed by covariate name.

    On fit failure, C-indices and all covariate fields are NaN rather than 0 —
    a non-convergent fold is missing data, not evidence of a bad model.
    """
    base = {
        "n_train": len(df_train), "n_test": len(df_test),
        "n_events_train": int(df_train[EVENT_COL].sum()),
        "n_events_test": int(df_test[EVENT_COL].sum()),
    }
    try:
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        cph.fit(df_train[[DURATION_COL, EVENT_COL] + covariates],
                duration_col=DURATION_COL, event_col=EVENT_COL)
        test_c = concordance_index(
            df_test[DURATION_COL], -cph.predict_partial_hazard(df_test), df_test[EVENT_COL]
        )
        cov = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
        cov.columns = ["coef", "hr", "hr_ci_low", "hr_ci_high", "p"]
        return {**base, "train_c": cph.concordance_index_, "test_c": test_c,
                "converged": True, "covariates": cov}
    except Exception as e:
        logger.warning("  fit failed for %s: %s", covariates, e)
        cov = pd.DataFrame(
            {"coef": np.nan, "hr": np.nan, "hr_ci_low": np.nan, "hr_ci_high": np.nan, "p": np.nan},
            index=covariates,
        )
        return {**base, "train_c": np.nan, "test_c": np.nan,
                "converged": False, "covariates": cov}


def _cv_compare_models(df: pd.DataFrame, z_cols: list[str], n_splits: int = N_SPLITS,
                        seed: int = Config.random_seed) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified k-fold CV comparing 'clinical' vs 'clinical+latent' models.

    The same fold indices are reused for both models, so every fold yields a
    paired (clinical_test_c, clinical+latent_test_c) observation.
    """
    models = {
        "clinical":        CLINICAL_COLS,
        "clinical+latent": CLINICAL_COLS + z_cols,
    }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strat_key = _strat_key(df)

    model_rows, covariate_rows = [], []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(df, strat_key)):
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
        for name, covariates in models.items():
            result = _fit_fold_model(df_train, df_test, covariates)
            model_rows.append({
                "fold": fold_id, "model": name,
                "train_c": result["train_c"], "test_c": result["test_c"],
                "n_train": result["n_train"], "n_test": result["n_test"],
                "n_events_train": result["n_events_train"],
                "n_events_test": result["n_events_test"],
                "converged": result["converged"],
            })
            for covariate, row in result["covariates"].iterrows():
                covariate_rows.append({
                    "fold": fold_id, "model": name, "covariate": covariate,
                    "coef": row["coef"], "hr": row["hr"],
                    "hr_ci_low": row["hr_ci_low"], "hr_ci_high": row["hr_ci_high"],
                    "p": row["p"],
                })

    return pd.DataFrame(model_rows), pd.DataFrame(covariate_rows)


def _summarize_covariates(covariate_folds: pd.DataFrame) -> pd.DataFrame:
    """Per (model, covariate): mean HR, mean p, and fraction of folds with p<0.05.

    p-values are never averaged for significance claims — frac_folds_p_lt_05
    (how often this covariate was significant across folds) is the reported
    stability metric; mean_p/mean_hr are included as descriptive summaries.
    """
    valid = covariate_folds.dropna(subset=["p"])
    summary = valid.groupby(["model", "covariate"], sort=False).agg(
        mean_hr=("hr", "mean"),
        median_hr=("hr", "median"),
        mean_p=("p", "mean"),
        n_folds_converged=("p", "size"),
    ).reset_index()
    frac_sig = (
        valid.assign(sig=valid["p"] < 0.05)
        .groupby(["model", "covariate"], sort=False)["sig"].mean()
        .reset_index(name="frac_folds_p_lt_05")
    )
    return summary.merge(frac_sig, on=["model", "covariate"])


def main():
    cfg    = Config()
    device = cfg.device

    stats  = load_preprocessing_stats(str(Path(cfg.jsons_dir) / "preprocessing_stats.json"))
    splits = load_splits(cfg.splits_dir)
    all_ids = splits["train"] + splits["val"] + splits["test"]

    with open(Path(cfg.jsons_dir) / "id_to_cohort.json") as f:
        id_to_cohort = json.load(f)

    clinical = pd.read_csv(cfg.clinical_csv)
    out_dir = Path(cfg.jsons_dir) / "cox_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_comparison = {}
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
        logger.info("  n=%d subjects (all AE splits pooled), %d-fold CV", len(df), N_SPLITS)

        model_folds, covariate_folds = _cv_compare_models(df, z_cols)
        covariate_summary = _summarize_covariates(covariate_folds)

        model_folds.to_csv(out_dir / f"latent{dim}_model_folds.csv", index=False)
        covariate_folds.to_csv(out_dir / f"latent{dim}_covariate_folds.csv", index=False)
        covariate_summary.to_csv(out_dir / f"latent{dim}_covariate_summary.csv", index=False)

        dim_summary = {}
        for name, g in model_folds.groupby("model"):
            dim_summary[name] = {
                "train_c_mean": g["train_c"].mean(), "train_c_std": g["train_c"].std(),
                "test_c_mean":  g["test_c"].mean(),  "test_c_std":  g["test_c"].std(),
            }
            logger.info(
                "  %-16s  train_C=%.4f±%.4f  test_C=%.4f±%.4f", name,
                dim_summary[name]["train_c_mean"], dim_summary[name]["train_c_std"],
                dim_summary[name]["test_c_mean"], dim_summary[name]["test_c_std"],
            )
        model_comparison[f"latent{dim}"] = dim_summary

    (out_dir / "model_comparison.json").write_text(json.dumps(model_comparison, indent=2))
    logger.info("Saved -> %s", out_dir / "model_comparison.json")


if __name__ == "__main__":
    main()
