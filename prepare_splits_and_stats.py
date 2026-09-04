"""Creates train/val/test splits, id_to_cohort.json, and bbox/padded_shape
in preprocessing_stats.json. Safe to re-run - skips steps whose outputs already exist.

Normalisation stats (norm_mean/std) are NOT computed here — they require cached
tensors and must be computed on the training split only (no leakage), which is
what preprocess_and_cache.py does.
"""
import json
import logging
from pathlib import Path

from config import Config
from utils import (
    load_patient_ids,
    pad_ucsf_ids,
    build_id_to_cohort,
    split_ids,
    save_splits,
    load_splits,
    compute_bounding_box,
    bbox_to_padded_shape,
    save_preprocessing_stats,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = Config()

    patient_ids = load_patient_ids(cfg.gbm_root, cfg.cohort_dirs)
    patient_ids["ucsf"] = pad_ucsf_ids(patient_ids["ucsf"])

    splits_dir = Path(cfg.splits_dir)
    if all((splits_dir / f"{name}_ids.txt").exists() for name in ("train", "val", "test")):
        logger.info("Splits already exist in %s — skipping", splits_dir)
        load_splits(cfg.splits_dir)  # sanity-check they're readable
    else:
        splits = split_ids(
            patient_ids,
            strategy=cfg.split_strategy,
            test_cohorts=cfg.test_cohorts,
            test_fraction=cfg.test_fraction,
            val_fraction=cfg.val_fraction,
            seed=cfg.random_seed,
        )
        save_splits(splits, cfg.splits_dir)

    id_to_cohort = build_id_to_cohort(patient_ids)
    id_to_cohort_path = Path(cfg.jsons_dir) / "id_to_cohort.json"
    id_to_cohort_path.parent.mkdir(parents=True, exist_ok=True)
    id_to_cohort_path.write_text(json.dumps(id_to_cohort))
    logger.info("Saved id_to_cohort.json -> %s", id_to_cohort_path)

    stats_path = Path(cfg.jsons_dir) / "preprocessing_stats.json"
    existing = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    if "bbox" in existing and "padded_shape" in existing:
        logger.info("bbox/padded_shape already in %s — skipping", stats_path)
    else:
        bbox = compute_bounding_box(cfg.brain_mask)
        padded_shape = bbox_to_padded_shape(bbox)
        save_preprocessing_stats(
            bbox, padded_shape,
            norm_mean=existing.get("norm_mean"), norm_std=existing.get("norm_std"),
            path=str(stats_path),
        )


if __name__ == "__main__":
    main()
