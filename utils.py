"""
Utilities for loading and managing 3D Tract Density Map (TDMap) datasets
"""

import json
import math
import os
import subprocess
import logging
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def load_patient_ids(root: str, cohort_dirs: dict) -> dict[str, list[str]]:
    """
    Read the first column of each cohort's demographics CSV as subject IDs.
    """
    patient_ids = {}
    total = 0
    for cohort, data_dir in cohort_dirs.items():
        csv_path = f"{root}/{data_dir}/demographics-TDMaps_streamTH-0.csv"
        df       = pd.read_csv(csv_path)

        # conditions for filtering relevant data for different cohorts - upenn and tcga are all ok
        # TODO: ARE THEY?
        if cohort == "ucsf":
            # NOTE: for SSL all that are wildtype are ok, but remove missing OS data for survival prediction tasks
            is_wildtype = df["Final pathologic diagnosis (WHO 2021)"].str.contains("wildtype")
            df = df[is_wildtype]
        elif cohort == "rhuh":
            had_no_prev_treatment = df["Previous treatment"].str.contains("no")
            df = df[had_no_prev_treatment]

        ids = df.iloc[:, 0].astype(str).tolist()
        patient_ids[cohort] = ids
        total += len(ids)

        print(f"  {cohort.upper():8s} — {len(ids):4d} subjects  (CSV: {df.shape})")

    print(f"\n  Total: {total} subjects across {len(cohort_dirs)} cohorts")
    return patient_ids


def pad_ucsf_ids(ids: list[str]) -> list[str]:
    """Zero-pad the numeric suffix of UCSF IDs to 4 digits."""
    fixed = []
    for val in ids:
        parts = val.split("-")
        parts[-1] = parts[-1].zfill(4)
        fixed.append("-".join(parts))
    return fixed

def unpad_ucsf_ids(id: str) -> str:
    """Strip leading zeros from the numeric suffix of a UCSF ID."""
    parts = id.split("-")
    parts[-1] = str(parts[-1][1:])
    return "-".join(parts)

def build_id_to_cohort(patient_ids: dict[str, list[str]]) -> dict[str, str]:
    return {sid: cohort for cohort, ids in patient_ids.items() for sid in ids}

def regrid_map(
    subject: str,
    template: str,
    root: str,
    data_dir: str,
    tissue: str,
    tdmap: str,
    voxel_res: float,
    nthreads: int,
) -> str:
    """Ensure a regridded TDMap exists for a subject, creating it via mrgrid if needed.
    Cached next to the original map, so this is a one-time cost per subject."""
    maps_dir = Path(root) / data_dir / subject / "maps"

    regridded = maps_dir / f"{subject}_tissue-{tissue}_{tdmap}_voxel-{voxel_res}.nii.gz"
    original  = maps_dir / f"{subject}_tissue-{tissue}_{tdmap}.nii.gz"

    if regridded.exists():
        return str(regridded)

    if not original.exists():
        raise FileNotFoundError(f"Original map missing for subject '{subject}': {original}")

    logger.info("Regridding %s -> %s", original.name, regridded.name)
    cmd = [
        "mrgrid", str(original), "regrid",
        "-template", template,
        "-voxel", str(voxel_res),
        str(regridded),
        "-quiet",
        "-nthreads", str(nthreads),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(
            f"mrgrid failed for subject '{subject}'.\n"
            f"stderr: {result.stderr.strip()}"
        )

    return str(regridded)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_VALID_NORMALISATIONS = {"none", "log1p", "zscore", "log1p_zscore"}


def _apply_normalisation(
    vol: torch.Tensor,
    normalisation: str,
    norm_mean: float | None,
    norm_std: float | None,
) -> torch.Tensor:
    """Apply log1p and/or z-score normalisation to a volume tensor in-place order."""
    if normalisation in ("log1p", "log1p_zscore"):
        vol = torch.log1p(vol)
    if normalisation in ("zscore", "log1p_zscore"):
        vol = (vol - norm_mean) / (norm_std + 1e-8)
    return vol


class CachedTDMapDataset(Dataset):
    """
    Fast drop-in for TDMapDataset that reads pre-cached float16 tensors written
    by preprocess_and_cache.py (crop + pad applied, no normalization).
    Normalization is applied on-the-fly — no file I/O after the initial load.

    Parameters
    ----------
    subjects      : Subject ID list.
    cache_dir     : Directory containing {subject_id}.pt files.
    normalisation : Same options as TDMapDataset.
    norm_mean     : Required for "zscore" / "log1p_zscore".
    norm_std      : Required for "zscore" / "log1p_zscore".
    """

    def __init__(
        self,
        subjects: list[str],
        cache_dir: str,
        normalisation: str = "zscore",
        norm_mean: float | None = None,
        norm_std: float | None = None,
    ) -> None:
        if normalisation not in _VALID_NORMALISATIONS:
            raise ValueError(f"normalisation must be one of {_VALID_NORMALISATIONS}, got {normalisation!r}")
        if normalisation in ("zscore", "log1p_zscore") and (norm_mean is None or norm_std is None):
            raise ValueError(f"norm_mean and norm_std are required when normalisation={normalisation!r}")

        self.subjects = list(subjects)
        self.cache_dir = Path(cache_dir)
        self.normalisation = normalisation
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load float16 tensor and cast to float32 for training (cast is near-zero cost)
        vol = torch.load(
            self.cache_dir / f"{self.subjects[idx]}.pt", weights_only=True
        ).to(torch.float32)
        return _apply_normalisation(vol, self.normalisation, self.norm_mean, self.norm_std)


class TDMapDataset(Dataset):
    """PyTorch Dataset for 3D Tract Density Maps, (1, D, H, W).

    Regridding is lazy: first access shells out to mrgrid and caches the result to
    disk; later accesses just read the cached file. Negative voxel values
    (interpolation artefacts) are clamped to zero on load.

    normalisation: "none" (clamp only) | "log1p" | "zscore" (needs norm_mean/std) |
    "log1p_zscore" (log1p then z-score, needs norm_mean/std).

    Pipeline when bbox/padded_shape are given: load -> clamp(0) -> crop -> pad -> [normalisation].
    """

    def __init__(
        self,
        subjects: list[str],
        template: str,
        root: str,
        data_dir: str,
        tissue: str,
        tdmap: str,
        voxel_res: float,
        nthreads: int,
        dtype: torch.dtype = torch.float32,
        bbox: dict[str, int] | None = None,
        padded_shape: tuple[int, int, int] | None = None,
        normalisation: str = "log1p_zscore",
        norm_mean: float | None = None,
        norm_std: float | None = None,
        brain_mask: torch.Tensor | None = None
    ) -> None:

        if normalisation not in _VALID_NORMALISATIONS:
            raise ValueError(f"normalisation must be one of {_VALID_NORMALISATIONS}, got {normalisation!r}")
        if normalisation in ("zscore", "log1p_zscore") and (norm_mean is None or norm_std is None):
            raise ValueError(f"norm_mean and norm_std are required when normalisation={normalisation!r}")

        self.subjects = list(subjects)
        self.template = template
        self.root = root
        self.data_dir = data_dir
        self.tissue = tissue
        self.tdmap = tdmap
        self.voxel_res = voxel_res
        self.nthreads = nthreads
        self.dtype = dtype
        self.bbox = bbox
        self.padded_shape = padded_shape
        self.normalisation = normalisation
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.brain_mask = brain_mask

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> torch.Tensor:
        subject = self.subjects[idx]

        path = regrid_map(
            subject=subject,
            template=self.template,
            root=self.root,
            data_dir=self.data_dir,
            tissue=self.tissue,
            tdmap=self.tdmap,
            voxel_res=self.voxel_res,
            nthreads=self.nthreads,
        )

        vol = nib.load(path).get_fdata()           # (D, H, W) float64
        vol = torch.from_numpy(vol).to(self.dtype) # (D, H, W) float32
        vol = torch.clamp(vol, min=0.0)            # remove interpolation artefacts
        vol = vol.unsqueeze(0)                     # (1, D, H, W) adds channel dim for CNNs

        if self.bbox is not None:
            b = self.bbox
            vol = vol[
                :,
                b["z_min"]:b["z_max"],
                b["y_min"]:b["y_max"],
                b["x_min"]:b["x_max"],
            ]

        if self.padded_shape is not None:
            pd = self.padded_shape[0] - vol.shape[1]
            ph = self.padded_shape[1] - vol.shape[2]
            pw = self.padded_shape[2] - vol.shape[3]
            # F.pad takes padding in reverse dim order: (W_left, W_right, H_left, H_right, D_left, D_right)
            vol = F.pad(vol, (0, pw, 0, ph, 0, pd))

        # if self.brain_mask is not None:
        # vol = vol * self.brain_mask # NOTE: This doesn't work when there's z-scoring because then some values are negative and the images get super weird
        # TODO: brain_mask is accepted/stored but unused everywhere right now. Intent was to
        # restrict volumes (or norm stats) to within-brain voxels only — revisit before
        # writing up the pipeline.
        return _apply_normalisation(vol, self.normalisation, self.norm_mean, self.norm_std)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n={len(self)}, tissue='{self.tissue}', "
            f"tdmap='{self.tdmap}', voxel_res={self.voxel_res}mm)"
            f"normalisation='{self.normalisation}')"
        )


# ---------------------------------------------------------------------------
# Preprocessing utilities
# ---------------------------------------------------------------------------

def compute_bounding_box(mask_path: str) -> dict[str, int]:
    """Tight bounding box of non-zero voxels in the brain mask. Computed once from
    the template mask (all subjects share registration) and applied to everyone.
    Returns dict with z_min/z_max/y_min/y_max/x_min/x_max (upper bounds exclusive)."""
    mask = nib.load(mask_path).get_fdata()
    coords = np.argwhere(mask > 0)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1  # +1 so indices are exclusive (used in slicing)
    return {
        "z_min": int(lo[0]), "z_max": int(hi[0]),
        "y_min": int(lo[1]), "y_max": int(hi[1]),
        "x_min": int(lo[2]), "x_max": int(hi[2]),
    }


def bbox_to_padded_shape(
    bbox: dict[str, int],
    multiple: int = 16,
) -> tuple[int, int, int]:
    """(D, H, W) padded to the next multiple of `multiple` (default 16 = 2**4,
    matching the encoder's 4 stride-2 layers in model.py)."""
    raw_d = bbox["z_max"] - bbox["z_min"]
    raw_h = bbox["y_max"] - bbox["y_min"]
    raw_w = bbox["x_max"] - bbox["x_min"]
    pad = lambda n: ((n + multiple - 1) // multiple) * multiple
    return (pad(raw_d), pad(raw_h), pad(raw_w))


def compute_normalisation_stats(dataset: Dataset, mask: torch.Tensor | None = None) -> dict[str, float]:
    """Compute mean/std over dataset voxels. If mask is given, only masked voxels
    contribute (excludes padding/background)."""
    total_sum, total_sq_sum, total_count = 0.0, 0.0, 0

    for i in range(len(dataset)):
        vol = dataset[i].float()
        flat = vol[mask > 0].flatten() if mask is not None else vol.flatten()
        logger.info(f"Mask present: {mask is not None}")
        total_sum    += flat.sum().item()
        total_sq_sum += flat.pow(2).sum().item()
        total_count  += flat.numel()

    mean = total_sum / total_count
    std  = math.sqrt(max(total_sq_sum / total_count - mean ** 2, 0.0)) # computational std formula using expectation values
    logger.info("Normalisation stats — mean: %.4f  std: %.4f  (n_voxels: %d)", mean, std, total_count)

    return {"mean": float(mean), "std": float(std)}



def load_brain_mask(
    mask_path: str,
    bbox: dict[str, int],
    padded_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Load brain mask, apply the same crop + pad as TDMapDataset.
    Returns (1, D, H, W) float32, values 0/1."""
    mask = nib.load(mask_path).get_fdata()
    mask = (mask > 0.5).astype(np.float32)
    mask = torch.from_numpy(mask).unsqueeze(0)  # (1, D, H, W)

    b = bbox
    mask = mask[
        :,
        b["z_min"]:b["z_max"],
        b["y_min"]:b["y_max"],
        b["x_min"]:b["x_max"],
    ]

    pd = padded_shape[0] - mask.shape[1]
    ph = padded_shape[1] - mask.shape[2]
    pw = padded_shape[2] - mask.shape[3]
    mask = F.pad(mask, (0, pw, 0, ph, 0, pd))

    return mask  # (1, D, H, W)


def save_preprocessing_stats(
    bbox: dict[str, int],
    padded_shape: tuple[int, int, int],
    norm_mean: float | None,
    norm_std: float | None,
    path: str,
) -> None:
    """Save all preprocessing parameters to a JSON file for reproducibility."""
    stats = {
        "bbox": bbox,
        "padded_shape": list(padded_shape),
        "norm_mean": norm_mean,
        "norm_std": norm_std,
    }
    Path(path).write_text(json.dumps(stats, indent=2))
    logger.info("Saved preprocessing stats -> %s", path)


def load_preprocessing_stats(path: str) -> dict:
    """Load preprocessing parameters saved by save_preprocessing_stats."""
    stats = json.loads(Path(path).read_text())
    stats["padded_shape"] = tuple(stats["padded_shape"])
    return stats


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset_from_ids(
    sids: list[str],
    id_to_cohort: dict[str, str],
    cfg,
    bbox: dict[str, int] | None = None,
    padded_shape: tuple[int, int, int] | None = None,
    norm_mean: float | None = None,
    norm_std: float | None = None,
    normalisation: str = "log1p_zscore",
    brain_mask: torch.Tensor | None = None,
    cache_dir: str | None = None,
) -> Dataset:
    """Build a dataset from subject IDs spanning multiple cohorts.

    Uses CachedTDMapDataset if cache_dir is given (pre-cached tensors from
    preprocess_and_cache.py), otherwise TDMapDataset (reads NIfTI directly).
    brain_mask is accepted but not currently applied anywhere — see the TODO
    in TDMapDataset.__getitem__.
    """
    if cache_dir is not None:
        return CachedTDMapDataset(
            subjects=sids,
            cache_dir=cache_dir,
            normalisation=normalisation,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

    cohort_buckets: dict[str, list[str]] = {c: [] for c in cfg.cohort_dirs}
    for sid in sids:
        cohort_buckets[id_to_cohort[sid]].append(sid)

    datasets = []
    for cohort, bucket in cohort_buckets.items():
        if not bucket:
            continue
        datasets.append(TDMapDataset(
            subjects=bucket,
            template=cfg.template,
            root=cfg.gbm_root,
            data_dir=cfg.cohort_dirs[cohort],
            tissue=cfg.tissue,
            tdmap=cfg.tdmap,
            voxel_res=cfg.voxel_res,
            nthreads=cfg.nthreads,
            bbox=bbox,
            padded_shape=padded_shape,
            normalisation=normalisation,
            norm_mean=norm_mean,
            norm_std=norm_std,
            brain_mask=brain_mask
        ))
    return ConcatDataset(datasets)


# ---------------------------------------------------------------------------
# Train / val / test splitting
# ---------------------------------------------------------------------------

def split_ids(
    patient_ids: dict[str, list[str]],
    strategy: str = "cohort_holdout",
    test_cohorts: list[str] | None = None,
    test_fraction: float = 0.15,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split subject IDs into train/val/test at the patient level (same
    subject never appears in more than one split).

    "cohort_holdout": test_cohorts held out entirely; rest split into train/val.
    "stratified": test_fraction then val_fraction sampled from each cohort.

    Parameters
    ----------
    patient_ids   : {cohort_name: [subject_id, ...]}
    strategy      : "cohort_holdout" or "stratified"
    test_cohorts  : held out entirely (cohort_holdout only)
    test_fraction : per-cohort test share (stratified only)
    val_fraction  : share of remaining pool for validation
    seed          : RNG seed
    # TODO: check if anything should be changed for reproducibility

    Returns
    -------
    {"train": [subject_id, ...], "val": [...], "test": [...]}
    """
    rng = np.random.default_rng(seed)

    if strategy == "cohort_holdout":
        if not test_cohorts:
            raise ValueError("strategy='cohort_holdout' requires test_cohorts to be set.")

        test_ids: list[str] = []
        pool: list[str] = []
        for cohort, ids in patient_ids.items():
            if cohort in test_cohorts:
                test_ids.extend(ids)
            else:
                pool.extend(ids)

    elif strategy == "stratified":
        test_ids = []
        pool     = []
        for ids in patient_ids.values():
            shuffled = ids.copy()
            rng.shuffle(shuffled)
            n_test = max(1, int(len(shuffled) * test_fraction))
            test_ids.extend(shuffled[:n_test])
            pool.extend(shuffled[n_test:])

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'cohort_holdout' or 'stratified'.")

    rng.shuffle(pool)
    n_val     = max(1, int(len(pool) * val_fraction))
    val_ids   = pool[:n_val]
    train_ids = pool[n_val:]

    logger.info(
        "Split [%s] → train: %d | val: %d | test: %d",
        strategy, len(train_ids), len(val_ids), len(test_ids),
    )
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def save_splits(splits: dict[str, list[str]], output_dir: str) -> None:
    """Persist the train/val/test ID lists (output of split_ids) as plain-text files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ids in splits.items():
        path = output_dir / f"{split_name}_ids.txt"
        path.write_text("\n".join(ids))
        logger.info("Saved %d IDs -> %s", len(ids), path)

def load_splits(output_dir: str) -> dict[str, list[str]]:
    """Load train/val/test ID lists saved by save_splits."""
    output_dir = Path(output_dir)
    return {
        path.stem.removesuffix("_ids"): path.read_text().splitlines()
        for path in sorted(output_dir.glob("*_ids.txt"))
    }