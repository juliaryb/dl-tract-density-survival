import os
import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    # Environment paths (edit for your machine)
    gbm_root: str = "/home/joan/Desktop/PROJECTS/Glioblastomas"
    mni_dir:  str = "/home/joan/Documents/MNI_ICBM_2009b_NLIN_ASYM"
    root:     str = "/home/joan/Desktop/PROJECTS/Julia/code/dl-tract-density-survival"
    clinical_csv: str = "/home/joan/Desktop/PROJECTS/Glioblastomas/RESULTS-GBM_4-cohorts_Tissues/data-clinical_TD-tissues_4-cohorts.csv"

    # Data
    voxel_res: float = 1.5
    tissue:    str   = "whole"
    tdmap:     str   = "TDMap-lesion"
    nthreads:  int   = 8

    cohort_dirs: dict = field(default_factory=lambda: {
        "ucsf":  "Glioblastoma_UCSF-PDGM_v3-20230111/TDMaps_Grade-IV",
        "upenn": "Glioblastoma_UPENN-GBM_v2-20221024/TDMaps_IDH1-WT",
        "rhuh":  "Glioblastoma_RHUH-GBM_v2-29102025/TDMaps_IDH1-WT",
        "tcga":  "Glioblastoma_TCGA-GBM_v1-20170717/TDMaps_IDH1-WT",
    })

    # Splits
    split_strategy: str   = "stratified"        # "stratified" or "cohort_holdout"
    test_cohorts:   list  = field(default_factory=lambda: ["rhuh", "tcga"])
    val_fraction:   float = 0.15
    test_fraction:  float = 0.15
    random_seed:    int   = 42

    # Normalisation: "none" | "log1p" | "zscore" | "log1p_zscore"
    normalisation: str = "zscore"

    # Model
    # encoder_channels controls depth (len) and width (values) of the conv stack.
    # Decoder mirrors this in reverse. Input dims must be divisible by 2**len(channels).
    encoder_channels: tuple = (8, 16, 32, 64)    
    latent_dim: int = 2

    # Training
    batch_size:   int   = 4
    num_workers:  int   = 4
    lr:           float = 1e-4
    weight_decay: float = 1e-4
    epochs:       int   = 300
    patience:     int   = 20

    # W&B
    wandb_project: str = "gbm-tdmap-autoencoder"

    # derived paths (not included in asdict, computed from fields above)
    @property
    def template(self) -> str:
        return os.path.join(self.mni_dir, f"T1_{self.voxel_res}mm.nii")

    @property
    def brain_mask(self) -> str:
        return os.path.join(self.mni_dir, f"T1_{self.voxel_res}mm_brain_mask.nii")

    @property
    def splits_dir(self) -> str:
        return os.path.join(self.root, "splits")

    @property
    def save_path(self) -> str:
        return os.path.join(self.root, "checkpoints", "autoencoder_best.pt")

    @property
    def preprocessing_path(self) -> str:
        return os.path.join(self.root, "preprocessing_stats.json")

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
