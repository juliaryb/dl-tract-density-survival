import os
import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    # Environment paths
    # gbm_root: str = "/home/joan/Desktop/PROJECTS/Glioblastomas" # uncomment for local
    gbm_root: str = "/net/tscratch/people/plgjuliaryb/data/Glioblastomas" # uncomment for Athena
    # mni_dir:  str = "/home/joan/Documents/MNI_ICBM_2009b_NLIN_ASYM"
    mni_dir:  str = "/net/tscratch/people/plgjuliaryb/data/MNI_ICBM_2009b_NLIN_ASYM"
    # root:     str = "/home/joan/Desktop/PROJECTS/Julia/code/dl-tract-density-survival"
    root:     str = "/net/tscratch/people/plgjuliaryb/data/dl-tract-density-survival-outputs"
    # clinical_csv: str = "/home/joan/Desktop/PROJECTS/Glioblastomas/RESULTS-GBM_4-cohorts_Tissues/data-clinical_TD-tissues_4-cohorts.csv"
    clinical_csv: str = "/net/tscratch/people/plgjuliaryb/data/Glioblastomas/data-clinical_TD-tissues_4-cohorts.csv"
    
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
    # encoder_channels: tuple = (8, 16, 32, 64)
    # encoder_channels: tuple = (16, 32, 64, 128)    
    latent_dim: int = 4
    kernel_size: int = 3

    # Training
    batch_size:           int   = 4
    num_workers:          int   = 4
    lr:                   float = 1e-4
    weight_decay:         float = 1e-4
    epochs:               int   = 300
    patience:             int   = 20
    use_lr_scheduler:     bool  = True
    early_stopping_delta: float = 1e-4
    checkpoint_every:     int   = 10 # (0 = off)

    # W&B
    # wandb_project: str = "gbm-tdmap-autoencoder"
    # wandb_project: str = "gbm-tdmap-autoencoder-athena"
    # wandb_project: str = "ae-baseline-sweep"
    wandb_project: str = "ae-latent-dims"

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
    def run_tag(self) -> str:
        sched_tag = "cosine" if self.use_lr_scheduler else "flat"
        return f"latent{self.latent_dim}_{self.normalisation}_{sched_tag}lr"

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.root, "checkpoints", self.run_tag)
    
    @property
    def history_dir(self) -> str:
        return os.path.join(self.root, "history")
    
    @property
    def latents_dir(self) -> str:
        return os.path.join(self.root, "latents")

    # @property
    # def save_path(self) -> str:
    #     return os.path.join(self.root, "checkpoints", "autoencoder_best.pt")

    @property
    def jsons_dir(self) -> str:
        return os.path.join(self.root, "jsons")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.root, "cache")

    # @property
    # def preprocessing_path(self) -> str:
    #     return os.path.join(self.root, "preprocessing_stats.json")

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def model_name(self) -> str:
        return "best.pt"

