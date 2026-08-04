## Project structure

`config.py` holds the shared `Config` dataclass (paths, data params, model/training hyperparams) used across every stage below.

#### Data analysis and preprocessing

- `utils.py` - dataset classes (`TDMapDataset`, `CachedTDMapDataset`), preprocessing (bbox/crop/pad, normalisation stats), split creation/loading.
- `preprocess_and_cache.py` (`preprocess.sh`) - caches cropped/padded volumes as float16 tensors and writes normalisation stats to `jsons/preprocessing_stats.json`.
- `autoencoder_preprocessing_and_training.ipynb` - one-time split creation, preprocessing sanity checks; can also run training locally.
- `splits/` - train/val/test ID lists. `jsons/` - preprocessing stats, id_to_cohort, cox results. `cache/` - cached float16 tensors (HPC only).

#### Model training and inference

- `model.py` - the `Autoencoder` architecture.
- `train.py` - training-loop and `encode_dataset` library functions (no CLI; used by train_sweep.py and the preprocessing notebook).
- `train_sweep.py` (`sweep.sh`) - trains one autoencoder for a given latent dim (SLURM array task).
- `evaluate.py` / `compare.py` (`compare.sh`) - reconstruction-quality metrics and comparisons across normalisation/scheduler baselines and latent dims.
- `autoencoder_inference.ipynb` - exploratory post-hoc inference/visualisation, not part of the reusable pipeline.
- `checkpoints/` - saved model weights. `latents/` - cached encoded codes. `history/` - training loss curves.

#### Survival analysis

- `cox.py` / `cox_screen.py` (`cox.sh`) - Cox survival analysis on trained autoencoders' latent codes (whole-latent-vector and per-component screening respectively).
- `FENS_analyses.ipynb`, `results-OHBM_June-10.ipynb` - exploratory/scratch analysis notebooks (conference prep, ad-hoc reruns), not part of the reusable pipeline.
