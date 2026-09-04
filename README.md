## Project structure

`config.py` holds the shared `Config` dataclass (paths, data params, model/training hyperparams) used across every stage below.

#### Data analysis and preprocessing

- `utils.py` - dataset classes (`TDMapDataset`, `CachedTDMapDataset`), preprocessing (bbox/crop/pad, normalisation stats), split creation/loading.
- `prepare_splits_and_stats.py` - idempotent bootstrap: train/val/test splits, `id_to_cohort.json`, bbox/padded_shape in `jsons/preprocessing_stats.json`.
- `preprocess_and_cache.py` (`preprocess.sh`) - caches cropped/padded volumes as float16 tensors and writes normalisation stats (train split only) to `jsons/preprocessing_stats.json`.
- `autoencoder_preprocessing_and_training.ipynb` - one-time split creation, preprocessing sanity checks; can also run training locally.
- `splits/` - train/val/test ID lists. `jsons/` - preprocessing stats, id_to_cohort, cox results. `cache/` - cached float16 tensors (HPC only).

#### Model training and inference

- `model.py` - the `Autoencoder` architecture: 4 strided Conv3d blocks (stride 2) encoder, mirrored ConvTranspose3d decoder, flatten+Linear bottleneck. Input dims must be divisible by 2^4=16 (see `bbox_to_padded_shape` in utils.py). Design notes:
  - Strided convs instead of pooling - preserves spatial location info in the latent vector, which matters since the spatial pattern of tract involvement is hypothesis-relevant for survival.
  - Flatten + Linear bottleneck instead of global average pooling - keeps each spatial position a distinct input to the linear layer, so location-specific features can still be learned.
  - LeakyReLU throughout except the final decoder layer - z-scored inputs produce negative activations that plain ReLU would zero out permanently (dying neurons).
  - No activation on the final decoder output - reconstruction targets are continuous and unbounded (z-scored values can be negative), so any output activation would wrongly constrain the range.
- `train.py` - training-loop and `encode_dataset` library functions (no CLI; used by train_sweep.py and the preprocessing notebook).
- `train_sweep.py` (`sweep.sh`) - trains one autoencoder for a given latent dim (SLURM array task).
- `evaluate.py` / `compare.py` (`compare.sh`) - reconstruction-quality metrics and comparisons across normalisation/scheduler baselines and latent dims.
- `autoencoder_inference.ipynb` - exploratory post-hoc inference/visualisation, not part of the reusable pipeline.
- `checkpoints/` - saved model weights. `latents/` - cached encoded codes. `history/` - training loss curves.

#### Survival analysis

- `cox.py` / `cox_screen.py` (`cox.sh`) - Cox survival analysis on trained autoencoders' latent codes (whole-latent-vector and per-component screening respectively).
- `FENS_analyses.ipynb`, `results-OHBM_June-10.ipynb` - exploratory/scratch analysis notebooks (conference prep, ad-hoc reruns), not part of the reusable pipeline.
