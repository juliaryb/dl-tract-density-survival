"""Train one autoencoder for a given latent dim. Called once per SLURM array task."""
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import wandb

from config import Config
from model import Autoencoder
from train import train, encode_dataset
from utils import build_dataset_from_ids, load_preprocessing_stats, load_splits

# print("CUDA available:", torch.cuda.is_available())
# print("Device count:", torch.cuda.device_count())
# print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
# else "CPU")


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_loader(ids, id_to_cohort, cfg, bbox, padded_shape, norm_mean, norm_std, shuffle, cache_dir=None):
    ds = build_dataset_from_ids(
        ids, id_to_cohort, cfg, bbox, padded_shape,
        norm_mean, norm_std, cfg.normalisation, cache_dir=cache_dir,
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, pin_memory=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--normalisation", default=None,
                        choices=["none", "log1p", "zscore", "log1p_zscore"],
                        help="Override cfg.normalisation")
    parser.add_argument("--no-lr-scheduler", action="store_true",
                        help="Disable cosine annealing (flat LR)")
    parser.add_argument("--early-stopping-delta", type=float, default=None,
                        help="Override cfg.early_stopping_delta")
    args = parser.parse_args()

    cfg = Config()
    cfg.latent_dim = args.latent_dim
    if args.normalisation is not None:
        cfg.normalisation = args.normalisation
    if args.no_lr_scheduler:
        cfg.use_lr_scheduler = False
    if args.early_stopping_delta is not None:
        cfg.early_stopping_delta = args.early_stopping_delta

    device = cfg.device

    stats_path = os.path.join(cfg.jsons_dir, "preprocessing_stats.json")
    stats        = load_preprocessing_stats(stats_path)
    bbox         = stats["bbox"]
    padded_shape = stats["padded_shape"]

    # Select normalisation stats appropriate for the chosen normalisation type.
    # log1p_zscore needs stats computed on log1p-transformed values (written by
    # preprocess_and_cache.py). Falls back to raw stats with a warning if not found.
    if cfg.normalisation == "zscore":
        norm_mean, norm_std = stats["norm_mean"], stats["norm_std"]
    elif cfg.normalisation == "log1p_zscore":
        if "log1p_norm_mean" in stats:
            norm_mean, norm_std = stats["log1p_norm_mean"], stats["log1p_norm_std"]
        else:
            logger.warning(
                "log1p stats not found in preprocessing_stats.json — "
                "run preprocess_and_cache.py first. Falling back to raw stats."
            )
            norm_mean, norm_std = stats["norm_mean"], stats["norm_std"]
    else:
        norm_mean = norm_std = None

    splits = load_splits(cfg.splits_dir)
    train_ids = splits["train"]
    val_ids   = splits["val"]
    test_ids  = splits["test"]
    all_ids   = train_ids + val_ids + test_ids

    with open(f"{cfg.jsons_dir}/id_to_cohort.json") as f:
        id_to_cohort = json.load(f)

    cache_dir = cfg.cache_dir if Path(cfg.cache_dir).exists() else None
    if cache_dir:
        logger.info("Using pre-cached tensors from %s", cache_dir)

    loader_kwargs = dict(id_to_cohort=id_to_cohort, cfg=cfg, bbox=bbox,
                         padded_shape=padded_shape, norm_mean=norm_mean, norm_std=norm_std,
                         cache_dir=cache_dir)

    train_loader = make_loader(train_ids, shuffle=True,  **loader_kwargs)
    val_loader   = make_loader(val_ids,   shuffle=False, **loader_kwargs)

    model = Autoencoder(padded_shape, cfg.latent_dim).to(device)
    logger.info("latent_dim=%d | parameters: %d", cfg.latent_dim, sum(p.numel() for p in model.parameters()))

    sched_tag = "cosine" if cfg.use_lr_scheduler else "flat"
    wandb.init(
        project=cfg.wandb_project,
        name=f"latent{cfg.latent_dim}_{cfg.normalisation}_{sched_tag}lr",
        config={
            "latent_dim":            cfg.latent_dim,
            "normalisation":         cfg.normalisation,
            "use_lr_scheduler":      cfg.use_lr_scheduler,
            "early_stopping_delta":  cfg.early_stopping_delta,
            "lr":                    cfg.lr,
            "weight_decay":          cfg.weight_decay,
            "batch_size":            cfg.batch_size,
            "patience":              cfg.patience,
            "checkpoint_every":      cfg.checkpoint_every,
        },
    )

    history = train(
        model, train_loader, val_loader, device, cfg.checkpoints_dir, cfg.model_name,
        epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        patience=cfg.patience, recon_dataset=val_loader.dataset,
        use_lr_scheduler=cfg.use_lr_scheduler,
        early_stopping_delta=cfg.early_stopping_delta,
        checkpoint_every=cfg.checkpoint_every,
    )

    history_path = Path(cfg.history_dir) / f"{cfg.run_tag}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(history, open(history_path, "w"))

    # model.load_state_dict(torch.load(save_path, map_location=device))

    # codes_dir = Path(cfg.latent_codes_dir(latent_dim))
    # codes_dir.mkdir(parents=True, exist_ok=True)

    # for split_name, ids in [("train", train_ids), ("val", val_ids),
    #                         ("test", test_ids), ("all", all_ids)]:
    #     loader = make_loader(ids, shuffle=False, **loader_kwargs)
    #     codes  = encode_dataset(model, loader, device).numpy()
    #     np.save(codes_dir / f"{split_name}.npy",     codes)
    #     np.save(codes_dir / f"{split_name}_ids.npy", np.array(ids))
    #     logger.info("Saved %s codes: %s", split_name, codes.shape)

    wandb.finish()

if __name__ == "__main__":
    main()