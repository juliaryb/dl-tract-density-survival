"""Train one autoencoder for a given latent dim. Called once per SLURM array task."""
import argparse
import json
import logging
from pathlib import Path
import os

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


def make_loader(ids, id_to_cohort, cfg, bbox, padded_shape, norm_mean, norm_std, shuffle):
    ds = build_dataset_from_ids(
        ids, id_to_cohort, cfg, bbox, padded_shape,
        norm_mean, norm_std, cfg.normalisation,
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, pin_memory=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, required=True)
    args = parser.parse_args()
    latent_dim = args.latent_dim

    cfg    = Config()
    device = cfg.device


    stats_path = os.path.join(cfg.jsons_dir, "preprocessing_stats.json")
    stats        = load_preprocessing_stats(stats_path)
    bbox         = stats["bbox"]
    padded_shape = stats["padded_shape"]
    norm_mean    = stats["norm_mean"]
    norm_std     = stats["norm_std"]

    splits = load_splits(cfg.splits_dir)
    train_ids = splits["train"]
    val_ids   = splits["val"]
    test_ids  = splits["test"]
    all_ids   = train_ids + val_ids + test_ids

    with open(f"{cfg.jsons_dir}/id_to_cohort.json") as f:
        id_to_cohort = json.load(f)

    loader_kwargs = dict(id_to_cohort=id_to_cohort, cfg=cfg, bbox=bbox,
                        padded_shape=padded_shape, norm_mean=norm_mean, norm_std=norm_std)

    train_loader = make_loader(train_ids, shuffle=True,  **loader_kwargs)
    val_loader   = make_loader(val_ids,   shuffle=False, **loader_kwargs)

    model = Autoencoder(padded_shape, latent_dim).to(device)
    logger.info("latent_dim=%d | parameters: %d", latent_dim, sum(p.numel() for p in
model.parameters()))

    wandb.init(
        project=cfg.wandb_project,
        name=f"latent{latent_dim}",
        config={"latent_dim": latent_dim, "normalisation": cfg.normalisation,
                "lr": cfg.lr, "weight_decay": cfg.weight_decay, "batch_size":
cfg.batch_size},
    )

    # TODO: add a tqqm to have an output from training that can be observed in the .err file on athena
    history = train(
        model, train_loader, val_loader, device, cfg.checkpoints_dir, cfg.model_name,
        epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        patience=cfg.patience, recon_dataset=val_loader.dataset
    )

    # TODO: maybe a file system per experiment idk
    # TODO: print the used device and maybe params 
    # json.dump(history, open(f"{cfg.jsons_dir}/history.json", "w"))

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