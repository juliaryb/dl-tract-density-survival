import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb

logger = logging.getLogger(__name__)


def _run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = F.mse_loss(recon, batch)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

    return total_loss / len(loader)


def _log_reconstructions(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    epoch: int,
    n: int = 4,
) -> None:
    model.eval()
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    fig, axes = plt.subplots(n, 2, figsize=(6, n * 2.5))
    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Reconstruction")

    with torch.no_grad():
        for row, idx in enumerate(indices):
            vol = dataset[idx].unsqueeze(0).to(device)
            recon, _ = model(vol)
            mid = vol.shape[2] // 2
            inp = vol[0, 0, mid].cpu().numpy()
            rec = recon[0, 0, mid].cpu().numpy()
            axes[row, 0].imshow(inp, cmap="hot", vmin=inp.min(), vmax=inp.max())
            axes[row, 1].imshow(rec, cmap="hot", vmin=inp.min(), vmax=inp.max())
            for ax in axes[row]:
                ax.axis("off")

    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    wandb.log({"reconstructions": wandb.Image(fig)}, step=epoch)
    plt.close(fig)
    model.train()


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str,
    epochs: int = 300,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    recon_dataset: torch.utils.data.Dataset | None = None,
    log_image_every: int = 10,
) -> dict[str, list[float]]:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_loader, device, optimizer)
        val_loss   = _run_epoch(model, val_loader,   device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        wandb.log({
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         scheduler.get_last_lr()[0],
            "epoch":      epoch,
        }, step=epoch)

        if recon_dataset is not None and epoch % log_image_every == 0:
            _log_reconstructions(model, recon_dataset, device, epoch)

        if epoch % 10 == 0:
            logger.info("Epoch %3d | train %.4f | val %.4f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    wandb.run.summary["best_val_loss"] = best_val_loss
    artifact = wandb.Artifact("autoencoder-best", type="model")
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)

    return history


def encode_dataset(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Encode all subjects to latent vectors. Returns (N, latent_dim) on
CPU."""
    model.eval()
    codes = []
    with torch.no_grad():
        for batch in loader:
            codes.append(model.encode(batch.to(device)).cpu())
    return torch.cat(codes)

# """
# Training loop, evaluation, and latent encoding for the TDMap autoencoder.

# W&B logging summary
# --------------------
# Per epoch : train_loss, val_loss, lr, grad_norm
# Every N   : reconstruction image panels (input vs recon, axial mid-slice)
# End of run: best_val_loss as run summary; best checkpoint uploaded as artifact
# """

# import logging
# import random
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# import wandb

# logger = logging.getLogger(__name__)


# # ---------------------------------------------------------------------------
# # Loss
# # ---------------------------------------------------------------------------

# def mse_loss(
#     pred: torch.Tensor,
#     target: torch.Tensor
# ) -> torch.Tensor:
#     """
#     Mean squared error between the original TDMap and reconstruction.
#     (If normalization was used for dataset creation then these normalized
#     images are considered.)
#     """

#     return ((pred - target)**2).mean()


# # ---------------------------------------------------------------------------
# # Epoch helpers
# # ---------------------------------------------------------------------------

# def _run_epoch(
#     model: nn.Module,
#     loader: torch.utils.data.DataLoader,
#     brain_mask: torch.Tensor,
#     device: torch.device,
#     optimizer: torch.optim.Optimizer | None = None,
# ) -> tuple[float, float | None]:
#     """
#     Run one full pass over a DataLoader.

#     optimizer=None activates evaluation mode (no gradient updates).

#     Returns
#     -------
#     (mean_loss, grad_norm) - grad_norm is None in eval mode.
#     """
#     is_train = optimizer is not None
#     model.train(is_train)

#     total_loss = 0.0
#     total_gnorm = 0.0
#     ctx = torch.enable_grad() if is_train else torch.no_grad()

#     with ctx:
#         for batch in loader:
#             batch = batch.to(device)
#             recon, _ = model(batch)
#             loss = mse_loss(recon, batch)

#             if is_train:
#                 optimizer.zero_grad()
#                 loss.backward()
#                 # prevent large weight updates that cause noisy outputs
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 gnorm = sum(
#                     p.grad.norm() ** 2
#                     for p in model.parameters()
#                     if p.grad is not None
#                 ) ** 0.5
#                 total_gnorm += gnorm.item()
#                 optimizer.step()

#             total_loss += loss.item()

#     n = len(loader)
#     grad_norm = total_gnorm / n if is_train else None
#     return total_loss / n, grad_norm


# # ---------------------------------------------------------------------------
# # W&B reconstruction images
# # ---------------------------------------------------------------------------

# def _log_reconstruction_images(
#     model: nn.Module,
#     dataset: torch.utils.data.Dataset,
#     brain_mask: torch.Tensor,
#     device: torch.device,
#     epoch: int,
#     n: int = 4,
# ) -> None:
#     """
#     Log a grid of input vs reconstruction axial mid-slices to W&B.

#     Samples n random subjects, runs a forward pass, and logs the result
#     as a W&B Image panel under the key 'reconstructions'.
#     """
#     model.eval()
#     indices = random.sample(range(len(dataset)), min(n, len(dataset)))

#     fig, axes = plt.subplots(n, 2, figsize=(6, n * 2.5))
#     axes[0, 0].set_title("Input", fontsize=9)
#     axes[0, 1].set_title("Reconstruction", fontsize=9)

#     with torch.no_grad():
#         for row, idx in enumerate(indices):
#             vol = dataset[idx].unsqueeze(0).to(device)
#             recon, _ = model(vol)

#             mid = vol.shape[2] // 2
#             inp = vol  [0, 0, mid].cpu().numpy()
#             rec = recon[0, 0, mid].cpu().numpy()
#             vmin, vmax = inp.min(), inp.max()

#             axes[row, 0].imshow(inp, cmap="hot", vmin=vmin, vmax=vmax)
#             axes[row, 1].imshow(rec, cmap="hot", vmin=vmin, vmax=vmax)
#             for ax in axes[row]:
#                 ax.axis("off")

#     plt.suptitle(f"Epoch {epoch}", fontsize=10)
#     plt.tight_layout()
#     wandb.log({"reconstructions": wandb.Image(fig)}, step=epoch)
#     plt.close(fig)
#     model.train()


# # ---------------------------------------------------------------------------
# # Main training function
# # ---------------------------------------------------------------------------

# def train(
#     model: nn.Module,
#     train_loader: torch.utils.data.DataLoader,
#     val_loader: torch.utils.data.DataLoader,
#     brain_mask: torch.Tensor,
#     device: torch.device,
#     save_path: str,
#     epochs: int = 300,
#     lr: float = 1e-4,
#     weight_decay: float = 1e-4,
#     patience: int = 20,
#     recon_dataset: torch.utils.data.Dataset | None = None,
#     log_image_every: int = 10,
# ) -> dict[str, list[float]]:
#     """
#     Train the autoencoder with early stopping on validation loss.

#     Saves the best model weights (lowest val loss) to `save_path` and
#     uploads them as a W&B Model artifact at the end of the run.

#     Parameters
#     ----------
#     model        : Autoencoder instance, already moved to `device`.
#     train_loader : DataLoader for training subjects.
#     val_loader   : DataLoader for validation subjects.
#     brain_mask   : (1, D, H, W) float tensor — will be moved to `device`.
#     device       : torch.device to run on.
#     save_path    : Path to save best model weights (.pt file).
#     epochs       : Maximum number of training epochs.
#     lr           : AdamW learning rate.
#     weight_decay : AdamW weight decay (L2 regularisation).
#     patience     : Early stopping patience (epochs without val improvement).
#     recon_dataset   : Optional dataset for logging reconstruction images.
#                       Pass None to skip image logging.
#     log_image_every : Log reconstruction panels every this many epochs.

#     Returns
#     -------
#     history dict with keys "train_loss" and "val_loss" (lists of per-epoch values).
#     """
#     brain_mask = brain_mask.to(device)
#     Path(save_path).parent.mkdir(parents=True, exist_ok=True)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

#     best_val_loss    = float("inf")
#     patience_counter = 0
#     history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

#     for epoch in range(epochs):
#         train_loss, grad_norm = _run_epoch(model, train_loader, brain_mask, device, optimizer)
#         val_loss,   _         = _run_epoch(model, val_loader,   brain_mask, device)
#         scheduler.step()

#         history["train_loss"].append(train_loss)
#         history["val_loss"].append(val_loss)

#         wandb.log({
#             "train_loss": train_loss,
#             "val_loss":   val_loss,
#             "lr":         scheduler.get_last_lr()[0],
#             "grad_norm":  grad_norm,
#             "epoch":      epoch,
#         }, step=epoch)

#         if recon_dataset is not None and epoch % log_image_every == 0:
#             _log_reconstruction_images(model, recon_dataset, brain_mask, device, epoch)

#         if epoch % 10 == 0:
#             logger.info("Epoch %3d | train %.4f | val %.4f", epoch, train_loss, val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss    = val_loss
#             patience_counter = 0
#             torch.save(model.state_dict(), save_path)
#             logger.info("  -> new best val loss %.4f - model saved", best_val_loss)
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
#                 break

#     logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    
#     wandb.run.summary["best_val_loss"] = best_val_loss

#     artifact = wandb.Artifact("autoencoder-best", type="model")
#     artifact.add_file(save_path)
#     wandb.log_artifact(artifact)
#     logger.info("Best checkpoint uploaded to W&B as artifact 'autoencoder-best'.")

#     return history


# # ---------------------------------------------------------------------------
# # Encoding (used after training to produce latent codes for Cox model)
# # ---------------------------------------------------------------------------

# def encode_dataset(
#     model: nn.Module,
#     loader: torch.utils.data.DataLoader,
#     device: torch.device,
# ) -> torch.Tensor:
#     """
#     Encode all subjects in a DataLoader into latent vectors.

#     Use this after training to produce the feature matrix for the Cox model.
#     TODO: The sentence below might be wrong? Check and correct.
#     Call separately for train, val, and test — never mix them.

#     Returns
#     -------
#     Float tensor of shape (N, latent_dim) on CPU.
#     """
#     model.eval()
#     codes = []
#     with torch.no_grad():
#         for batch in loader:
#             z = model.encode(batch.to(device))
#             codes.append(z.cpu())
#     return torch.cat(codes, dim=0)
