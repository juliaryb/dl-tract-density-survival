import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
import os

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
            # loss = F.mse_loss(recon, batch)
            loss = ((recon - batch) ** 2).mean()
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
    checkpoints_dir: str,
    model_name: str,
    epochs: int = 300,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    recon_dataset: torch.utils.data.Dataset | None = None,
    log_image_every: int = 10,
    use_lr_scheduler: bool = True,
    early_stopping_delta: float = 1e-4,
    checkpoint_every: int = 10,
) -> dict[str, list[float]]:
    save_path = os.path.join(checkpoints_dir, model_name)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if use_lr_scheduler else None
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    logger.info("Training on device: %s | model device: %s", device,
    next(model.parameters()).device)

    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_loader, device, optimizer)
        val_loss   = _run_epoch(model, val_loader,   device)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        wandb.log({
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         scheduler.get_last_lr()[0] if scheduler is not None else lr,
            "epoch":      epoch,
        }, step=epoch)

        if recon_dataset is not None and epoch % log_image_every == 0:
            _log_reconstructions(model, recon_dataset, device, epoch)

        if epoch % 10 == 0:
            logger.info("Epoch %3d | train %.4f | val %.4f", epoch, train_loss, val_loss)

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            periodic_path = Path(save_path).parent / f"epoch{epoch+1}.pt"
            torch.save(model.state_dict(), periodic_path)

        if val_loss < best_val_loss - early_stopping_delta:
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
    return torch.cat(codes) # concatenation of codes along the first dim (batch size)
