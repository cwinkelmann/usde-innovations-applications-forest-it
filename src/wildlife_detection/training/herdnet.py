"""HerdNet-style training with FIDT density maps for point-based wildlife detection."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from wildlife_detection.training.datasets import HerdNetDataset


def build_simple_herdnet(num_classes=1):
    """Build a simplified HerdNet-like model using a ResNet-34 encoder.

    For the full DLA-based architecture, install HerdNet from source:
    https://github.com/Alexandre-Delplanque/HerdNet

    Parameters
    ----------
    num_classes : int
        Number of output channels (default 1 for single-class density map).

    Returns
    -------
    torch.nn.Sequential
    """
    import torchvision.models as models

    backbone = models.resnet34(weights="IMAGENET1K_V1")
    encoder = nn.Sequential(*list(backbone.children())[:-2])

    return nn.Sequential(
        encoder,
        nn.Conv2d(512, 256, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 64, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, num_classes, 1),
        nn.Sigmoid(),
    )


def evaluate_herdnet(model, loader, device):
    """Evaluate HerdNet model on a data loader.

    Uses simple connected-component peak detection and count-based matching.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    loader : torch.utils.data.DataLoader
        Validation/test data loader.
    device : torch.device
        Target device.

    Returns
    -------
    tuple of (float, float, float, float)
        (precision, recall, f1, mae)
    """
    from scipy.ndimage import label as ndlabel

    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    total_count_error = 0.0
    n_samples = 0

    with torch.no_grad():
        for imgs, fidt_maps, counts in loader:
            imgs = imgs.to(device)
            preds = model(imgs)

            if preds.shape[-2:] != fidt_maps.shape[-2:]:
                preds = nn.functional.interpolate(
                    preds, size=fidt_maps.shape[-2:], mode="bilinear", align_corners=False,
                )

            for i in range(imgs.size(0)):
                pred_map = preds[i, 0].cpu().numpy()
                gt_count = counts[i].item()

                pred_binary = pred_map > 0.3
                _, n_pred = ndlabel(pred_binary)

                total_count_error += abs(n_pred - gt_count)
                n_samples += 1

                tp = min(n_pred, gt_count)
                fp = max(0, n_pred - gt_count)
                fn = max(0, gt_count - n_pred)

                total_tp += tp
                total_fp += fp
                total_fn += fn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    mae = total_count_error / max(n_samples, 1)

    return precision, recall, f1, mae


def train_herdnet(cfg):
    """Run HerdNet training with Weights & Biases logging.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Configuration with keys: data_dir, split_manifest, patch_size,
        down_ratio, fidt_radius, epochs, batch_size, lr, weight_decay,
        warmup_iters, lr_scheduler.{patience, min_lr},
        wandb.{project, entity, tags}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.wandb.tags),
    )

    data_dir = Path(cfg.data_dir)
    annotations = pd.read_csv(data_dir / "annotations_points.csv")
    split_df = pd.read_csv(cfg.split_manifest)

    train_files = split_df[split_df["split"] == "train"]["tile_filename"].tolist()
    val_files = split_df[split_df["split"] == "val"]["tile_filename"].tolist()

    train_ds = HerdNetDataset(
        data_dir / "tiles", annotations, train_files,
        cfg.patch_size, cfg.down_ratio, cfg.fidt_radius, augment=True,
    )
    val_ds = HerdNetDataset(
        data_dir / "tiles", annotations, val_files,
        cfg.patch_size, cfg.down_ratio, cfg.fidt_radius, augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    model = build_simple_herdnet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg.lr_scheduler.patience, min_lr=cfg.lr_scheduler.min_lr,
    )
    criterion = nn.MSELoss()

    best_f1 = 0.0

    for epoch in range(cfg.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (imgs, fidt_maps, counts) in enumerate(train_loader):
            imgs, fidt_maps = imgs.to(device), fidt_maps.to(device)

            # Linear warmup
            total_iter = epoch * len(train_loader) + batch_idx
            if total_iter < cfg.warmup_iters:
                lr = cfg.lr * (total_iter + 1) / cfg.warmup_iters
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            optimizer.zero_grad()
            preds = model(imgs)

            if preds.shape[-2:] != fidt_maps.shape[-2:]:
                preds = nn.functional.interpolate(
                    preds, size=fidt_maps.shape[-2:], mode="bilinear", align_corners=False,
                )

            loss = criterion(preds, fidt_maps)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, fidt_maps, counts in val_loader:
                imgs, fidt_maps = imgs.to(device), fidt_maps.to(device)
                preds = model(imgs)
                if preds.shape[-2:] != fidt_maps.shape[-2:]:
                    preds = nn.functional.interpolate(
                        preds, size=fidt_maps.shape[-2:], mode="bilinear", align_corners=False,
                    )
                val_loss += criterion(preds, fidt_maps).item()

        val_loss /= len(val_loader)

        precision, recall, f1, mae = evaluate_herdnet(model, val_loader, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/precision": precision,
            "val/recall": recall,
            "val/f1": f1,
            "val/mae": mae,
            "lr": current_lr,
        })

        print(f"Epoch {epoch+1}/{cfg.epochs} — train_loss: {train_loss:.4f}, "
              f"val_loss: {val_loss:.4f}, F1: {f1:.4f}, MAE: {mae:.2f}")

        # Log sample heatmaps every 10 epochs
        if (epoch + 1) % 10 == 0:
            sample_imgs, sample_fidts, sample_counts = next(iter(val_loader))
            sample_imgs = sample_imgs.to(device)
            with torch.no_grad():
                sample_preds = model(sample_imgs)
                if sample_preds.shape[-2:] != sample_fidts.shape[-2:]:
                    sample_preds = nn.functional.interpolate(
                        sample_preds, size=sample_fidts.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )

            images = []
            for i in range(min(4, len(sample_imgs))):
                gt_map = sample_fidts[i, 0].numpy()
                pred_map = sample_preds[i, 0].cpu().numpy()

                gt_vis = (gt_map * 255).astype(np.uint8)
                pred_vis = (np.clip(pred_map, 0, 1) * 255).astype(np.uint8)

                images.append(wandb.Image(sample_imgs[i].cpu(), caption=f"input (count={sample_counts[i].item()})"))
                images.append(wandb.Image(gt_vis, caption="ground truth FIDT"))
                images.append(wandb.Image(pred_vis, caption="predicted heatmap"))

            wandb.log({"val/heatmaps": images})

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), data_dir / "best_herdnet.pth")

    wandb.finish()
    print(f"Training complete. Best val F1: {best_f1:.4f}")
