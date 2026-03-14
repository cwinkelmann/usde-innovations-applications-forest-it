"""Segmentation model training (U-Net / Segformer) for wildlife detection."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from wildlife_detection.training.datasets import TileMaskDataset


def compute_iou(pred, target, num_classes):
    """Compute mean IoU across foreground classes.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted class indices, shape (N, H, W).
    target : torch.Tensor
        Ground truth class indices, shape (N, H, W).
    num_classes : int
        Total number of classes (including background).

    Returns
    -------
    float
        Mean IoU over foreground classes.
    """
    ious = []
    for c in range(1, num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    return np.mean(ious) if ious else 0.0


def build_segmentation_model(model_name, backbone, num_classes, device):
    """Build a segmentation model (U-Net or Segformer).

    Parameters
    ----------
    model_name : str
        "unet" or "segformer".
    backbone : str
        Backbone name (e.g. "resnet34" for U-Net, "mit-b2" for Segformer).
    num_classes : int
        Number of output classes.
    device : torch.device
        Target device.

    Returns
    -------
    torch.nn.Module
    """
    if model_name == "unet":
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    elif model_name == "segformer":
        from transformers import SegformerForSemanticSegmentation
        model = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/{backbone}-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def train_segmentation(cfg):
    """Run segmentation training with Weights & Biases logging.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Configuration with keys: data_dir, split_manifest, model, backbone,
        num_classes, imgsz, epochs, batch, lr, wandb.{project, entity, tags}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.wandb.tags),
    )

    data_dir = Path(cfg.data_dir)
    split_df = pd.read_csv(cfg.split_manifest)

    train_files = split_df[split_df["split"] == "train"]["tile_filename"].tolist()
    val_files = split_df[split_df["split"] == "val"]["tile_filename"].tolist()

    train_ds = TileMaskDataset(data_dir / "tiles", data_dir / "masks", train_files, cfg.imgsz, augment=True)
    val_ds = TileMaskDataset(data_dir / "tiles", data_dir / "masks", val_files, cfg.imgsz, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=4)

    model = build_segmentation_model(cfg.model, cfg.backbone, cfg.num_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_iou = 0.0
    is_segformer = cfg.model == "segformer"

    for epoch in range(cfg.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            if is_segformer:
                outputs = model(pixel_values=imgs, labels=masks)
                loss = outputs.loss
            else:
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                if is_segformer:
                    outputs = model(pixel_values=imgs, labels=masks)
                    loss = outputs.loss
                    logits = nn.functional.interpolate(
                        outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                    )
                else:
                    logits = model(imgs)
                    loss = criterion(logits, masks)

                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())

        val_loss /= len(val_loader)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        val_iou = compute_iou(all_preds, all_targets, cfg.num_classes)

        # F1 (foreground only)
        fg_pred = (all_preds > 0)
        fg_target = (all_targets > 0)
        tp = (fg_pred & fg_target).sum().float()
        precision = tp / fg_pred.sum().float() if fg_pred.sum() > 0 else 0.0
        recall = tp / fg_target.sum().float() if fg_target.sum() > 0 else 0.0
        val_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_f1": float(val_f1),
        })

        print(f"Epoch {epoch+1}/{cfg.epochs} — train_loss: {train_loss:.4f}, "
              f"val_loss: {val_loss:.4f}, val_iou: {val_iou:.4f}, val_f1: {float(val_f1):.4f}")

        # Log sample predictions every 10 epochs
        if (epoch + 1) % 10 == 0:
            sample_imgs, sample_masks = next(iter(val_loader))
            sample_imgs = sample_imgs.to(device)
            with torch.no_grad():
                if is_segformer:
                    out = model(pixel_values=sample_imgs)
                    logits = nn.functional.interpolate(
                        out.logits, size=sample_masks.shape[-2:], mode="bilinear", align_corners=False
                    )
                else:
                    logits = model(sample_imgs)
                sample_preds = logits.argmax(dim=1).cpu().numpy()

            images = []
            for i in range(min(4, len(sample_imgs))):
                images.append(wandb.Image(
                    sample_imgs[i].cpu(),
                    masks={
                        "prediction": {"mask_data": sample_preds[i]},
                        "ground_truth": {"mask_data": sample_masks[i].numpy()},
                    },
                ))
            wandb.log({"val/samples": images})

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), data_dir / "best_segmentation.pth")

    wandb.finish()
    print(f"Training complete. Best val IoU: {best_iou:.4f}")
