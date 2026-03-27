"""PyTorch datasets for segmentation and point-based detection."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from wildlife_detection.utils.density import generate_fidt_map


class TileMaskDataset(Dataset):
    """Dataset for semantic segmentation: loads JPEG tiles and PNG masks.

    Parameters
    ----------
    tile_dir : str or Path
        Directory containing JPEG tile images.
    mask_dir : str or Path
        Directory containing PNG mask images.
    filenames : list of str
        Tile filenames to include.
    imgsz : int
        Target image size (square).
    augment : bool
        Whether to apply data augmentation (horizontal flip, color jitter).
    """

    def __init__(self, tile_dir, mask_dir, filenames, imgsz, augment=False):
        self.tile_dir = Path(tile_dir)
        self.mask_dir = Path(mask_dir)
        self.filenames = filenames
        self.imgsz = imgsz
        self.augment = augment

        self.img_transform = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2,
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(self.tile_dir / fname).convert("RGB")
        mask_fname = fname.replace(".jpg", ".png")
        mask = Image.open(self.mask_dir / mask_fname)

        if self.augment and torch.rand(1).item() > 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)

        if self.augment:
            img = self.color_jitter(img)

        img = self.img_transform(img)
        mask = torch.from_numpy(
            np.array(mask.resize((self.imgsz, self.imgsz), Image.NEAREST))
        ).long()

        return img, mask


class HerdNetDataset(Dataset):
    """Dataset for HerdNet: loads JPEG tiles and generates FIDT density maps on-the-fly.

    Parameters
    ----------
    tile_dir : str or Path
        Directory containing JPEG tile images.
    annotations_df : pandas.DataFrame
        Point annotations with columns: tile_filename, local_x, local_y.
    tile_filenames : list of str
        Tile filenames to include.
    patch_size : int
        Target patch size (square).
    down_ratio : int
        Downsampling ratio for the FIDT map (default 2).
    fidt_radius : int
        Radius of foreground markers in the FIDT map (default 1).
    augment : bool
        Whether to apply data augmentation (horizontal/vertical flip).
    """

    def __init__(self, tile_dir, annotations_df, tile_filenames, patch_size,
                 down_ratio=2, fidt_radius=1, augment=False):
        self.tile_dir = Path(tile_dir)
        self.patch_size = patch_size
        self.down_ratio = down_ratio
        self.fidt_radius = fidt_radius
        self.augment = augment
        self.filenames = tile_filenames
        self.annotations = annotations_df

        self.img_transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(self.tile_dir / fname).convert("RGB")

        tile_annots = self.annotations[self.annotations["tile_filename"] == fname]
        points = list(zip(tile_annots["local_x"].values, tile_annots["local_y"].values))

        if self.augment and torch.rand(1).item() > 0.5:
            img = transforms.functional.hflip(img)
            points = [(self.patch_size - x, y) for x, y in points]

        if self.augment and torch.rand(1).item() > 0.5:
            img = transforms.functional.vflip(img)
            points = [(x, self.patch_size - y) for x, y in points]

        img = self.img_transform(img)

        map_h = self.patch_size // self.down_ratio
        map_w = self.patch_size // self.down_ratio
        scaled_points = [(x / self.down_ratio, y / self.down_ratio) for x, y in points]
        fidt_map = generate_fidt_map(scaled_points, map_h, map_w, self.fidt_radius)
        fidt_map = torch.from_numpy(fidt_map).unsqueeze(0)

        return img, fidt_map, len(points)
