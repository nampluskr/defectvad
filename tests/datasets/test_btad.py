# tests/datasets/btad.py

import logging
import os
import sys


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_CIR = os.path.join(PROJECT_DIR, "src")
CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

if SOURCE_CIR not in sys.path:
    sys.path.insert(0, SOURCE_CIR)

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from defectvad.data.btad import BTADDataset
from defectvad.data.transforms import get_image_transform, get_mask_transform
from defectvad.common.utils import set_seed, set_logging
from defectvad.common.visualizer import to_numpy_rgb, to_numpy_gray


def test(category):
    set_seed(42)
    set_logging()
    logger = logging.getLogger(__name__)

    dataset = BTADDataset(
        root_dir="/mnt/d/deep_learning/datasets/btad",
        category=category,
        split="test",
        transform=get_image_transform(img_size=256, normalize=True),
        mask_transform=get_mask_transform(img_size=256),
    ).info()

    dataset = dataset.subset(category='03', label=1).info()

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        num_workers=8
    )

    cnt, max_samples = 1, 5
    for batch in dataloader:
        cnt += 1
        image = to_numpy_rgb(batch["image"], denormalize=True)[0]
        mask = to_numpy_gray(batch["mask"])[0]

        logger.info("")
        logger.info(f" > image: {image.shape}")
        logger.info(f" > mask:  {mask.shape} (min: {mask.min()}, max: {mask.max()})")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(image)
        ax2.imshow(mask, cmap="gray")
        fig.tight_layout()
        plt.show()

        if cnt > max_samples:
            break

if __name__ == "__main__":
    test(category=['01', '02', '03'])
