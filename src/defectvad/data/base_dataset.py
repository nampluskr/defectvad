# defectvad/data/base_datasets.py

import logging
from abc import ABC, abstractmethod
import os
from PIL import Image
import copy

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    NAME = ""    # dataset name: mvtec | visa | btad
    CATEGORIES = []

    def __init__(self, root_dir, category, split, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.category = [category] if isinstance(category, str) else list(category)
        self.category.sort()
        self.split = split
        self.transform = transform or T.ToTensor()
        self.mask_transform = mask_transform or T.ToTensor()
        self.samples = []

        if split == "train":
            self._load_train_samples()
        elif split == "test":
            self._load_test_samples()
        else:
            raise ValueError(f"split must be 'train' or 'test': {split}")

    @abstractmethod
    def _load_train_samples(self):
        raise NotImplementedError

    @abstractmethod
    def _load_test_samples(self):
        raise NotImplementedError

    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        self.height = image.shape[1]
        self.width = image.shape[2]
        return image

    def _load_mask(self, mask_path):
        # if self.split == "train":
        #     return torch.tensor(0.0).float()
        if mask_path is None:
            return torch.zeros((1, self.height, self.width)).long()

        mask = Image.open(mask_path).convert('L')
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = (mask > 0).long()
        return mask

    def count_category(self, category):
        return sum(sample["category"] == category for sample in self.samples)

    def count_normal(self, category=None):
        if category is None:
            return sum(sample["label"] == 0 for sample in self.samples)
        return sum(sample["label"] == 0 and sample["category"] == category for sample in self.samples)

    def count_anomaly(self, category=None):
        if category is None:
            return sum(sample["label"] == 1 for sample in self.samples)
        return sum(sample["label"] == 1 and sample["category"] == category for sample in self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "image": self._load_image(sample["image_path"]),
            "label": torch.tensor(sample["label"]).long(),
            "defect_type": sample["defect_type"],
            "mask": self._load_mask(sample["mask_path"]),
            "dataset": self.NAME,
            "category": sample["category"],
            "filename": os.path.basename(sample['image_path']),
        }

    def info(self):
        logger.info("")
        logger.info(f"*** {self.split.capitalize()} dataset: {self.NAME} (total {len(self)})")
        for category in self.category:
            logging.info(f" > {category + ':':11} {self.count_category(category):4d} "
                  f"(normal {self.count_normal(category):3d}, "
                  f"anomaly {self.count_anomaly(category):3d})"
            )
        return self

    def subset(self, category=None, label=None, defect_type=None):
        subset = copy.copy(self)
        samples = self.samples

        if category is not None:
            samples = [s for s in samples if s["category"] == category]
            subset.category = [category]
        else:
            subset.category = list(self.category)

        if label is not None:
            samples = [s for s in samples if s["label"] == label]

        if defect_type is not None:
            samples = [s for s in samples if s["defect_type"] == defect_type]

        subset.samples = samples
        return subset
