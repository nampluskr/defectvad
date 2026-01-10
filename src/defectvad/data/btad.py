# defectvad/data/btad.py

import os
from glob import glob

from .base_dataset import BaseDataset


class BTADDataset(BaseDataset):
    NAME = "btad"
    CATEGORIES = ['01', '02', '03']

    def _load_train_samples(self):
        normal_dir = os.path.join(self.category_dir, "train", "ok")
        for image_path in sorted(glob(os.path.join(normal_dir, "*.*"))):
            # ext = os.path.splitext(image_path)[1].lower()
            # if ext in ("png", "jpg", "bmp"):
            self.samples.append({
                "image_path": image_path,
                "label": 0,
                "defect_type": "normal",
                "mask_path": None
            })

    def _load_test_samples(self):
        normal_dir = os.path.join(self.category_dir, "test", "ok")
        anomaly_dir = os.path.join(self.category_dir, "test", "ko")
        mask_dir = os.path.join(self.category_dir, "ground_truth", "ko")

        for image_path in sorted(glob(os.path.join(normal_dir, "*.*"))):
            # ext = os.path.splitext(image_path)[1].lower()
            # if ext in ("png", "jpg", "bmp"):
            self.samples.append({
                "image_path": image_path,
                "label": 0,
                "defect_type": "normal",
                "mask_path": None
            })

        for image_path in sorted(glob(os.path.join(anomaly_dir, "*.*"))):
            # ext = os.path.splitext(image_path)[1].lower()
            # if ext in ("png", "jpg", "bmp"):
            image_name = os.path.basename(image_path)
            mask_name = os.path.splitext(image_name)[0] + ".png"
            mask_path = os.path.join(mask_dir, mask_name)

            self.samples.append({
                "image_path": image_path,
                "label": 1,
                "defect_type": "anomaly",
                "mask_path": mask_path
            })