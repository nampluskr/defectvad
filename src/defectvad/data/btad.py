# defectvad/data/btad.py

import os
from glob import glob

from .base_dataset import BaseDataset


class BTADDataset(BaseDataset):
    NAME = "btad"
    CATEGORIES = ['01', '02', '03']

    def _load_train_samples(self):
        self.samples = []
        for category in self.category:
            assert category in self.CATEGORIES

            category_dir = os.path.join(self.root_dir, category)
            normal_dir = os.path.join(category_dir, "train", "ok")
            valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
            for image_path in sorted(glob(os.path.join(normal_dir, "*.*"))):
                if os.path.splitext(image_path)[1].lower() not in valid_exts:
                    continue

                self.samples.append({
                    "category": category,
                    "image_path": image_path,
                    "label": 0,
                    "defect_type": "normal",
                    "mask_path": None
                })

    def _load_test_samples(self):
        self.samples = []
        for category in self.category:
            assert category in self.CATEGORIES

            category_dir = os.path.join(self.root_dir, category)
            normal_dir = os.path.join(category_dir, "test", "ok")
            anomaly_dir = os.path.join(category_dir, "test", "ko")
            mask_dir = os.path.join(category_dir, "ground_truth", "ko")
            valid_exts = (".png", ".jpg", ".jpeg", ".bmp")

            for image_path in sorted(glob(os.path.join(normal_dir, "*.*"))):
                if os.path.splitext(image_path)[1].lower() not in valid_exts:
                    continue

                self.samples.append({
                    "category": category,
                    "image_path": image_path,
                    "label": 0,
                    "defect_type": "normal",
                    "mask_path": None
                })

            for image_path in sorted(glob(os.path.join(anomaly_dir, "*.*"))):
                if os.path.splitext(image_path)[1].lower() not in valid_exts:
                    continue

                image_name = os.path.basename(image_path)
                mask_name = os.path.splitext(image_name)[0]
                mask_ext = ".png" if category in ('01', '02') else ".bmp"
                mask_path = os.path.join(mask_dir, mask_name + mask_ext)

                self.samples.append({
                    "category": category,
                    "image_path": image_path,
                    "label": 1,
                    "defect_type": "anomaly",
                    "mask_path": mask_path if os.path.exists(mask_path) else None,
                })