# defectvad/data/mvtec.py

import os
from glob import glob

from .base_dataset import BaseDataset


class MVTecDataset(BaseDataset):
    NAME = "mvtec"
    CATEGORIES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    def _load_train_samples(self):
        self.samples = []
        for category in self.category:
            assert category in self.CATEGORIES

            category_dir = os.path.join(self.root_dir, category)
            normal_dir = os.path.join(category_dir, "train", "good")
            for image_path in sorted(glob(os.path.join(normal_dir, "*.png"))):
                self.samples.append({
                    "category": category,
                    "image_path": image_path,
                    "label": 0,
                    "defect_type": "normal",
                    "mask_path": None,
                })

    def _load_test_samples(self):
        self.samples = []
        for category in self.category:
            assert category in self.CATEGORIES

            category_dir = os.path.join(self.root_dir, category)
            test_dir = os.path.join(category_dir, "test")
            mask_dir = os.path.join(category_dir, "ground_truth")

            for defect_type in sorted(os.listdir(test_dir)):
                for image_path in sorted(glob(os.path.join(test_dir, defect_type, "*.png"))):

                    if defect_type == "good":
                        self.samples.append({
                            "category": category,
                            "image_path": image_path,
                            "label": 0,
                            "defect_type": "normal",
                            "mask_path": None
                        })
                    else:
                        image_name = os.path.basename(image_path)
                        mask_name = os.path.splitext(image_name)[0] + "_mask.png"
                        mask_path = os.path.join(mask_dir, defect_type, mask_name)

                        self.samples.append({
                            "category": category,
                            "image_path": image_path,
                            "label": 1,
                            "defect_type": defect_type,
                            "mask_path": mask_path if os.path.exists(mask_path) else None,
                        })

