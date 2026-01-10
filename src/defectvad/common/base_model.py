# defectvad/common/base_trainer.py

import logging
from abc import ABC
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class BaseModel(ABC):
    def __init__(self, model, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.name = model.__class__.__name__

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, data):
        return self.predict(data)

    def predict(self, data):
        if isinstance(data, torch.Tensor):
            return self.predict_images(data)
        if isinstance(data, dict):
            return self.predict_batch(data)
        if isinstance(data, DataLoader):
            return self.predict_dataloader(data)
        raise TypeError(f"Unsupported input type: {type(data)}")

    @torch.no_grad()
    def predict_images(self, images):
        self.model.eval()
        images = images.to(self.device)
        predictions = self.model(images)
        outputs = {"image": images, **predictions}
        return {k: v.cpu() if torch.is_tensor(v) else v for k, v in outputs.items()}

    @torch.no_grad()
    def predict_batch(self, batch):
        self.model.eval()
        images = batch["image"].to(self.device)
        predictions = self.model(images)
        outputs = {**batch, **predictions}
        return {k: v.cpu() if torch.is_tensor(v) else v for k, v in outputs.items()}

    @torch.no_grad()
    def predict_dataloader(self, dataloader):
        outputs = {}
        with tqdm(dataloader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(">> Predicting")

            for batch in progress_bar:
                batch_outputs = self.predict_batch(batch)

                for key, value in batch_outputs.items():
                    outputs.setdefault(key, []).append(value)

        for key, value in outputs.items():
            if torch.is_tensor(value[0]):
                outputs[key] = torch.cat(value, dim=0)
            else:
                outputs[key] = value

        return outputs

    def save(self, weights_path):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        torch.save(self.model.state_dict(), weights_path)
        logger.info(f" > {self.name} weights is saved to {os.path.basename(weights_path)}")

    def load(self, weights_path, strict=True):
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=strict)
        logger.info(f" > {self.name} weights is loaded from {os.path.basename(weights_path)}")

    def info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        total_bytes = 0
        for tensor in self.model.state_dict().values():
            total_bytes += tensor.numel() * tensor.element_size()

        if total_bytes >= 1024 ** 3:
            size_value = total_bytes / (1024 ** 3)
            size_unit = "GB"
        else:
            size_value = total_bytes / (1024 ** 2)
            size_unit = "MB"

        logger.info("")
        logging.info(f"*** {self.name}:")
        logger.info(f" > Total params.:     {total_params:,}")
        logger.info(f" > Trainable params.: {trainable_params:,}")
        logger.info(f" > Model size (disk): {size_value:.2f} {size_unit}")
        return self