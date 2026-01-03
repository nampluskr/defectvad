# common/base_trainer.py

from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn


class BaseModel(ABC):
    def __init__(self, model, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save(self, weights_path):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        torch.save(self.model.state_dict(), weights_path)

    def load(self, weights_path, strict=True):
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=strict)

    def predict_step(self, images):
        images = images.to(self.device)
        predictions = self.model(images)
        return {"image": images, **predictions}
