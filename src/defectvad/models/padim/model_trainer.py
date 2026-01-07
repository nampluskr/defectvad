# src/defectvad/models/padim/trainer.py

import torch
import torch.optim as optim

from defectvad.common.base_model import BaseModel
from defectvad.common.base_trainer import BaseTrainer
from .torch_model import PadimModel


class Padim(BaseModel):
    def __init__(self, backbone="resnet50", layers=["layer1", "layer2", "layer3"], n_features=None):

        model = PadimModel(
            backbone=backbone,
            layers=layers,
            n_features=n_features,
        )
        super().__init__(model)


class PadimTrainer(BaseTrainer):
    def __init__(self, model):
        if not isinstance(model, Padim):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

        super().__init__(model, loss_fn=None)

    def on_train_start(self):
        super().on_train_start()
        self.max_epochs = 1

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        _ = self.model(images)
        return {"loss": torch.tensor(0.0).float().to(self.device)}

    def on_train_end(self):
        super().on_train_end()
        self.model.fit()
