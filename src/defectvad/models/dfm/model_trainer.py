# src/defectvad/models/dfm/model_trainer.py

from tqdm import tqdm
import torch
import torch.optim as optim

from defectvad.common.base_model import BaseModel
from defectvad.common.base_trainer import BaseTrainer
from .torch_model import DFMModel


class DFM(BaseModel):
    def __init__(self, backbone="resnet50", layer="layer3", 
        pooling_kernel_size=4, pca_level=0.97, score_type="fre"):

        model = DFMModel(
            backbone=backbone,
            layer=layer,
            pooling_kernel_size=pooling_kernel_size,
            n_comps=pca_level,
            score_type=score_type,
        )
        super().__init__(model)


class DFMTrainer(BaseTrainer):
    def __init__(self, model):
        if not isinstance(model, DFM):
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