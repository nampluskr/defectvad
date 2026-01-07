# src/defectvad/models/supersimplenet/trainer.py

import torch
import torch.optim as optim

from defectvad.common.base_model import BaseModel
from defectvad.common.base_trainer import BaseTrainer
from .torch_model import SupersimplenetModel
from .loss import SSNLoss


class Supersimplenet(BaseModel):
    def __init__(self, perlin_threshold=0.2, backbone="wide_resnet50_2.tv_in1k",
        layers=["layer2", "layer3"], adapt_cls_features=False, supervised=False):

        model = SupersimplenetModel(
            perlin_threshold=perlin_threshold,
            backbone=backbone,
            layers=layers,
            stop_grad=False if supervised else True,
            adapt_cls_features=adapt_cls_features,
        )
        super().__init__(model)


class SupersimplenetTrainer(BaseTrainer):
    def __init__(self, model, supervised=False):
        if not isinstance(model, Supersimplenet):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

        loss_fn = SSNLoss()
        super().__init__(model, loss_fn=loss_fn)
        self.norm_clip_val = 1 if supervised else 0

    def configure_optimizers(self):
        self.optimizer = optim.AdamW([
                {"params": self.model.adaptor.parameters(), "lr": 0.0001},
                {"params": self.model.segdec.parameters(), "lr": 0.0002, "weight_decay": 0.00001},
        ])
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(self.max_epochs * 0.8), int(self.max_epochs * 0.9)],
            gamma=0.4,
        )
        self.gradient_clip_val = self.norm_clip_val

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        # masks = batch["mask"].squeeze(1).to(self.device)
        masks = None
        labels = batch["label"].to(self.device)
        anomaly_map, anomaly_score, masks, labels = self.model(images, masks, labels)
        loss = self.loss_fn(pred_map=anomaly_map, pred_score=anomaly_score, target_mask=masks, target_label=labels)
        return {"loss": loss}