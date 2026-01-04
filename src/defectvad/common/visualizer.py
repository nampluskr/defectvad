# common/predictor.py

import torch
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, preds, image_threshold=None, pixel_threshold=None):
        self.images = preds["image"]
        self.anomaly_maps = preds.get("anomaly_map", None)
        self.labels = preds.get("label", None)
        self.masks = preds.get("mask", None)
        self.pred_scores = preds.get("pred_score", None)

        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold

        if "pred_label" in preds:
            self.pred_labels = preds["pred_label"]
        elif image_threshold is not None:
            self.pred_labels = (self.pred_scores >= image_threshold).long()
        else:
            self.pred_labels = None

        if "pred_mask" in preds:
            self.pred_masks = preds["pred_mask"]
        elif pixel_threshold is not None:
            self.pred_masks = (self.anomaly_maps >= pixel_threshold).long()
        else:
            self.pred_masks = None

    def set_image_threshold(self, threshold):
        self.image_threshold = threshold
        if threshold is not None:
            self.pred_labels = (self.pred_scores >= threshold).long()
        else:
            self.pred_labels = None

    def set_pixel_threshold(self, threshold):
        self.pixel_threshold = threshold
        if threshold is not None:
            self.pred_masks = (self.anomaly_maps >= threshold).long()
        else:
            self.pred_masks = None

    def _get_mask(self, idx):
        return self.masks[idx] if self.masks is not None else None

    def _get_pred_label(self, idx):
        return self.pred_labels[idx] if self.pred_labels is not None else None

    def _get_pred_mask(self, idx):
        return self.pred_masks[idx] if self.pred_masks is not None else None

    def visualize(self, image, anomaly_map=None, label=None, pred_label=None,
                mask=None, pred_mask=None, denormalize=True, cmap="jet", overlay=False):

        image = to_numpy_rgb(image, denormalize=denormalize)
        anomaly_map = to_numpy_gray(anomaly_map) if anomaly_map is not None else None
        label = int(label) if label is not None else None
        pred_label = int(pred_label) if pred_label is not None else None
        mask = to_numpy_gray(mask) if mask is not None else None
        pred_mask = to_numpy_gray(pred_mask) if pred_mask is not None else None

        num_plots = 1
        if anomaly_map is not None:
            num_plots += 1
        if mask is not None:
            num_plots += 1
        if pred_mask is not None:
            num_plots += 1

        fig, axes = plt.subplots(1, num_plots, figsize=(3 * num_plots, 3))
        if num_plots == 1:
            axes = [axes]
        idx = 0

        # axes[0]: Image
        axes[idx].imshow(image)
        title = "Image"
        if label is not None:
            title += f" (label={label})"
        axes[idx].set_title(title)
        idx += 1

        # axes[1]: Anomaly Map
        if anomaly_map is not None:
            if overlay:
                axes[idx].imshow(overlay_map(image, anomaly_map, alpha=0.5, cmap=cmap))
                title = "Anomaly Overlay"
            else:
                axes[idx].imshow(anomaly_map, cmap=cmap)
                title = "Anomaly Map"

            if pred_label is not None:
                title += f" (Pred={pred_label})"
            axes[idx].set_title(title)
            idx += 1

        # axes[2]: GT Mask
        if mask is not None:
            axes[idx].imshow(mask, cmap="gray")
            axes[idx].set_title("GT Mask")
            idx += 1

        # axes[3]: Pred Mask
        if pred_mask is not None:
            axes[idx].imshow(pred_mask, cmap="gray")
            axes[idx].set_title("Pred Mask")
            idx += 1

        for ax in axes:
            ax.axis("off")

        fig.tight_layout()
        plt.show()


    def show_normal(self, max_samples=2, denormalize=True):
        cnt = 0
        for i in range(self.images.shape[0]):
            if self.labels is not None and self.labels[i] == 0:
                cnt += 1
                self.visualize(
                    self.images[i],
                    self.anomaly_maps[i],
                    label=self.labels[i],
                    pred_label=self._get_pred_label(i),
                    mask=self._get_mask(i),
                    pred_mask=self._get_pred_mask(i),
                    denormalize=denormalize,
                    cmap="jet",
                    overlay=True
                )
            if cnt == max_samples:
                break

    def show_anomaly(self, max_samples=2, denormalize=True):
        cnt = 0
        for i in range(self.images.shape[0]):
            if self.labels is not None and self.labels[i] == 1:
                cnt += 1
                self.visualize(
                    self.images[i],
                    self.anomaly_maps[i],
                    label=self.labels[i],
                    pred_label=self._get_pred_label(i),
                    mask=self._get_mask(i),
                    pred_mask=self._get_pred_mask(i),
                    denormalize=denormalize,
                    cmap="jet",
                    overlay=True
                )
            if cnt == max_samples:
                break


#####################################################################
# Helper functions
#####################################################################

def to_numpy_rgb(tensors, denormalize=False):
    if tensors.dim() == 4 and tensors.shape[1] == 3:
        tensors = tensors.permute(0, 2, 3, 1)   # (B, 3, H, W) -> (B, H, W, 3)
    elif tensors.dim() == 3 and tensors.shape[0] == 3:
        tensors = tensors.permute(1, 2, 0)      # (3, H, W) -> (H, W, 3)
    else:
        raise ValueError(f"Expected (B, 3, H, W) or (3, H, W), got {tensors.shape}")

    images = tensors.numpy()

    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        images = images * std + mean

    return np.clip(images, 0, 1)

def to_numpy_gray(tensors):
    if tensors.dim() == 4 and tensors.shape[1] == 1:
        tensors = tensors.squeeze(1)    # (B, 1, H, W) - > (B, H, W)
    elif tensors.dim() == 3 and tensors.shape[0] == 1:
        tensors = tensors.squeeze(0)    # (1, H, W) -> (H, W)
    elif tensors.dim() == 3:
        pass                            # already (B, H, W)
    elif tensors.dim() == 2:
        pass                            # already (H, W)
    else:
        raise ValueError(f"Expected (B, 1, H, W), (B, H, W), (1, H, W), or (H, W), got {tensors.shape}")

    images = tensors.numpy()
    return np.clip(images, 0, 1)

def overlay_map(image, map, alpha=0.5, cmap="jet"):
    map_norm = (map - map.min()) / (map.max() - map.min() + 1e-8)
    heatmap = plt.get_cmap(cmap)(map_norm)[..., :3]
    overlay = (1 - alpha) * image + alpha * heatmap
    return np.clip(overlay, 0, 1)
