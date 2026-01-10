# experiments/predict.py

import logging
import os
import sys
from argparse import ArgumentParser
import torch

from utils import set_environment, get_last_file


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_CIR = os.path.join(PROJECT_DIR, "src")
CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

if SOURCE_CIR not in sys.path:
    sys.path.insert(0, SOURCE_CIR)

from defectvad.common.config import load_config
from defectvad.common.utils import set_seed, set_logging
from defectvad.common.factory import create_dataset, create_dataloader, create_model
from defectvad.common.evaluator import Evaluator
from defectvad.common.visualizer import Visualizer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--category", type=str, required=True, nargs="+")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--timestamp", type=str, default=None)

    parser.add_argument("--max_samples", type=int, default=-1)  # visualizer
    parser.add_argument("--calibrate", action="store_true")     # evaluator
    parser.add_argument("--save_anomaly", action="store_true")  # visualizer
    parser.add_argument("--save_normal", action="store_true")   # visualizer
    parser.add_argument("--image_level", action="store_true")   # visualizer
    parser.add_argument("--pixel_level", action="store_true")   # visualizer
    return parser.parse_args()


def predict(dataset, category, model, max_samples, calibrate, 
    save_anomaly, save_normal, image_level, pixel_level, timestamp=None):
    # ===============================================================
    # Load configs
    # ===============================================================

    category = "-".join(sorted(category))
    experiment_dir = os.path.join(OUTPUT_DIR, dataset, category, model)

    if timestamp is None:
        weight_file = get_last_file(experiment_dir, ext="*.pth")
        config_file = get_last_file(experiment_dir, ext="*.yaml")
        log_file = get_last_file(experiment_dir, ext="*.log")
    else:
        weight_file = f"weights_{dataset}_{category}_{model}_{timestamp}.pth"
        config_file = f"configs_{dataset}_{category}_{model}_{timestamp}.yaml"
        config_file = f"train_{dataset}_{category}_{model}_{timestamp}.log"

    set_logging(experiment_dir, log_file)
    logger = logging.getLogger(__name__)
    logger.info(f" > Logging initialized: {log_file}")

    config = load_config(experiment_dir, config_file)
    set_environment(config)
    set_seed(config["seed"])

    # ===============================================================
    # Create Datasets / Dataloaders / Model
    # ===============================================================

    anomaly_model = create_model(config["model"]).info()
    anomaly_model.load(os.path.join(experiment_dir, weight_file))

    train_dataset = create_dataset("train", config["dataset"])
    test_dataset = create_dataset("test", config["dataset"])
    train_loader = create_dataloader(train_dataset, config["train_loader"])
    test_loader = create_dataloader(test_dataset, config["test_loader"])

    # ===============================================================
    # Prediction: test_loader (batch_size=1)
    # ===============================================================

    test_dataset.info()
    logger.info("")
    logger.info("*** Prediction (dataloader):")
    preds = anomaly_model.predict(test_loader)

    for k, v in preds.items():
        logger.info(f" > {k}: {v.shape if torch.is_tensor(v) else len(v)}")

    # ===============================================================
    # Evaluation: test_loader (batch_size=1)
    # ===============================================================

    visualizer = Visualizer(preds)
    evaluator = Evaluator(anomaly_model)

    logger.info("")
    logger.info("*** Evaluation: Set thresholds")

    if calibrate:
        if image_level:
            image_thresholds = evaluator.calibrate_image_thresholds(train_loader)
            visualizer.set_image_threshold(image_thresholds["95%"])

        if pixel_level:
            pixel_thresholds = evaluator.calibrate_pixel_thresholds(train_loader)
            visualizer.set_pixel_threshold(pixel_thresholds["99%"])
    else:
        if image_level:
            image_results = evaluator.evaluate_image_level(test_loader)
            visualizer.set_image_threshold(image_results["th"])

        if pixel_level:
            pixel_results = evaluator.evaluate_pixel_level(test_loader)
            visualizer.set_pixel_threshold(pixel_results["th"])

    # ===============================================================
    # Visualization: test_loader (batch_size=1)
    # ===============================================================

    logger.info("")
    logger.info("*** Visualization: Save anomaly maps")

    if save_anomaly:
        visualizer.save_anomaly(
            save_dir=os.path.join(experiment_dir, "anomaly"), 
            max_samples=max_samples,
            denormalize=config["dataset"]["normalize"], 
        )

    if save_normal:
        visualizer.save_normal(
            save_dir=os.path.join(experiment_dir, "normal"), 
            max_samples=max_samples,
            denormalize=config["dataset"]["normalize"],
        )


if __name__ == "__main__":

    if 1:
        args = parse_args()
        predict(**args.__dict__)
    if 0:
        args = {
            "dataset": "mvtec",
            "category": ["carpet", "grid", "leather", "tile", "wood"],
            "model": "stfpm",
            "max_samples": 10,
            "calibrate": False,
            "save_anomaly": True,
            "save_normal": True,
            "image_level": True,
            "pixel_level": True,
            "timestamp": None,
        }
        predict(**args)
