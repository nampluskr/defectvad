# experiments/train.py

import logging
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

from utils import set_environment, set_logging


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_CIR = os.path.join(PROJECT_DIR, "src")
CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

if SOURCE_CIR not in sys.path:
    sys.path.insert(0, SOURCE_CIR)

from defectvad.common.config import load_config, merge_configs, save_config
from defectvad.common.utils import set_seed
from defectvad.common.factory import create_dataset, create_dataloader
from defectvad.common.factory import create_model, create_trainer
from defectvad.common.evaluator import Evaluator
from defectvad.common.visualizer import Visualizer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--category", type=str, required=True, nargs="+")
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--max_epochs", type=int, default=10)   # trainer
    parser.add_argument("--save_model", action="store_true")    # trainer
    parser.add_argument("--validate", action="store_true")      # trainer
    parser.add_argument("--pixel_level", action="store_true")   # evaluator
    return parser.parse_args()


def update_config(dataset, category, model, max_epochs, validate, save_model, pixel_level):
    config = merge_configs(
        load_config(CONFIG_DIR, "defaults.yaml"),
        load_config(os.path.join(CONFIG_DIR, "datasets"), f"{dataset}.yaml"),
        load_config(os.path.join(CONFIG_DIR, "models"), f"{model}.yaml"),
    )

    # Update conifig values
    config["dataset"]["path"] = config["path"][dataset]
    config["dataset"]["category"] = sorted(category)
    config["trainer"]["max_epochs"] = max_epochs
    config["trainer"]["save_model"] = save_model
    config["trainer"]["validate"] = validate
    config["evaluator"]["pixel_level"] = pixel_level
    
    categories = "-".join(config["dataset"]["category"])
    config["experiment"]["name"] = f"{dataset}_{categories}_{model}"
    config["experiment"]["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["experiment"]["output_dir"] = os.path.join(OUTPUT_DIR, dataset, categories, model)
    return config


def train(config):
    # ===============================================================
    # Load configs
    # ===============================================================

    experiment_dir = config["experiment"]["output_dir"]
    experiment_name = config["experiment"]["name"]
    timestamp = config["experiment"]["timestamp"]

    weight_file = f"weights_{experiment_name}_{timestamp}.pth"
    config_file = f"configs_{experiment_name}_{timestamp}.yaml"
    log_file = f"train_{experiment_name}_{timestamp}.log"

    set_logging(experiment_dir, log_file)
    logger = logging.getLogger(__name__)
    logger.info(f" > Logging initialized: {log_file}")

    set_environment(config)
    set_seed(config["seed"])

    # ===============================================================
    # Create Datasets / Dataloaders / Model / Trainer
    # ===============================================================

    vad = create_model(config["model"])
    trainer = create_trainer(vad, config["trainer"])

    train_dataset = create_dataset("train", config["dataset"])
    test_dataset = create_dataset("test", config["dataset"])

    train_loader = create_dataloader(train_dataset, config["train_loader"])
    test_loader = create_dataloader(test_dataset, config["test_loader"])

    # ===============================================================
    # Training: train_loader
    # ===============================================================

    train_dataset.info()
    max_epochs = config["trainer"]["max_epochs"]

    if config["trainer"]["validate"]:
        trainer.fit(train_loader, max_epochs=max_epochs, valid_loader=test_loader)
    else:
        trainer.fit(train_loader, max_epochs=max_epochs, valid_loader=None)

    if config["trainer"]["save_model"]:
        vad.save(os.path.join(experiment_dir, weight_file))
        save_config(config, os.path.join(experiment_dir, config_file))

    # ===============================================================
    # Evaluation: test_loader (batch_size=1)
    # ===============================================================

    test_dataset.info()
    evaluator = Evaluator(vad)

    logger.info("")
    logger.info("*** Evaluation")
    categories = "-".join(config["dataset"]["category"])
    logger.info(f" > {categories}:")
    image_results = evaluator.evaluate_image_level(test_loader)
    logger.info("   Image-level: " + ", ".join([f"{k}:{v:.3f}" for k, v in image_results.items()]))

    if config["evaluator"]["pixel_level"]:
        pixel_results = evaluator.evaluate_pixel_level(test_loader)
        logger.info("   Pixel-level: " + ", ".join([f"{k}:{v:.3f}" for k, v in pixel_results.items()]))

    categories = config["dataset"]["category"]
    if len(categories) > 1:
        for category in categories:
            test_dataset = test_dataset.subset(category)
            test_loader = create_dataloader(test_dataset, config["test_loader"])

            logger.info(f" > {category}:")
            image_results = evaluator.evaluate_image_level(test_loader)
            logger.info("   Image-level: " + ", ".join([f"{k}:{v:.3f}" for k, v in image_results.items()]))

            if config["evaluator"]["pixel_level"]:
                pixel_results = evaluator.evaluate_pixel_level(test_loader)
                logger.info("   Pixel-level: " + ", ".join([f"{k}:{v:.3f}" for k, v in pixel_results.items()]))


if __name__ == "__main__":

    if 1:
        args = parse_args()
        config = update_config(**args.__dict__)
    if 0:
        args = {
            "dataset": "mvtec",
            "category": ["tile", "grid"],
            # "category": ["bottle"],
            "model": "stfpm",
            "max_epochs": 10,
            "validate": True,
            "save_model": False,
            "pixel_level": False
        }
        config = update_config(**args)

    train(config)