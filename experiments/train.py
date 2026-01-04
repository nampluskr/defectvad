# experiments/train.py

import os
import sys
import argparse
import torch

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--max_epochs", type=int, default=10)
    return parser.parse_args()


def get_config(dataset, category, model, seed, normalize, batch_size, max_epochs):
    config = merge_configs(
        load_config(CONFIG_DIR, "defaults.yaml"),
        load_config(os.path.join(CONFIG_DIR, "datasets"), f"{dataset}.yaml"),
        load_config(os.path.join(CONFIG_DIR, "models"), f"{model}.yaml"),
    )

    # Update conifig values
    config["seed"] = seed
    config["dataset"]["path"] = config["path"][dataset]
    config["dataset"]["category"] = category
    config["dataset"]["normalize"] = normalize
    config["train_loader"]["batch_size"] = batch_size
    config["trainer"]["max_epochs"] = max_epochs
    return config


def set_environment(config):
    os.environ["BACKBONE_DIR"] = config["path"]["backbone"] # IMPORTANT: used in backbone loading
    os.environ["DATASET_DIR"] = config["path"]["dataset"]   # IMPORTTAT: used in dataset loading


def run_training(config):
    set_environment(config)
    set_seed(config["seed"])

    train_dataset = create_dataset("train", config["dataset"])
    test_dataset = create_dataset("test", config["dataset"])

    train_loader = create_dataloader(train_dataset, config["train_loader"])
    test_loader = create_dataloader(test_dataset, config["test_loader"])

    model = create_model(config["model"])
    trainer = create_trainer(model, config["trainer"])

    # Save trained model weights and configs
    dataset_name = config["dataset"]["name"]
    category_name = config["dataset"]["category"]
    model_name = config["model"]["name"]
    max_epochs = config["trainer"]["max_epochs"]
    experiment_name = f"{dataset_name}_{category_name}_{model_name}_{max_epochs}epoch"

    experiment_dir = os.path.join(OUTPUT_DIR, dataset_name, category_name, model_name)
    weights_name = f"weights_{experiment_name}.pth"
    configs_name = f"configs_{experiment_name}.yaml"

    if config["trainer"]["validate"]:
        trainer.fit(train_loader, max_epochs=max_epochs, valid_loader=test_loader)
    else:
        trainer.fit(train_loader, max_epochs=max_epochs, valid_loader=None)

    model.save(os.path.join(experiment_dir, weights_name))
    save_config(config, os.path.join(experiment_dir, configs_name))


if __name__ == "__main__":

    if 1:
        args = parse_args()
    if 0:
        args = {
            "dataset": "mvtec",
            "category": "bottle",
            "model": "stfpm",
            "seed": 42,
            "normalize": True,
            "batch_size": 16,
            "max_epochs": 10,
        }
    config = get_config(**args.__dict__)
    run_training(config)
