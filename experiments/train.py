# experiments/train.py

import os
import sys
import argparse

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
source_dir = os.path.join(project_dir, "src")

if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

_CONFIG_DIR = os.path.join(project_dir, "configs")
_OUTPUT_DIR = os.path.join(project_dir, "outputs")
_MODEL_DIR = os.path.join(source_dir, "defectvad", "models")

from defectvad.common.config import load_config, merge_configs
from defectvad.common.utils import set_seed
from defectvad.common.factory import create_dataset, create_dataloader
from defectvad.common.factory import create_model, create_trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()


def get_config(dataset_name, category_name, model_name):
    config = merge_configs(
        load_config(_CONFIG_DIR, "defaults.yaml"),
        load_config(os.path.join(_MODEL_DIR, model_name), "config.yaml"),
    )
    config["dataset"]["path"] = config["path"][dataset_name]
    config["dataset"]["category"] = category_name

    seed = config["seed"]
    normalize = "ON" if config["dataset"]["normalize"] else "OFF"
    batch_size = config["train_loader"]["batch_size"]
    max_epochs = config["trainer"]["params"]["max_epochs"]

    config["output"]["path"] = os.path.join(_OUTPUT_DIR, dataset_name, category_name, model_name)
    config["output"]["weights"] = f"weights_{dataset_name}_{category_name}_{model_name}_epochs-{max_epochs}.pth"
    config["output"]["params"] = f"seed: {seed} / normalize: {normalize} / batch_size: {batch_size} / max_epochs: {max_epochs}"

    os.environ["BACKBONE_DIR"] = config["path"]["backbone"]    # IMPORTANT: used in backbone loading
    os.environ["DATASET_DIR"] = config["path"]["dataset"]      # IMPORTTAT: used in dataset loading
    return config


def run_training(config):
    print(f" > Experiment weights: {config['output']['weights']}")
    print(f" > Experiment params:  {config['output']['params']}")

    set_seed(config["seed"])
    train_dataset = create_dataset("train", config["dataset"])
    test_dataset = create_dataset("test", config["dataset"])
    train_loader = create_dataloader(train_dataset, config["train_loader"])
    test_loader = create_dataloader(test_dataset, config["test_loader"])

    set_seed(config["seed"])
    model = create_model(config["model"])

    trainer = create_trainer(model, config["trainer"])
    trainer.fit(train_loader, valid_loader=test_loader)

    weights_path = os.path.join(config["output"]["path"], config["output"]["weights"])
    model.save(weights_path)    # TODO: print("Model weights saved")


def run_evaluation(config):
    print(f" > Experiment weights: {config['output']['weights']}")
    print(f" > Experiment params:  {config['output']['params']}")

    set_seed(config["seed"])
    train_dataset = create_dataset("train", config["dataset"])
    test_dataset = create_dataset("test", config["dataset"])
    train_loader = create_dataloader(train_dataset, config["train_loader"])
    test_loader = create_dataloader(test_dataset, config["test_loader"])

    set_seed(config["seed"])
    model = create_model(config["model"])
    weights_path = os.path.join(config["output"]["path"], config["output"]["weights"])
    model.load(weights_path)    # TODO: print("Model weights loaded")


def run_visualization(config):
    pass


if __name__ == "__main__":

    if 0:
        args = parse_args()
        config = get_config(args.dataset, args.category, args.model)
        run_training(config)
    if 0:
        config = get_config("mvtec", "bottle", "stfpm")
        run_training(config)
    if 1:
        config = get_config("mvtec", "bottle", "stfpm")
        run_evaluation(config)

