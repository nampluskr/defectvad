# experiments/train.py

import os
import sys
import argparse
import torch

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
source_dir = os.path.join(project_dir, "src")

if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

_CONFIG_DIR = os.path.join(project_dir, "configs")
_OUTPUT_DIR = os.path.join(project_dir, "outputs")

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
    parser.add_argument("--max_epochs", type=int, default=10)
    return parser.parse_args()


def get_config(dataset, category, model, max_epochs):
    config = merge_configs(
        load_config(_CONFIG_DIR, "defaults.yaml"),
        load_config(os.path.join(_CONFIG_DIR, "datasets"), f"{dataset}.yaml"),
        load_config(os.path.join(_CONFIG_DIR, "models"), f"{model}.yaml"),
    )
    config["dataset"]["path"] = config["path"][dataset]
    config["dataset"]["category"] = category

    experiment_name = f"{dataset}_{category}_{model}_{max_epochs}epoch"
    seed = config["seed"]
    normalize = "ON" if config["dataset"]["normalize"] else "OFF"
    batch_size = config["train_loader"]["batch_size"]

    config["output"]["name"] = experiment_name
    config["output"]["dataset"] = dataset
    config["output"]["category"] = category
    config["output"]["model"] = model
    config["output"]["max_epochs"] = max_epochs

    config["trainer"]["params"]["max_epochs"] = max_epochs

    config["output"]["path"] = os.path.join(_OUTPUT_DIR, dataset, category, model)
    config["output"]["params"] = f"seed: {seed}, normalize: {normalize}, batch_size: {batch_size}, max_epochs: {max_epochs}"
    config["output"]["weights"] = f"weights_{experiment_name}.pth"
    config["output"]["config"] = f"configs_{experiment_name}.yaml"

    os.environ["BACKBONE_DIR"] = config["path"]["backbone"]    # IMPORTANT: used in backbone loading
    os.environ["DATASET_DIR"] = config["path"]["dataset"]      # IMPORTTAT: used in dataset loading
    return config


def run_training(config):
    print(f" > Experiment: {config['output']['name']}")
    print(f" > Parameters: {config['output']['params']}")

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
    config_path = os.path.join(config["output"]["path"], config["output"]["config"])
    model.save(weights_path)
    save_config(config, config_path)


def run_evaluation(config):
    print(f" > Experiment: {config['output']['name']}")
    print(f" > Parameters: {config['output']['params']}")

    set_seed(config["seed"])
    train_dataset = create_dataset("train", config["dataset"])
    test_dataset = create_dataset("test", config["dataset"])
    train_loader = create_dataloader(train_dataset, config["train_loader"])
    test_loader = create_dataloader(test_dataset, config["test_loader"])

    set_seed(config["seed"])
    model = create_model(config["model"])
    weights_path = os.path.join(config["output"]["path"], config["output"]["weights"])
    model.load(weights_path)

    evaluator = Evaluator(model)

    print(f"\n*** Evaluation: Test Dataset (normal + anomaly)")

    image_results = evaluator.evaluate_image_level(test_loader)
    print(" > Image: " + ", ".join([f"{k}:{v:.3f}" for k, v in image_results.items()]))
    pixel_results = evaluator.evaluate_pixel_level(test_loader)
    print(" > Pixel: " + ", ".join([f"{k}:{v:.3f}" for k, v in pixel_results.items()]))

    print(f"\n*** Threshold calibration: Train Dataset (normal-only)")

    image_thresholds = evaluator.calibrate_image_thresholds(train_loader)
    image_results = evaluator.evaluate_image_level(test_loader, threshold=image_thresholds["95%"])
    print(" > Image: " + ", ".join([f"{k}:{v:.3f}" for k, v in image_results.items()]))

    pixel_thresholds = evaluator.calibrate_pixel_thresholds(train_loader)
    pixel_results = evaluator.evaluate_pixel_level(test_loader, threshold=pixel_thresholds["99%"])
    print(" > Pixel: " + ", ".join([f"{k}:{v:.3f}" for k, v in pixel_results.items()]))

def run_prediction(config):
    print(f" > Experiment: {config['output']['name']}")
    print(f" > Parameters: {config['output']['params']}")

    set_seed(config["seed"])
    train_dataset = create_dataset("train", config["dataset"])
    test_dataset = create_dataset("test", config["dataset"])
    train_loader = create_dataloader(train_dataset, config["train_loader"])
    test_loader = create_dataloader(test_dataset, config["test_loader"])

    set_seed(config["seed"])
    model = create_model(config["model"])
    weights_path = os.path.join(config["output"]["path"], config["output"]["weights"])
    model.load(weights_path)

    print("\n*** Prediction (dataloader):")
    preds = model.predict(test_loader)
    for k, v in preds.items():
        if torch.is_tensor(v):
            print(f" > {k}: {v.shape}")

    # print("\n*** Prediction (batch):")
    # batch = next(iter(test_loader))
    # preds = model.predict(batch)
    # for k, v in preds.items():
    #     if torch.is_tensor(v):
    #         print(f" > {k}: {v.shape}")

    # print("\n*** Prediction (images):")
    # batch = next(iter(test_loader))
    # batch = {"image": batch["image"], "label": batch["label"]}
    # preds = model.predict(batch)
    # for k, v in preds.items():
    #     if torch.is_tensor(v):
    #         print(f" > {k}: {v.shape}")

    evaluator = Evaluator(model)
    visualizer = Visualizer(preds)

    visualizer.show_anomaly(max_samples=3)
    visualizer.show_normal(max_samples=3)

    image_results = evaluator.evaluate_image_level(test_loader)
    pixel_results = evaluator.evaluate_pixel_level(test_loader)

    visualizer.set_image_threshold(image_results["th"])
    visualizer.set_pixel_threshold(pixel_results["th"])

    visualizer.show_anomaly(max_samples=3)
    visualizer.show_normal(max_samples=3)

    # Threshold calibration
    # image_thresholds = evaluator.calibrate_image_thresholds(train_loader)
    # visualizer.set_image_threshold(image_thresholds["95%"])

    # pixel_thresholds = evaluator.calibrate_pixel_thresholds(train_loader)
    # visualizer.set_pixel_threshold(pixel_thresholds["99%"])


if __name__ == "__main__":

    if 0:
        args = parse_args()
        config = get_config(args.dataset, args.category, args.model, args.max_epochs)
        run_training(config)
    if 0:
        config = get_config("mvtec", "bottle", "stfpm", 20)
        run_training(config)

    if 0:
        dataset, category, model, max_epochs = "mvtec", "bottle", "stfpm", 20
        experiment_name = f"{dataset}_{category}_{model}_{max_epochs}epoch"
        output_dir = os.path.join(_OUTPUT_DIR, dataset, category, model)
        config = load_config(output_dir, f"configs_{experiment_name}.yaml")

        os.environ["BACKBONE_DIR"] = config["path"]["backbone"]    # IMPORTANT: used in backbone loading
        os.environ["DATASET_DIR"] = config["path"]["dataset"]      # IMPORTTAT: used in dataset loading

        run_evaluation(config)

    if 1:
        dataset, category, model, max_epochs = "mvtec", "bottle", "stfpm", 20
        experiment_name = f"{dataset}_{category}_{model}_{max_epochs}epoch"
        output_dir = os.path.join(_OUTPUT_DIR, dataset, category, model)
        config = load_config(output_dir, f"configs_{experiment_name}.yaml")

        os.environ["BACKBONE_DIR"] = config["path"]["backbone"]    # IMPORTANT: used in backbone loading
        os.environ["DATASET_DIR"] = config["path"]["dataset"]      # IMPORTTAT: used in dataset loading

        run_prediction(config)

