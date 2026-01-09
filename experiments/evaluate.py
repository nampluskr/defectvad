# experiments/evaluate.py

import os
import sys
from argparse import ArgumentParser

from utils import set_environment, get_last_file


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_CIR = os.path.join(PROJECT_DIR, "src")
CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

if SOURCE_CIR not in sys.path:
    sys.path.insert(0, SOURCE_CIR)

from defectvad.common.config import load_config
from defectvad.common.utils import set_seed
from defectvad.common.factory import create_dataset, create_dataloader, create_model
from defectvad.common.evaluator import Evaluator


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--category", type=str, required=True, nargs="+")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--timestamp", type=str, default=None)

    parser.add_argument("--image_level", action="store_true")   # evaluator
    parser.add_argument("--pixel_level", action="store_true")   # evaluator
    return parser.parse_args()


def evaluate(dataset, category, model, image_level, pixel_level, timestamp=None):
    # ===============================================================
    # Load configs
    # ===============================================================

    category = "-".join(sorted(category))
    experiment_dir = os.path.join(OUTPUT_DIR, dataset, category, model)

    if timestamp is None:
        weights_name = get_last_file(experiment_dir, ext="*.pth")
        configs_name = get_last_file(experiment_dir, ext="*.yaml")
    else:
        weights_name = f"weights_{dataset}_{category}_{model}_{timestamp}.pth"
        configs_name = f"configs_{dataset}_{category}_{model}_{timestamp}.yaml"

    config = load_config(experiment_dir, configs_name)
    set_environment(config)
    set_seed(config["seed"])

    # ===============================================================
    # Create Datasets / Dataloaders / Model
    # ===============================================================

    train_dataset = create_dataset("train", config["dataset"])
    test_dataset = create_dataset("test", config["dataset"])

    train_loader = create_dataloader(train_dataset, config["train_loader"])
    test_loader = create_dataloader(test_dataset, config["test_loader"])

    vad = create_model(config["model"])
    vad.load(os.path.join(experiment_dir, weights_name))

    # ===============================================================
    # Evaluation: test_loader (batch_size=1)
    # ===============================================================

    test_dataset.info()
    evaluator = Evaluator(vad)

    print("\n*** Evaluation start...")
    if image_level:
        print(f" > {category}:")
        image_results = evaluator.evaluate_image_level(test_loader)
        print("   Image-level: " + ", ".join([f"{k}:{v:.3f}" for k, v in image_results.items()]))

    if pixel_level:
        pixel_results = evaluator.evaluate_pixel_level(test_loader)
        print("   Pixel-level: " + ", ".join([f"{k}:{v:.3f}" for k, v in pixel_results.items()]))

    categories = category.split("-")
    if len(categories) > 1:
        for category in categories:
            test_dataset = test_dataset.subset(category)
            test_loader = create_dataloader(test_dataset, config["test_loader"])

            if image_level:
                print(f" > {category}:")
                image_results = evaluator.evaluate_image_level(test_loader)
                print("   Image-level: " + ", ".join([f"{k}:{v:.3f}" for k, v in image_results.items()]))

            if pixel_level:
                pixel_results = evaluator.evaluate_pixel_level(test_loader)
                print("   Pixel-level: " + ", ".join([f"{k}:{v:.3f}" for k, v in pixel_results.items()]))

    print("*** Evaluation completed!")

if __name__ == "__main__":

    if 1:
        args = parse_args()
        evaluate(**args.__dict__)
    if 0:
        args = {
            "dataset": "mvtec",
            "category": ["carpet", "grid", "leather", "tile", "wood"],
            "model": "stfpm",
            "image_level": True,
            "pixel_level": True,
        }
        evaluate(**args)