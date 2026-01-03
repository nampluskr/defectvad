# experiments/train.py

import os
import sys
import argparse

CONFIG_DIR = ""
OUTPUT_DIR = ""
MODEL_DIR = ""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()


def setup():
    global CONFIG_DIR, OUTPUT_DIR, MODEL_DIR

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    source_dir = os.path.join(project_dir, "src")

    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    CONFIG_DIR = os.path.join(project_dir, "configs")
    OUTPUT_DIR = os.path.join(project_dir, "outputs")
    MODEL_DIR = os.path.join(source_dir, "defectvad", "models")


def main(dataset_name, category_name, model_name):
    from defectvad.common.utils import set_seed
    from defectvad.common.config import load_config, merge_configs
    from defectvad.common.factory import create_dataset, create_dataloader
    from defectvad.common.factory import create_model, create_trainer

    # ===============================================================
    # Configurations
    # ===============================================================

    config = merge_configs(
        load_config(CONFIG_DIR, "defaults.yaml"),
        load_config(os.path.join(MODEL_DIR, model_name), "config.yaml"),
    )

    os.environ["BACKBONE_DIR"] = config["path"]["backbone"]    # IMPORTANT: used in backbone loading
    os.environ["DATASET_DIR"] = config["path"]["dataset"]      # IMPORTTAT: used in dataset loading

    config["dataset"]["path"] = config["path"][dataset_name]
    config["dataset"]["category"] = category_name

    set_seed(config["seed"])

    # ===============================================================
    # Training: Datasets / Dataloaders / Model / Trainer
    # ===============================================================

    train_dataset = create_dataset("train", config["dataset"])
    test_dataset = create_dataset("test", config["dataset"])

    train_loader = create_dataloader(train_dataset, config["train_loader"])
    test_loader = create_dataloader(test_dataset, config["test_loader"])

    model = create_model(config["model"])
    trainer = create_trainer(model, config["trainer"])

    trainer.fit(train_loader, valid_loader=test_loader)

    # ===============================================================
    # Training: Datasets / Dataloaders / Model / Trainer
    # ===============================================================


if __name__ == "__main__":

    # args = parse_args()
    # setup()
    # main(args.dataset, args.catetory, args.model)

    setup()
    main("mvtec", "bottle", "stfpm")
