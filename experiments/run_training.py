# experiments/run_training.py

#####################################################################
# Experiment split lists
#####################################################################

DATASET_LIST = ["btad"]
CATEGORY_LIST = {
    # "mvtec": ["carpet", "grid", "leather", "tile", "wood"],  # texture
    # "mvtec": [["carpet", "grid", "leather", "tile", "wood"]],  # texture
    # "mvtec": [["bottle", "grid"]],  # test category
    # "mvtec": ["bottle"],  # test category
    # "visa": ["pipe_fryum"],  # test category
    "btad": ["01"],  # test category
}
# MODEL_LIST = ["stfpm"]
# MODEL_LIST = ["efficientad"]
# MODEL_LIST = ["reversedistill"]
MODEL_LIST = ["reversedistill", "efficientad", "stfpm"]
# MODEL_LIST = ["fastflow", "csflow", "uflow"]
# MODEL_LIST = ["dinomaly"]

MAX_EPOCHS = 10          # 1 (memory-based: dfkde, dfm, padim, patchcore)
VALIDATE = True        # False (memory-based: dfkde, dfm)
SAVE_MODEL = True
PIXEL_LEVEL = True

#####################################################################
# Script file path (absolute)
#####################################################################

import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_FILE = os.path.join(PROJECT_DIR, "experiments", "train.py")


#####################################################################
# Run function
#####################################################################

def run(script_file, dataset_list, category_list, model_list):
    import subprocess

    if not os.path.isfile(SCRIPT_FILE):
        raise FileNotFoundError(SCRIPT_FILE)

    total = 0
    for dataset in dataset_list:
        if dataset not in category_list:
            raise ValueError(f"category_list not defined for dataset: {dataset}")
        total += len(category_list[dataset]) * len(model_list)

    counter = 0
    for dataset in dataset_list:
        if dataset not in category_list:
            raise ValueError(f"category_list not defined for dataset: {dataset}")

        for model in model_list:
            for category in category_list[dataset]:
                counter += 1
                category = [category] if isinstance(category, str) else category

                print("\n" + "-" * 80)
                print(f" [Training {counter}/{total}] {dataset} | "
                      f"{', '.join(category)} | {model} ({MAX_EPOCHS} epochs)"
                )
                print("-" * 80)

                cmd = [sys.executable, script_file]
                cmd.extend(["--dataset", dataset])
                cmd.extend(["--category"] + category)
                cmd.extend(["--model", model])
                cmd.extend(["--max_epochs", str(MAX_EPOCHS)])

                if VALIDATE:    cmd.extend(["--validate"])
                if SAVE_MODEL:  cmd.extend(["--save_model"])
                if PIXEL_LEVEL: cmd.extend(["--pixel_level"])

                result = subprocess.run(cmd, cwd=PROJECT_DIR)

                if result.returncode != 0:
                    print("[ERROR] execution failed")
                    print(f"  dataset : {dataset}")
                    print(f"  category: {category}")
                    print(f"  model   : {model}")
                    return

    print(" [FINISHED] All training completed!")


if __name__ == "__main__":

    run(SCRIPT_FILE, DATASET_LIST, CATEGORY_LIST, MODEL_LIST)