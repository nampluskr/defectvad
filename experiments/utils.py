# experiments/utils.py

import os
from glob import glob


def set_environment(config):
    os.environ["BACKBONE_DIR"] = config["path"]["backbone"] # IMPORTANT: used in backbone loading
    os.environ["DATASET_DIR"] = config["path"]["dataset"]   # IMPORTTAT: used in dataset loading


def get_last_file(experiment_dir, ext="*.*"):
    path_list = glob(os.path.join(experiment_dir, ext))
    file_list = [os.path.basename(path) for path in path_list]
    return sorted(file_list)[-1]