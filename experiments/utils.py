# experiments/utils.py

import logging
import os
from glob import glob


def set_environment(config):
    os.environ["BACKBONE_DIR"] = config["path"]["backbone"] # IMPORTANT: used in backbone loading
    os.environ["DATASET_DIR"] = config["path"]["dataset"]   # IMPORTTAT: used in dataset loading


def get_last_file(experiment_dir, ext="*.*"):
    path_list = glob(os.path.join(experiment_dir, ext))
    file_list = [os.path.basename(path) for path in path_list]
    return sorted(file_list)[-1]


def set_logging(log_dir, log_file="train.log", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_file)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.handlers:
        root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Silence noisy third-party libs
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    return log_path

