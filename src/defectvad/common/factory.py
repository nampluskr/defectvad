# common/factory.py

import importlib
from torch.utils.data import DataLoader
import torchvision.transforms as T


def create_dataset(split, config):
    module = importlib.import_module(config["module"])
    module_class = getattr(module, config["class"])

    image_transforms = [T.Resize((config["img_size"], config["img_size"]))]
    if config["crop_size"] is not None:
        image_transforms.append(T.CenterCrop((config["crop_size"], config["crop_size"])))
    image_transforms.append(T.ToTensor())
    if config["normalize"]:
        image_transforms.append(T.Normalize(mean=config["mean"], std=config["std"]))
    
    mask_transforms = [T.Resize((config["img_size"], config["img_size"]))]
    if config["crop_size"] is not None:
        mask_transforms.append(T.CenterCrop((config["crop_size"], config["crop_size"])))
    mask_transforms.append(T.ToTensor())

    return module_class(
        root_dir=config["path"],
        category=config["category"],
        split=split,    # "train" or "test"
        transform=T.Compose(image_transforms),
        mask_transform=T.Compose(mask_transforms),
    )


def create_dataloader(dataset, config):
    return DataLoader(dataset, **config)


def create_model(config):
    module = importlib.import_module(config["module"])
    module_class = getattr(module, config["class"])
    if config["params"] is not None:
        return module_class(**config["params"])
    else:
        return module_class()


def create_trainer(model, config):
    module = importlib.import_module(config["module"])
    module_class = getattr(module, config["class"])
    if config["params"] is not None:
        return module_class(model, **config["params"])
    else:
        return module_class(model)