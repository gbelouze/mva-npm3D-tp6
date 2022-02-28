import math
import random
from pathlib import Path

import numpy as np
import torch  # type: ignore
from torch.utils.data import Dataset  # type: ignore
from torchvision import transforms  # type: ignore
from tp6.data.ply import read_ply


class RandomRotation(object):
    def __init__(self, ax="z"):
        if ax == "x" or ax == [1, 0, 0]:
            self.permute = (2, 0, 1)
        elif ax == "y" or ax == [0, 1, 0]:
            self.permute = (0, 2, 1)
        elif ax == "z" or ax == [0, 0, 1]:
            self.permute = (0, 1, 2)
        else:
            raise ValueError

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        """
        pointcloud: (N, 3) array of 3D points
        """
        theta = random.random() * 2.0 * math.pi
        rot = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ]
        )[self.permute, :][:, self.permute]
        return pointcloud @ rot


class RandomFlip(object):
    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        if np.random.random() > 0.5:
            return pointcloud * np.array([[-1, 1, 1]])
        return pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, pointcloud.shape)
        return pointcloud + noise


class ShufflePoints(object):
    def __call__(self, pointcloud):
        np.random.shuffle(pointcloud)
        return pointcloud


class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def default_transforms(augmentation=True):
    if augmentation:
        return transforms.Compose(
            [
                RandomRotation(ax="z"),
                RandomFlip(),
                RandomNoise(),
                ShufflePoints(),
                ToTensor(),
            ]
        )
    return ToTensor()


class PointCloudData(Dataset):
    def __init__(self, root_dir: Path, folder="train", transform=default_transforms):
        self.root_dir = root_dir
        folders = [dir for dir in root_dir.iterdir() if dir.is_dir()]
        self.classes = {folder.name: i for i, folder in enumerate(folders)}
        self.transforms = transform(augmentation=(folder == "train"))
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir / category / folder
            for file in new_dir.iterdir():
                if file.suffix == ".ply":
                    sample = {
                        "ply_path": (new_dir / file).resolve(),
                        "category": category,
                    }
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ply_path = self.files[idx]["ply_path"]
        category = self.files[idx]["category"]
        data = read_ply(ply_path)
        pointcloud = self.transforms(np.vstack((data["x"], data["y"], data["z"])).T)
        return {"pointcloud": pointcloud, "category": self.classes[category]}
