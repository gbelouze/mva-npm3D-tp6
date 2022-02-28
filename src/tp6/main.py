import time
from pathlib import Path

import numpy as np
import torch  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from tp6.data.paths import data_dir
from tp6.models.pointnet import (
    PointCloudData,
    PointMLP,
    PointNetBasic,
    PointNetFull,
    train,
)

if __name__ == "__main__":

    t0 = time.time()

    train_ds = PointCloudData(Path(data_dir / "ModelNet10_PLY"))
    test_ds = PointCloudData(Path(data_dir / "ModelNet10_PLY"), folder="test")

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print("Train dataset size: ", len(train_ds))
    print("Test dataset size: ", len(test_ds))
    print("Number of classes: ", len(train_ds.classes))
    print("Sample pointcloud shape: ", train_ds[0]["pointcloud"].size())
    print()

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32)

    # model = PointMLP()
    # model = PointNetBasic()
    model = PointNetFull()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(
        "Number of parameters in the Neural Networks: ",
        sum([np.prod(p.size()) for p in model_parameters]),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    model.to(device)

    train(model, device, train_loader, test_loader, epochs=25)

    print("Total time for training : ", time.time() - t0)
