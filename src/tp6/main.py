import time
from pathlib import Path

import click
import numpy as np
import torch  # type: ignore
import tp6.data.paths as paths
from rich import print as rprint
from torch.utils.data import DataLoader  # type: ignore
from tp6.data.dataset import PointCloudData
from tp6.models.pointnet import PointMLP, PointNetBasic, PointNetFull, train


@click.command()
@click.option(
    "-d",
    "--data",
    default=paths.small_data_dir,
    type=click.Path(exists=True),
    help="Data folder to use.",
)
@click.option(
    "-m",
    "--model-type",
    default="BASIC",
    type=click.Choice(["MLP", "BASIC", "FULL"], case_sensitive=False),
    help="Architecture to use.",
)
@click.option("-e", "--epochs", default=25, help="Number of epochs.")
def cli(data, model_type, epochs):

    data_dir = Path(data)
    rprint(f"Loading data from [blue]{data_dir.name}[/]")
    train_ds = PointCloudData(data_dir)
    test_ds = PointCloudData(data_dir, folder="test")

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    rprint(f"{len(train_ds.classes)} classes: [blue]{inv_classes}[/]")
    rprint(f"Train dataset size: [blue]{len(train_ds)}[/]")
    rprint(f"Test dataset size: [blue]{len(test_ds)}[/]")
    rprint(f"Sample pointcloud shape: [blue]{train_ds[0]['pointcloud'].size()}[/]")
    print()

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32)
    if model_type == "MLP":
        rprint("Using architecture [blue]PointMLP[/]")
        model = PointMLP()
    elif model_type == "BASIC":
        rprint("Using architecture [blue]PointNetBasic[/]")
        model = PointNetBasic()
    elif model_type == "FULL":
        rprint("Using architecture [blue]PointNetFulli[/]")
        model = PointNetFull()
    else:
        raise ValueError

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([np.prod(p.size()) for p in model_parameters])
    rprint(f"Number of parameters: [blue]{n_parameters}[/]")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    cuda = "[green]cuda[/]" if torch.cuda.is_available() else "[red strike]cuda[/]"
    cpu = "[red strike]cpu[/]" if torch.cuda.is_available() else "[green]cpu[/]"
    rprint(f"Device:\t{cuda}\t{cpu}")

    train(
        model,
        device,
        train_loader,
        test_loader,
        epochs=epochs,
        with_tnet=(model_type == "FULL"),
    )


if __name__ == "__main__":

    t0 = time.time()
    cli()
    rprint(f"Total time for training : [cyan]{time.time() - t0}[/]s")
