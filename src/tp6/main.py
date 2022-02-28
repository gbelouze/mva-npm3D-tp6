import time
from pathlib import Path

import click
import numpy as np
import torch  # type: ignore
import tp6.data.paths as paths
from rich import print as rprint
from torch.utils.data import DataLoader  # type: ignore
from tp6.data.dataset import PointCloudData
from tp6.models.pointnet import PointMLP, PointNetBasic, PointNetFull
from tp6.models.pointnet import test as test_model
from tp6.models.pointnet import train as train_model


def test(load_from, data_dir):
    model = torch.load(load_from)
    test_ds = PointCloudData(data_dir, folder="test")
    test_loader = DataLoader(dataset=test_ds, batch_size=32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with_tnet = isinstance(model, PointNetFull)

    val_acc = test_model(model, device, test_loader, with_tnet=with_tnet)
    rprint(f"[blue]Test accuracy[/] : {val_acc:.2f}%")


def train(data_dir, model_type="BASIC", epochs=25, save_to=None):
    """Train and save a 3D neural network."""

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

    train_model(
        model,
        device,
        train_loader,
        test_loader,
        epochs=epochs,
        with_tnet=(model_type == "FULL"),
    )
    if save_to is not None:
        if str(save_to) == save_to.name:
            out = paths.trained_dir / save_to.name
        else:
            save_to.parent.mkdir(parents=True)
            out = save_to
        torch.save(model.state_dict(), out.resolve())


@click.group()
def cli():
    pass


@cli.command(name="test")
@click.argument(
    "load_from", type=click.Path(exists=True, path_type=Path, dir_okay=False)
)
@click.option(
    "-d",
    "--data-dir",
    "--data",
    default=paths.small_data_dir,
    type=click.Path(exists=True, path_type=Path),
    help="Data folder to use.",
)
def test_command(load_from, data_dir):
    test(load_from, data_dir)


@cli.command(name="train")
@click.option(
    "-d",
    "--data-dir",
    "--data",
    default=paths.small_data_dir,
    type=click.Path(exists=True, path_type=Path),
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
@click.option(
    "-o",
    "--save-to",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Path to save the trained model to.",
)
def train_command(data_dir, model_type, epochs, save_to):
    train(data_dir, model_type, epochs, save_to)


if __name__ == "__main__":
    cli()
