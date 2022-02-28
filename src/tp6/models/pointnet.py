import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from tqdm import tqdm  # type: ignore


class PointMLP(nn.Module):
    def __init__(self, classes=40):
        super().__init__()
        self.classes = classes

        self.B1 = nn.Sequential(
            nn.Flatten(), nn.Linear(3072, 512), nn.BatchNorm1d(512), nn.ReLU()
        )

        self.B2 = nn.Sequential(
            nn.Linear(512, 256), nn.Dropout(p=0.3), nn.BatchNorm1d(256), nn.ReLU()
        )

        self.B3 = nn.Sequential(nn.Linear(256, classes), nn.LogSoftmax(dim=1))

    def forward(self, input):
        return self.B3(self.B2(self.B1(input)))


class PointNetBasic(nn.Module):
    def __init__(self, classes=40):
        super().__init__()

        self.B1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.B2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1025, kernel_size=1),
            nn.BatchNorm1d(1025),
            nn.ReLU(),
        )
        self.B3 = nn.Sequential(
            nn.MaxPool1d(1024),
            nn.Flatten(start_dim=1),
            nn.Linear(1025, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, classes),
            nn.BatchNorm1d(classes),
            nn.ReLU(),
        )

    def forward(self, input):
        x = self.B1(input)
        x = self.B2(x)
        x = self.B3(x)
        return F.log_softmax(x, dim=1)


class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.B1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1025, kernel_size=1),
            nn.BatchNorm1d(1025),
            nn.ReLU(),
        )
        self.B2 = nn.Sequential(
            nn.MaxPool1d(1024),
            nn.Flatten(start_dim=1),
            nn.Linear(1025, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, input):
        x = self.B1(input)
        x = self.B2(x)
        return x


class PointNetFull(nn.Module):
    def __init__(self, classes=40):
        super().__init__()
        self.basic = PointNetBasic(classes=classes)
        self.tnet = Tnet()

    def forward(self, input):
        m3x3 = self.tnet(input).view(-1, 3, 3)
        x = torch.bmm(m3x3, input)
        return self.basic(x), m3x3


def basic_loss(outputs, labels):
    criterion = torch.nn.NLLLoss()
    return criterion(outputs, labels)


def pointnet_full_loss(outputs, labels, m3x3, alpha=0.001):
    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)) / float(bs)


def train(model, device, train_loader, test_loader=None, epochs=250):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss = 0
    with tqdm(total=epochs * len(train_loader)) as pbar:
        for epoch in range(epochs):
            model.train()
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data["pointcloud"].to(device).float(), data[
                    "category"
                ].to(device)
                optimizer.zero_grad()
                # outputs = model(inputs.transpose(1, 2))
                outputs, m3x3 = model(inputs.transpose(1, 2))
                # loss = basic_loss(outputs, labels)
                loss = pointnet_full_loss(outputs, labels, m3x3)
                loss.backward()
                optimizer.step()
                pbar.update(1)

            model.eval()
            correct = total = 0
            if test_loader:
                with torch.no_grad():
                    for data in test_loader:
                        inputs, labels = data["pointcloud"].to(device).float(), data[
                            "category"
                        ].to(device)
                        # outputs = model(inputs.transpose(1, 2))
                        outputs, __ = model(inputs.transpose(1, 2))
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                val_acc = 100.0 * correct / total
                tqdm.write(
                    f"Epoch: {epoch + 1}, Loss: {loss:.3f}, Test accuracy: {val_acc:.1f} %"
                )

            scheduler.step()
        pbar.update(1)
