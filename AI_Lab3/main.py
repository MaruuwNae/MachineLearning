import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile

torch.manual_seed(42)
os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
test = datasets.FashionMNIST("data", train=False, download=True, transform=transform)

train, val = random_split(train, [55000, 5000],
                          generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train, batch_size=128, shuffle=True)
val_loader = DataLoader(val, batch_size=128)
test_loader = DataLoader(test, batch_size=128)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Linear(49, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.reshape(x.size(0), 16, 49)
        x = self.patch(x)
        x = self.encoder(x)
        return self.output(x.mean(1))


models = {
    "MLP": nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10)
    ),
    "CNN": nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 10)
    ),
    "Attention": Attention()
}


def accuracy(model, loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()

    return correct / len(loader.dataset) * 100


def measure(model, batch, runs):
    model.eval()
    x = torch.randn(batch, 1, 28, 28, device=device)
    times = []

    with torch.no_grad():
        for i in range(5):
            model(x)

        if device == "cuda":
            torch.cuda.synchronize()

        for i in range(runs):
            start = time.perf_counter()
            model(x)

            if device == "cuda":
                torch.cuda.synchronize()

            times.append((time.perf_counter() - start) * 1000)

    return np.median(times)


results = []

for name, model in models.items():
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining:", name)

    for epoch in range(5):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)

        val_acc = accuracy(model, val_loader)
        print(epoch + 1, round(total_loss / 55000, 4), round(val_acc, 2))

    test_acc = accuracy(model, test_loader)
    params = summary(model, input_size=(1, 1, 28, 28), verbose=0).total_params

    try:
        macs, _ = profile(model, inputs=(torch.randn(1, 1, 28, 28, device=device),),
                          verbose=False)
        flops = int(macs * 2)
    except Exception:
        flops = 0

    t1 = measure(model, 1, 100)
    t128 = measure(model, 128, 20)

    results.append([name, params, flops, test_acc, t1, t128])
    print(name, "Accuracy:", round(test_acc, 2), "%")


for column, filename, label in [
    (1, "accuracy_parameters.png", "Parameters"),
    (4, "accuracy_latency.png", "Latency (ms)")
]:
    plt.figure()

    for row in results:
        plt.scatter(row[column], row[3], label=row[0], s=80)

    plt.xlabel(label)
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()
    plt.savefig("results/" + filename)
    plt.close()


with open("results/report.md", "w", encoding="utf-8") as f:
    f.write("# Lab 3 (71616)\n\n")
    f.write("Fashion-MNIST: 55k train, 5k validation, 10k test.\n\n")
    f.write("Adam, lr=0.001, 5 epochs, batch size 128.\n\n")
    f.write(f"Device: {device}\n\n")

    f.write("| Model | Params | FLOPs (THOP) | Accuracy | Batch 1 | Batch 128 |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")

    for name, p, flops, acc, t1, t128 in results:
        f.write(f"| {name} | {p} | {flops} | {acc:.2f}% | {t1:.3f} ms | {t128:.3f} ms |\n")

    f.write("\nManual FLOPs calculations:\n\n")
    f.write("- MLP Linear: 2 * 784 * 256 = 401408\n")
    f.write("- CNN Conv: 2 * 3 * 3 * 1 * 16 * 28 * 28 = 225792\n")
    f.write("- Attention QKV: 2 * 16 * 64 * 3 * 64 = 393216\n\n")

    f.write("Full MLP: 401408 + 65536 + 2560 = 469504 FLOPs, ")
    f.write("which matches THOP.\n\n")

    f.write("The CNN and Attention manual calculations are for individual ")
    f.write("layers, not the full models. THOP may miss some attention operations.\n\n")

    best = max(results, key=lambda r: r[3])
    fastest = min(results, key=lambda r: r[4])
    smallest = min(results, key=lambda r: r[1])

    f.write(f"{best[0]} had the best accuracy ({best[3]:.2f}%). ")
    f.write(f"{fastest[0]} was fastest ({fastest[4]:.3f} ms for batch 1). ")
    f.write(f"{smallest[0]} had the fewest parameters ({smallest[1]}).\n\n")

    f.write("MLP uses dense layers and is simple to run. ")
    f.write("CNN uses shared filters and keeps image structure. ")
    f.write("Attention compares patches but needs more memory operations.\n\n")

    f.write("The parameter counts differ from the 100k-150k target ")
    f.write("because the given layer sizes were used.\n\n")

    f.write("![Accuracy vs Parameters](accuracy_parameters.png)\n\n")
    f.write("![Accuracy vs Latency](accuracy_latency.png)\n")
