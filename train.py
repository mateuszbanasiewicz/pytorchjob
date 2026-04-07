"""
Prosty przykład treningowy: klasyfikacja cyfr MNIST - 
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ── Model ─────────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Trening ────────────────────────────────────────────────────────────────────
def train(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct = 0.0, 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += output.argmax(dim=1).eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}]"
                f"  Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / len(loader.dataset)
    print(f">> Train  Epoch {epoch}: loss={avg_loss:.4f}, acc={accuracy:.2f}%")
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            correct += output.argmax(dim=1).eq(target).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / len(loader.dataset)
    print(f">> Val    Epoch {epoch}: loss={avg_loss:.4f}, acc={accuracy:.2f}%")
    return avg_loss, accuracy


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MNIST CNN Training")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--data-dir",   type=str,   default="/data")
    parser.add_argument("--output-dir", type=str,   default="/output")
    args = parser.parse_args()

    # Katalogi
    os.makedirs(args.data_dir,   exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")
    print(f"Parametry: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")

    # Dane
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(args.data_dir, train=True,  download=True, transform=transform)
    val_ds   = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model, optimizer, loss
    model     = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, device, epoch)
        _, val_acc = evaluate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        # Zapisz najlepszy model
        if val_acc > best_acc:
            best_acc = val_acc
            path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), path)
            print(f"   ✓ Zapisano model → {path}  (acc={best_acc:.2f}%)")

    # Zapisz ostatni checkpoint
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        "epoch":       args.epochs,
        "model_state": model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "best_acc":    best_acc,
    }, final_path)
    print(f"\nTrening zakończony. Best val acc: {best_acc:.2f}%")
    print(f"Checkpoint zapisany → {final_path}")


if __name__ == "__main__":
    main()
