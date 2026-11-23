"""Test rapide : un epoch du CNN sur un petit sous-ensemble (320 fichiers),
pour vérifier si le pipeline est juste lent à grande échelle, ou bloqué.

Lancer avec : python -m scripts.debug_quick_test
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data import load_splits, ASVspoofDataset
from src.models.cnn2d import CNN2DAntiSpoof
from src.train import run_epoch, DEVICE


def main():
    print("Device utilisé :", DEVICE)

    print("Chargement du dataset...")
    t0 = time.time()
    splits = load_splits()
    print(f"  -> fait en {time.time() - t0:.1f}s")

    train_dataset = ASVspoofDataset(splits["train"], duration_s=4.0)
    small_train = Subset(train_dataset, range(320))
    small_loader = DataLoader(small_train, batch_size=32, shuffle=True)

    cnn = CNN2DAntiSpoof()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    print("Lancement d'un epoch sur 320 fichiers...")
    t0 = time.time()
    loss = run_epoch(cnn, small_loader, criterion, optimizer)
    elapsed = time.time() - t0

    print(f"1 epoch sur 320 fichiers : {elapsed:.1f}s | loss = {loss:.4f}")


if __name__ == "__main__":
    main()