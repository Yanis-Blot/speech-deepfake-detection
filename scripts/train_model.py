"""Entraînement générique, avec early stopping sur l'EER de validation.

Lancer avec : python -m scripts.train_model --model cnn2d
              python -m scripts.train_model --model mlp
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import load_splits, ASVspoofDataset
from src.models import build_model
from src.train import run_epoch, predict_scores
from src.metrics import compute_eer

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DURATION_S = 4.0
MAX_EPOCHS = 50
PATIENCE = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mlp", "cnn2d"])
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_name = args.run_name or args.model

    splits = load_splits()
    train_loader = DataLoader(
        ASVspoofDataset(splits["train"], duration_s=DURATION_S),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        ASVspoofDataset(splits["validation"], duration_s=DURATION_S),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model = build_model(args.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_eer = float("inf")
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        labels, scores = predict_scores(model, val_loader)
        val_eer, _ = compute_eer(labels.numpy(), scores.numpy())

        print(f"epoch {epoch:2d} | loss train = {train_loss:.4f} | EER validation = {val_eer:.4f}")

        if val_eer < best_eer:
            best_eer = val_eer
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_dir / "model.pt")
            print(f"           -> meilleur modèle sauvegardé (EER={best_eer:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Arrêt anticipé après {PATIENCE} epochs sans amélioration.")
                break

    print(f"Meilleur EER validation : {best_eer:.4f}")


if __name__ == "__main__":
    main()