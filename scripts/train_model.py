"""Entraînement générique, avec early stopping sur l'EER de validation.

Lancer avec : python -m scripts.train_model --model cnn2d
              python -m scripts.train_model --model wav2vec2
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import load_splits, ASVspoofDataset
from src.models import build_model, get_hyperparams
from src.train import run_epoch, predict_scores, DEVICE
from src.metrics import compute_eer

DURATION_S = 4.0
MAX_EPOCHS = 50
PATIENCE = 5
NUM_WORKERS = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mlp", "cnn2d", "wav2vec2"])
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_name = args.run_name or args.model
    hyperparams = get_hyperparams(args.model)
    print(f"Device : {DEVICE}")
    print(f"Hyperparamètres : {hyperparams}")

    splits = load_splits()
    train_loader = DataLoader(
        ASVspoofDataset(splits["train"], duration_s=DURATION_S),
        batch_size=hyperparams["batch_size"], shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        ASVspoofDataset(splits["validation"], duration_s=DURATION_S),
        batch_size=hyperparams["batch_size"], shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = build_model(args.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / f"history_{run_name}.csv"

    best_eer = float("inf")
    epochs_without_improvement = 0

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_eer", "best_so_far"])

        for epoch in range(MAX_EPOCHS):
            train_loss = run_epoch(model, train_loader, criterion, optimizer, desc=f"train {epoch}")
            labels, scores = predict_scores(model, val_loader, desc=f"val {epoch}")
            val_eer, _ = compute_eer(labels.numpy(), scores.numpy())

            print(f"epoch {epoch:2d} | loss train = {train_loss:.4f} | EER validation = {val_eer:.4f}")

            is_best = val_eer < best_eer
            if is_best:
                best_eer = val_eer
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_dir / "model.pt")
                print(f"           -> meilleur modèle sauvegardé (EER={best_eer:.4f})")
            else:
                epochs_without_improvement += 1

            writer.writerow([epoch, train_loss, val_eer, best_eer])
            f.flush()

            if epochs_without_improvement >= PATIENCE:
                print(f"Arrêt anticipé après {PATIENCE} epochs sans amélioration.")
                break

    print(f"Meilleur EER validation : {best_eer:.4f}")
    print(f"Historique sauvegardé dans : {history_path}")


if __name__ == "__main__":
    main()