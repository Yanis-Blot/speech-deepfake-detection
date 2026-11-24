"""Évaluation finale sur le split TEST — à lancer une seule fois, modèle déjà entraîné.

Lancer avec : python -m scripts.evaluate_model --model cnn2d --run-name cnn2d
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import load_splits, ASVspoofDataset
from src.models import build_model
from src.train import collect_predictions, DEVICE
from src.metrics import compute_eer, per_attack_eer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mlp", "cnn2d"])
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    checkpoint_path = Path("checkpoints") / args.run_name / "model.pt"
    model = build_model(args.model)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    print(f"Modèle chargé depuis {checkpoint_path}")

    splits = load_splits()
    test_loader = DataLoader(
        ASVspoofDataset(splits["test"], duration_s=4.0),
        batch_size=32, shuffle=False, num_workers=4,
    )

    labels, scores, system_ids = collect_predictions(model, test_loader, desc="test")

    eer, threshold = compute_eer(labels.numpy(), scores.numpy())
    print(f"\nEER global sur TEST : {eer:.4f} (seuil : {threshold:.4f})")

    print("\nEER par type d'attaque :")
    per_attack = per_attack_eer(labels.numpy(), scores.numpy(), system_ids)
    for attack_id, attack_eer in per_attack.items():
        print(f"  {attack_id} : {attack_eer:.4f}")


if __name__ == "__main__":
    main()