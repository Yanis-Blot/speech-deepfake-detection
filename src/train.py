"""Boucle d'entraînement — COMMUNE à tous les modèles."""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer=None):
    """Une passe sur tout le loader. Si optimizer est fourni -> entraînement, sinon -> évaluation seule."""
    model.to(DEVICE)
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        inputs = {key: batch[key].to(DEVICE) for key in model.input_keys}
        labels = batch["label"].to(DEVICE)

        with torch.set_grad_enabled(is_training):
            logits = model(**inputs)
            loss = criterion(logits, labels.float())

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def predict_scores(model, loader):
    """Fait tourner le modèle sur tout le loader SANS entraîner, et retourne (labels, scores)."""
    model.to(DEVICE)
    model.eval()
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in loader:
            inputs = {key: batch[key].to(DEVICE) for key in model.input_keys}
            logits = model(**inputs)
            scores = torch.sigmoid(logits)

            all_labels.append(batch["label"])
            all_scores.append(scores.cpu())

    return torch.cat(all_labels), torch.cat(all_scores)