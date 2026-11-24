"""Boucle d'entraînement — COMMUNE à tous les modèles."""

import torch
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer=None, desc="epoch"):
    """Une passe sur tout le loader. Si optimizer est fourni -> entraînement, sinon -> évaluation seule."""
    model.to(DEVICE)
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    # vérifications explicites : le modèle est-il vraiment sur le GPU ?
    model_device = next(model.parameters()).device
    print(f"[{desc}] modèle chargé sur : {model_device}")
    if DEVICE.type == "cuda":
        print(f"[{desc}] mémoire GPU allouée avant epoch : {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    total_loss = 0.0
    n_batches = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
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
        progress.set_postfix(loss=f"{loss.item():.4f}")

    if DEVICE.type == "cuda":
        print(f"[{desc}] mémoire GPU allouée après epoch  : {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    return total_loss / n_batches


def predict_scores(model, loader, desc="eval"):
    """Fait tourner le modèle sur tout le loader SANS entraîner, et retourne (labels, scores)."""
    model.to(DEVICE)
    model.eval()
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            inputs = {key: batch[key].to(DEVICE) for key in model.input_keys}
            logits = model(**inputs)
            scores = logits.float()

            all_labels.append(batch["label"])
            all_scores.append(scores.cpu())

    return torch.cat(all_labels), torch.cat(all_scores)

def collect_predictions(model, loader, desc="eval"):
    """Comme predict_scores, mais garde aussi system_id pour l'analyse par attaque."""
    model.to(DEVICE)
    model.eval()
    all_labels, all_scores, all_system_ids = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            inputs = {key: batch[key].to(DEVICE) for key in model.input_keys}
            logits = model(**inputs)
            scores = torch.sigmoid(logits)

            all_labels.append(batch["label"])
            all_scores.append(scores.cpu())
            all_system_ids.extend(batch["system_id"])

    return torch.cat(all_labels), torch.cat(all_scores), all_system_ids