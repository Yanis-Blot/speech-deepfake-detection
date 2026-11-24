"""Test isolé : vérifier que wav2vec2 charge et tourne sur un batch factice avant intégration."""

import torch
from transformers import Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-base"  # pré-entraîné self-supervised, PAS fine-tuné ASR

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    print(f"Chargement de {MODEL_NAME} ...")
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    print(f"Nombre de paramètres : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Batch factice : 4 échantillons de 4s à 16kHz = 64000 samples, comme votre pipeline existant
    batch_size = 4
    waveform = torch.randn(batch_size, 64_000, device=device)
    attention_mask = torch.ones(batch_size, 64_000, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(waveform, attention_mask=attention_mask)

    print(f"Forme de last_hidden_state : {outputs.last_hidden_state.shape}")
    print(f"Mémoire GPU allouée : {torch.cuda.memory_allocated() / 1e6:.1f} MB")


if __name__ == "__main__":
    main()