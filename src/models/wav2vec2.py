"""Modèle anti-spoofing : front-end wav2vec2 pré-entraîné, fine-tuné."""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

PRETRAINED_NAME = "facebook/wav2vec2-base"


class Wav2Vec2AntiSpoof(nn.Module):
    input_keys = ("waveform", "attention_mask")

    def __init__(self, pretrained_name=PRETRAINED_NAME, freeze_feature_extractor=True):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_name)

        if freeze_feature_extractor:
            # Gèle les couches convolutives de bas niveau (extraction de features brutes),
            # ne fine-tune que les blocs Transformer au-dessus. Pratique standard en
            # fine-tuning wav2vec2 : ça réduit la mémoire/temps de calcul et évite
            # d'abîmer des représentations acoustiques de bas niveau déjà bien apprises.
            self.wav2vec2.feature_extractor._freeze_parameters()

        hidden_size = self.wav2vec2.config.hidden_size  # 768 pour wav2vec2-base
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, waveform, attention_mask):
        outputs = self.wav2vec2(waveform, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [batch, n_frames, hidden_size]

        # Le masque qu'on a (sur les samples bruts) ne correspond plus à la longueur
        # après les convolutions internes de wav2vec2 (64000 samples -> 199 frames).
        # wav2vec2 fournit une méthode interne pour recalculer le masque à la bonne
        # résolution, en tenant compte de son propre sous-échantillonnage.
        frame_mask = self.wav2vec2._get_feature_vector_attention_mask(
            hidden.shape[1], attention_mask
        )

        mask_expanded = frame_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        logits = self.classifier(pooled).squeeze(-1)
        return logits