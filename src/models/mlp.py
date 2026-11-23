"""Premier modèle, volontairement minimal : juste pour vérifier que le pipeline fonctionne de bout en bout."""

import torch.nn as nn


class TinyMLP(nn.Module):
    """MLP très simple sur le waveform brut. Pas un vrai modèle de référence — un test de branchement."""

    input_keys = ("waveform", "attention_mask")   # convention commune à tous les modèles

    def __init__(self, input_size=64000, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, waveform, attention_mask):
        # NOTE : le mask n'est pas encore utilisé ici -> limite connue, à corriger
        # quand on passera à un vrai modèle (features + pooling masqué).
        logits = self.net(waveform)       # [B, 64000] -> [B, 1]
        return logits.squeeze(-1)         # [B, 1] -> [B]