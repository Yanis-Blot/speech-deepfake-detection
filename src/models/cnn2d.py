"""CNN 2D anti-spoofing : convolutions masquées sur un spectrogramme Log-STFT."""

import torch.nn as nn

from src.features import LogSTFT, time_mask_from_sample_mask, N_FFT


class MaskedConvBlock(nn.Module):
    """Conv2D + ReLU + MaxPool2D, avec mise à jour du mask temporel après le pooling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x, time_mask):
        """x : [B, C, F, T] ; time_mask : [B, T] -> renvoie (x_pooled, time_mask_pooled)"""
        x = self.relu(self.conv(x))
        x = self.pool(x)

        # le pooling (kernel=2) divise la dimension temporelle par 2 (arrondi au-dessus)
        new_t = x.shape[-1]
        time_mask = time_mask[:, ::2][:, :new_t]
        return x, time_mask


class CNN2DAntiSpoof(nn.Module):
    """CNN 2D complet : extraction Log-STFT + plusieurs MaskedConvBlock + pooling temporel masqué + classification."""

    input_keys = ("waveform", "attention_mask")

    def __init__(self, channels=(16, 32, 64, 128)):
        super().__init__()
        self.extractor = LogSTFT()

        blocks = []
        in_channels = 1
        for out_channels in channels:
            blocks.append(MaskedConvBlock(in_channels, out_channels))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

        # calcule la taille finale de l'axe fréquence après tous les poolings,
        # pour dimensionner correctement la couche de classification
        n_freq_bins = N_FFT // 2 + 1
        for _ in self.blocks:
            n_freq_bins = -(-n_freq_bins // 2)   # division entière arrondie vers le haut (ceil)

        self.classifier = nn.Linear(channels[-1] * n_freq_bins, 1)

    def forward(self, waveform, attention_mask):
        spectro = self.extractor(waveform)
        n_frames = spectro.shape[-1]
        mask = time_mask_from_sample_mask(attention_mask, n_frames=n_frames)

        x = spectro.unsqueeze(1)
        for block in self.blocks:
            x, mask = block(x, mask)

        # pooling temporel masqué : moyenne sur le temps SEULEMENT, en ignorant le padding
        mask_expanded = mask.unsqueeze(1).unsqueeze(1).float()      # [B, 1, 1, T']
        x = x * mask_expanded
        valid_counts = mask.sum(dim=-1).clamp(min=1).view(-1, 1, 1)  # évite une division par 0
        pooled = x.sum(dim=-1) / valid_counts                         # [B, C, F'] -- fréquence conservée

        flat = pooled.flatten(start_dim=1)          # [B, C*F']
        logits = self.classifier(flat).squeeze(-1)  # [B]
        return logits