"""Extraction de features audio — COMMUN à tous les modèles basés sur spectrogramme.

Log-STFT : transforme le waveform brut [N] en un spectrogramme log-amplitude [F, T]
(F = nombre de fréquences, T = nombre de pas de temps), l'entrée attendue par le CNN.
"""

import torch
import torch.nn as nn

N_FFT = 512          # taille de la fenêtre de calcul -> donne 257 bandes de fréquence (N_FFT//2 + 1)
WIN_LENGTH = 400     # 25 ms à 16kHz : durée d'analyse standard en traitement de la parole
HOP_LENGTH = 160     # 10 ms à 16kHz : décalage entre deux fenêtres (donc 60% de recouvrement)


class LogSTFT(nn.Module):
    """Transforme un batch de waveforms en spectrogrammes log-amplitude."""

    def __init__(self, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, waveform):
        """waveform : [B, N] -> spectrogram : [B, F, T]"""
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        magnitude = stft.abs()                     # on garde l'énergie, pas la phase (peu informative ici)
        log_magnitude = torch.log1p(magnitude)      # log(1+x) : évite log(0), compresse les grandes valeurs
        return log_magnitude




def time_mask_from_sample_mask(sample_mask, hop_length=HOP_LENGTH, n_frames=None):
    """Convertit un mask au niveau échantillon [B, N] en mask au niveau trame [B, T].

    Approximation : chaque trame t du spectrogramme correspond environ à l'échantillon
    t * hop_length. On compte donc combien de trames tombent avant la fin du contenu réel.
    """
    valid_samples = sample_mask.sum(dim=-1)                      # [B] : nb d'échantillons réels
    valid_frames = (valid_samples // hop_length + 1)             # conversion approximative

    if n_frames is not None:
        valid_frames = valid_frames.clamp(max=n_frames)

    arange = torch.arange(n_frames, device=sample_mask.device)
    time_mask = arange.unsqueeze(0) < valid_frames.unsqueeze(1)   # [B, T] : True si trame réelle
    return time_mask