"""Lecture des données — COMMUN à tous les modèles."""

import io
import os
from pathlib import Path

import soundfile as sf
import torch

DATASET_NAME = "Bisher/ASVspoof_2019_LA"
SAMPLE_RATE = 16_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "huggingface_cache"


def load_splits(cache_dir=CACHE_DIR):
    """Ouvre train / validation / test depuis le cache local, sans rien télécharger."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Cache introuvable : {cache_dir}. Lancer d'abord download_dataset.py.")

    os.environ["HF_HUB_OFFLINE"] = "1"
    from datasets import Audio, load_dataset

    ds = load_dataset(DATASET_NAME, cache_dir=str(cache_dir))
    for split in ds:
        ds[split] = ds[split].cast_column("audio", Audio(decode=False))
    return ds


def read_waveform(audio):
    """Audio non décodé {'bytes': ..., 'path': ...} -> waveform float32 de forme [N]."""
    source = io.BytesIO(audio["bytes"]) if audio["bytes"] is not None else audio["path"]
    samples, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Sampling rate inattendu : {sample_rate} Hz (attendu {SAMPLE_RATE})")
    mono = samples.mean(axis=1)
    return torch.from_numpy(mono)


def leading_silence_duration(waveform, threshold=0.01):
    """Durée du silence au DÉBUT du signal, en secondes."""
    is_sound = waveform.abs() > threshold
    if not is_sound.any():
        return waveform.numel() / SAMPLE_RATE
    first_sound_idx = is_sound.nonzero()[0].item()
    return first_sound_idx / SAMPLE_RATE


def trailing_silence_duration(waveform, threshold=0.01):
    """Durée du silence à la FIN du signal, en secondes."""
    is_sound = waveform.abs() > threshold
    if not is_sound.any():
        return waveform.numel() / SAMPLE_RATE
    last_sound_idx = is_sound.nonzero()[-1].item()
    return (waveform.numel() - 1 - last_sound_idx) / SAMPLE_RATE


def fixed_length(waveform, duration_s, sample_rate=SAMPLE_RATE):
    """Force le waveform à une durée fixe : coupe s'il est trop long, complète par des zéros sinon.

    Retourne (waveform_fixe, mask) :
    - waveform_fixe : toujours exactement duration_s * sample_rate échantillons
    - mask : True pour les échantillons RÉELS, False pour le remplissage (padding)
    """
    target_len = int(duration_s * sample_rate)
    current_len = waveform.numel()

    if current_len >= target_len:
        cropped = waveform[:target_len]
        mask = torch.ones(target_len, dtype=torch.bool)
        return cropped, mask

    pad_len = target_len - current_len
    padded = torch.nn.functional.pad(waveform, (0, pad_len))
    mask = torch.zeros(target_len, dtype=torch.bool)
    mask[:current_len] = True
    return padded, mask


class ASVspoofDataset(torch.utils.data.Dataset):
    """Combine load_splits + read_waveform + fixed_length en un seul objet utilisable par DataLoader."""

    def __init__(self, hf_split, duration_s=4.0):
        self.hf_split = hf_split
        self.duration_s = duration_s

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        example = self.hf_split[idx]
        waveform = read_waveform(example["audio"])
        waveform, mask = fixed_length(waveform, self.duration_s)

        return {
            "waveform": waveform,
            "attention_mask": mask,
            "label": torch.tensor(example["key"], dtype=torch.long),
            "audio_file_name": example["audio_file_name"],
            "system_id": example["system_id"],
        }