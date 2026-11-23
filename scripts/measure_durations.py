"""Mesure la distribution réelle des durées audio sur train, pour choisir
une durée fixe (fixed_length) justifiée plutôt qu'arbitraire.
"""

import numpy as np

from src.data import load_splits, read_waveform, SAMPLE_RATE


def main():
    ds = load_splits()
    train = ds["train"]

    durations_s = []
    for example in train:
        w = read_waveform(example["audio"])
        durations_s.append(w.numel() / SAMPLE_RATE)

    durations_s = np.array(durations_s)

    print(f"Nombre de fichiers : {len(durations_s)}")
    print(f"Min / Max           : {durations_s.min():.2f}s / {durations_s.max():.2f}s")
    print(f"Moyenne / Médiane    : {durations_s.mean():.2f}s / {np.median(durations_s):.2f}s")
    for p in (50, 75, 90, 95, 99):
        print(f"{p}e percentile       : {np.percentile(durations_s, p):.2f}s")


if __name__ == "__main__":
    main()