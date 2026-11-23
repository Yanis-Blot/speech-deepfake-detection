"""Baseline triviale : est-ce que la durée du silence (début + fin) suffit à détecter le spoof ?

On teste plusieurs seuils d'amplitude, car le résultat peut être sensible à ce choix.
"""

from src.data import load_splits, read_waveform, leading_silence_duration, trailing_silence_duration
from src.metrics import compute_eer

THRESHOLDS_A_TESTER = [0.005, 0.01, 0.02, 0.05]


def main():
    ds = load_splits()
    validation = ds["validation"]

    # on lit chaque waveform UNE SEULE FOIS, pas une fois par seuil (sinon on relit 25000 fichiers x4)
    waveforms = []
    labels = []
    for example in validation:
        waveforms.append(read_waveform(example["audio"]))
        labels.append(example["key"])

    for threshold in THRESHOLDS_A_TESTER:
        scores_debut_seul = []
        scores_debut_fin = []
        for w in waveforms:
            debut = leading_silence_duration(w, threshold=threshold)
            fin = trailing_silence_duration(w, threshold=threshold)
            scores_debut_seul.append(-debut)
            scores_debut_fin.append(-(debut + fin))

        eer_debut, _ = compute_eer(labels, scores_debut_seul)
        eer_total, _ = compute_eer(labels, scores_debut_fin)
        print(f"seuil={threshold:<6} | EER (début seul) = {eer_debut:.4f} | EER (début+fin) = {eer_total:.4f}")


if __name__ == "__main__":
    main()