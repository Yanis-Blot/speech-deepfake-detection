"""Métriques d'évaluation — COMMUN à tous les modèles.

Étape 2 : l'EER (Equal Error Rate), la métrique standard en anti-spoofing.
"""

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(labels, scores):
    """Calcule l'EER à partir des vrais labels et des scores prédits.

    labels : 0 = bonafide (vraie voix), 1 = spoof (fausse voix)
    scores : score de "spoof" prédit par le modèle (plus haut = plus suspect)

    Retourne (eer, threshold) : l'EER (entre 0 et 1) et le seuil de décision
    qui l'atteint.
    """
    # roc_curve attend un label "positif" : ici le spoof (1)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    # fpr = taux de fausses alarmes : des vraies voix classées spoof à tort
    # tpr = taux de vrais positifs : des spoof bien détectés
    # donc (1 - tpr) = taux de spoof RATÉS (classés bonafide à tort)
    fnr = 1 - tpr

    # l'EER est le point où "on rejette trop de vraies voix" (fpr)
    # = "on laisse passer trop de fausses voix" (fnr)
    # on cherche où les deux courbes se croisent
    diff = fpr - fnr
    idx = np.nanargmin(np.abs(diff))

    eer = (fpr[idx] + fnr[idx]) / 2
    threshold = thresholds[idx]
    return float(eer), float(threshold)

def per_attack_eer(labels, scores, system_ids):
    """EER calculé séparément pour chaque type d'attaque, en combinant à chaque fois
    tous les bonafide avec UN SEUL type de spoof à la fois."""
    import numpy as np
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    system_ids = np.asarray(system_ids)

    bonafide_mask = labels == 0
    attack_ids = sorted(set(system_ids[~bonafide_mask]))

    results = {}
    for attack_id in attack_ids:
        attack_mask = system_ids == attack_id
        subset_mask = bonafide_mask | attack_mask
        eer, _ = compute_eer(labels[subset_mask], scores[subset_mask])
        results[attack_id] = eer
    return results

