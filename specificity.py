from sklearn.metrics import recall_score

def specificity(y_true, y_pred):
    """
    Oblicza specyficzność (specificity) na podstawie etykiet prawdziwych i przewidywanych.

    Args:
        y_true (array-like): Rzeczywiste etykiety klas.
        y_pred (array-like): Przewidywane etykiety klas.

    Returns:
        float: Wartość specyficzności.
    """
    # Używamy recall_score z pos_label=0 do obliczenia specificity
    return recall_score(y_true, y_pred, pos_label=0)

import numpy as np
from sklearn.metrics import confusion_matrix

def specificity_macro(y_true, y_pred):
    """
    Multiclass macro-specificity (one-vs-rest).
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    specificity_per_class = []

    for i in range(n_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        spec = TN / (TN + FP + 1e-12)
        specificity_per_class.append(spec)

    return float(np.mean(specificity_per_class))