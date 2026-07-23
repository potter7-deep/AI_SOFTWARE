# ============================================================
# EVALUATION: Metrics
# Computes accuracy, precision, recall, F1, confusion matrix,
# and ROC-AUC for each learning-based diagnostic module.
# ============================================================

import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)


def evaluate_predictions(y_true: List[str], y_pred: List[str],
                          labels: List[str] = None) -> Dict:
    """
    Compute the standard classification metrics for one module's
    predictions against ground-truth labels.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        'accuracy': round(float(accuracy), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_score': round(float(f1), 4),
        'confusion_matrix': cm.tolist(),
        'labels': labels,
    }


def evaluate_module_with_proba(y_true_idx: np.ndarray, y_proba: np.ndarray,
                                labels: List[str]) -> Dict:
    """
    Additionally compute a multi-class ROC-AUC (one-vs-rest) when the
    module can produce class probabilities (ML classifier / neural net).
    """
    try:
        auc = roc_auc_score(y_true_idx, y_proba, multi_class='ovr',
                             labels=list(range(len(labels))))
    except Exception:
        auc = None
    return {'roc_auc_ovr': round(float(auc), 4) if auc is not None else None}


def compare_modules(results_by_module: Dict[str, Dict]) -> str:
    """
    Build a simple text table comparing accuracy/precision/recall/F1
    across modules -- used both in the console report and the final
    written report.
    """
    header = f"{'Module':<18}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}"
    lines = [header, "-" * len(header)]
    for name, m in results_by_module.items():
        lines.append(
            f"{name:<18}{m['accuracy']:>10.4f}{m['precision']:>12.4f}"
            f"{m['recall']:>10.4f}{m['f1_score']:>10.4f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-test with a tiny synthetic example
    y_true = ["flu", "covid19", "flu", "dengue", "covid19"]
    y_pred = ["flu", "covid19", "covid19", "dengue", "covid19"]
    result = evaluate_predictions(y_true, y_pred)
    print("Accuracy :", result['accuracy'])
    print("Precision:", result['precision'])
    print("Recall   :", result['recall'])
    print("F1       :", result['f1_score'])
    print("Confusion Matrix:", result['confusion_matrix'])
    print("Evaluation module test passed!")
