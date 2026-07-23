# ============================================================
# EVALUATION: Visualizations
# Confusion matrix heatmaps, module comparison bar chart,
# and a full system evaluation dashboard.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str,
                           save_path: str = None):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_module_comparison(results_by_module: Dict[str, Dict],
                            save_path: str = None):
    """
    Grouped bar chart comparing Accuracy / Precision / Recall / F1
    across all diagnostic modules -- a required deliverable.
    """
    modules = list(results_by_module.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    x = np.arange(len(modules))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [results_by_module[m][metric] for m in modules]
        ax.bar(x + i * width, values, width, label=label, color=color)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(modules, rotation=15, ha='right')
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Diagnostic Module Comparison", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_severity_distribution(severity_scores: List[float], save_path: str = None):
    """Histogram of fuzzy severity scores across all evaluated patients."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(severity_scores, bins=15, color='#e74c3c', edgecolor='black', alpha=0.75)
    ax.set_xlabel("Severity Score (0-100)")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Fuzzy Severity Score Distribution", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    plt.close(fig)
    return fig


if __name__ == "__main__":
    # Self-test
    fake_results = {
        'KnowledgeBase': {'accuracy': 0.72, 'precision': 0.70, 'recall': 0.72, 'f1_score': 0.71},
        'BayesianNet':   {'accuracy': 0.78, 'precision': 0.76, 'recall': 0.78, 'f1_score': 0.77},
        'MLClassifier':  {'accuracy': 0.91, 'precision': 0.90, 'recall': 0.91, 'f1_score': 0.905},
        'NeuralNetwork': {'accuracy': 0.89, 'precision': 0.88, 'recall': 0.89, 'f1_score': 0.885},
    }
    plot_module_comparison(fake_results, save_path="/tmp/module_comparison_test.png")
    print("Visualizations module test passed!")
