# ============================================================
# EVALUATION: Full System Evaluation Script
# Generates a labeled synthetic test set, runs every diagnostic
# module against it, computes metrics, and saves all required
# deliverable charts into reports/.
# ============================================================

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.agent import PatientPercept
from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.ml_classifier import MLDiagnosticClassifier
from modules.fuzzy_controller import FuzzySeverityAssessor
from evaluation.metrics import evaluate_predictions, compare_modules
from evaluation.visualizations import plot_confusion_matrix, plot_module_comparison

try:
    from modules.neural_network import NeuralDiagnosticModel
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("⚠️  TensorFlow not installed here — Neural Network module will be "
          "skipped in this evaluation run. Install `tensorflow` and re-run "
          "to include it.")


# Disease profiles used to generate a labeled synthetic test set (mirrors
# ml_classifier.py's own synthetic data so ground truth is well-defined).
TEST_PROFILES = {
    'flu':           {'fever': 0.90, 'cough': 0.85, 'fatigue': 0.88, 'headache': 0.70,
                       'body_aches': 0.80},
    'covid19':       {'fever': 0.88, 'cough': 0.80, 'fatigue': 0.90, 'loss_of_smell': 0.85,
                       'headache': 0.65},
    'dengue':        {'fever': 0.98, 'rash': 0.75, 'joint_pain': 0.85, 'headache': 0.90,
                       'fatigue': 0.80},
    'cardiac_event': {'chest_pain': 0.92, 'shortness_of_breath': 0.88, 'fatigue': 0.70,
                       'sweating': 0.75},
    'diabetes':      {'fatigue': 0.82, 'frequent_urination': 0.95, 'excessive_thirst': 0.92,
                       'blurred_vision': 0.70},
    'common_cold':   {'cough': 0.90, 'fever': 0.50, 'headache': 0.60, 'fatigue': 0.55},
    'tuberculosis':  {'cough': 0.95, 'weight_loss': 0.85, 'night_sweats': 0.80, 'fatigue': 0.88},
    'meningitis':    {'headache': 0.95, 'stiff_neck': 0.90, 'fever': 0.92,
                       'light_sensitivity': 0.85},
}

DIAGNOSIS_ALIASES = {'cardiac': 'cardiac_event', 'healthy': 'common_cold'}


def normalize(label: str) -> str:
    label = (label or '').lower().strip()
    for suffix in ('_suspected', '_confirmed'):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    return DIAGNOSIS_ALIASES.get(label, label)


def generate_test_patients(n_per_class: int = 15, seed: int = 7):
    rng = np.random.default_rng(seed)
    patients, labels = [], []
    pid = 1
    for disease, probs in TEST_PROFILES.items():
        for _ in range(n_per_class):
            symptoms = [s for s, p in probs.items() if rng.random() < p]
            if not symptoms:  # guarantee at least one symptom
                symptoms = [rng.choice(list(probs.keys()))]
            temp = float(rng.normal(38.5 if 'fever' in symptoms else 37.0, 0.6))
            hr = int(rng.normal(100 if 'chest_pain' in symptoms else 85, 10))
            patients.append(PatientPercept(
                patient_id=f"T{pid:03d}", symptoms=symptoms, age=int(rng.integers(18, 75)),
                temperature=round(temp, 1), heart_rate=max(50, hr),
                blood_pressure="120/80"))
            labels.append(disease)
            pid += 1
    return patients, labels


def run_evaluation():
    print("=" * 60)
    print("  FULL SYSTEM EVALUATION")
    print("=" * 60)

    patients, y_true = generate_test_patients()
    print(f"\nGenerated {len(patients)} labeled synthetic test patients "
          f"across {len(TEST_PROFILES)} diseases.\n")

    kb = MedicalKnowledgeBase()
    bn = SimpleBayesianDiagnostics()
    ml = MLDiagnosticClassifier()
    ml.train(verbose=False)
    fz = FuzzySeverityAssessor()

    modules = {'KnowledgeBase': kb, 'BayesianNet': bn, 'MLClassifier': ml}
    if HAS_TENSORFLOW:
        nn = NeuralDiagnosticModel()
        nn.train(epochs=30, verbose=0)
        modules['NeuralNetwork'] = nn

    predictions = {name: [] for name in modules}
    severity_scores = []

    for patient in patients:
        for name, module in modules.items():
            result = module.analyze(patient)
            predictions[name].append(normalize(result.get('diagnosis')))
        severity_scores.append(fz.assess(
            patient.temperature, patient.heart_rate, len(patient.symptoms)
        )['severity_score'])

    os.makedirs("reports", exist_ok=True)
    results_by_module = {}
    all_labels = sorted(TEST_PROFILES.keys())

    for name, y_pred in predictions.items():
        metrics = evaluate_predictions(y_true, y_pred, labels=all_labels)
        results_by_module[name] = metrics
        print(f"\n--- {name} ---")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-Score : {metrics['f1_score']:.4f}")

        cm = np.array(metrics['confusion_matrix'])
        plot_confusion_matrix(
            cm, all_labels, f"Confusion Matrix — {name}",
            save_path=f"reports/confusion_matrix_{name.lower()}.png"
        )

    print("\n" + "=" * 60)
    print(compare_modules(results_by_module))
    print("=" * 60)

    plot_module_comparison(results_by_module, save_path="reports/module_comparison.png")

    print(f"\n✅ Evaluation complete. Charts saved to reports/")
    return results_by_module


if __name__ == "__main__":
    run_evaluation()
