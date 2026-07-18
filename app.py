# ============================================================
# CAPSTONE MAIN APPLICATION
# Intelligent Healthcare Diagnostic Assistant
# Introduction to AI — 13-Week Capstone
# ============================================================

import sys
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

# Import all modules
from modules.agent          import HealthcareDiagnosticAgent, PatientPercept
from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net   import SimpleBayesianDiagnostics
from modules.ml_classifier  import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner        import TreatmentPlanner

# ── ANSI Colors ────────────────────────────────────────────
class C:
    HEADER = '\033[95m'; BLUE   = '\033[94m'
    GREEN  = '\033[92m'; YELLOW = '\033[93m'
    RED    = '\033[91m'; BOLD   = '\033[1m'
    END    = '\033[0m'

def banner():
    print(f"""
{C.BOLD}{C.BLUE}
╔══════════════════════════════════════════════════════════╗
║        🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI           ║
║         Introduction to AI — Capstone Project            ║
║  Modules: Agents | Logic | Bayes | ML | DNN | Fuzzy      ║
╚══════════════════════════════════════════════════════════╝
{C.END}""")

def section(title: str):
    print(f"\n{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")
    print(f"{C.BOLD}{C.YELLOW}  {title}{C.END}")
    print(f"{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")

def build_system() -> HealthcareDiagnosticAgent:
    """Instantiate and wire all AI modules"""
    section("🔧 Building AI System — Registering Modules")

    agent = HealthcareDiagnosticAgent()

    print("\n  Initializing modules...")
    modules = {
        'KnowledgeBase': MedicalKnowledgeBase(),
        'BayesianNet':   SimpleBayesianDiagnostics(),
        'MLClassifier':  MLDiagnosticClassifier(),
        'NeuralNetwork': NeuralDiagnosticModel(),
        'FuzzyLogic':    FuzzySeverityAssessor(),
    }

    for name, module in modules.items():
        agent.register_module(name, module)

    # Pre-train the learning-based modules once, up front, so the
    # first patient run doesn't stall on training.
    print("\n  Pre-training ML classifier (Decision Tree / RF / GB)...")
    modules['MLClassifier'].train(verbose=False)

    print("  Pre-training neural network (this can take a minute)...")
    modules['NeuralNetwork'].train(epochs=30, verbose=0)

    print(f"\n  {C.GREEN}✅ All modules registered and ready.{C.END}")
    return agent


def print_report(report: dict, plan: dict):
    """Pretty-print the agent's final diagnostic report + treatment plan."""
    section(f"📋 FINAL REPORT — Patient {report['patient_id']}")

    print(f"  Symptoms         : {', '.join(report['symptoms'])}")
    print(f"  Diagnosis        : {C.BOLD}{report['diagnosis']}{C.END}")
    print(f"  Confidence       : {report['confidence']:.2%}")
    print(f"  Urgency          : {C.RED if report['urgency']=='CRITICAL' else C.YELLOW}"
          f"{report['urgency']}{C.END}")
    print(f"  Next Action      : {report['next_action']}")

    print(f"\n  Recommendations:")
    for r in report['recommendations']:
        print(f"    - {r}")

    if plan and plan.get('plan'):
        print(f"\n  Treatment Plan ({plan['steps']} steps):")
        for step in plan['plan']:
            print(f"    Step {step['step']:2d}: {step['action']:<28} [{step['duration']}]")
    elif plan:
        print(f"\n  Treatment Plan: {plan.get('error', 'unavailable')}")


# NOTE / KNOWN LIMITATION: the sub-modules don't share one disease
# vocabulary -- knowledge_base.py emits suffixed forms like
# "cardiac_event_suspected"/"covid19_confirmed", bayesian_net.py uses
# short forms like "cardiac", while ml_classifier.py/neural_network.py
# use "cardiac_event". This means the agent's majority vote in
# _aggregate_diagnosis() under-counts real agreement across modules.
# This map normalizes any module's output to the canonical label the
# planner understands, without needing to rewrite each module's
# internal label choices.
DIAGNOSIS_ALIASES = {
    'cardiac': 'cardiac_event',
    'healthy': 'common_cold',   # closest low-severity fallback
}

def normalize_diagnosis(label: str) -> str:
    label = (label or '').lower().strip()
    for suffix in ('_suspected', '_confirmed'):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    return DIAGNOSIS_ALIASES.get(label, label)


def run_patient(agent: HealthcareDiagnosticAgent, planner: TreatmentPlanner,
                percept: PatientPercept) -> dict:
    """Run one patient through the full Perceive -> Think -> Act pipeline,
    then generate a treatment plan from the resulting diagnosis."""
    section(f"🧑‍⚕️ Processing Patient {percept.patient_id}")

    report = agent.run(percept)
    canonical_diagnosis = normalize_diagnosis(report['diagnosis'])
    plan = planner.create_treatment_plan(canonical_diagnosis, report['urgency'])
    print_report(report, plan)
    return {'report': report, 'plan': plan}


def demo_patients():
    """A handful of realistic test cases covering different diagnoses/urgencies."""
    return [
        PatientPercept(
            patient_id="P001", symptoms=["fever", "cough", "fatigue", "loss_of_smell"],
            age=34, temperature=38.9, heart_rate=98, blood_pressure="120/80"),
        PatientPercept(
            patient_id="P002", symptoms=["cough", "fever", "headache", "body_aches"],
            age=45, temperature=38.2, heart_rate=88, blood_pressure="118/76"),
        PatientPercept(
            patient_id="P003", symptoms=["fever", "rash", "joint_pain", "headache"],
            age=27, temperature=39.4, heart_rate=105, blood_pressure="110/70"),
        PatientPercept(
            patient_id="P004", symptoms=["chest_pain", "shortness_of_breath", "sweating"],
            age=61, temperature=37.4, heart_rate=125, blood_pressure="150/95"),
        PatientPercept(
            patient_id="P005", symptoms=["fatigue", "frequent_urination", "excessive_thirst",
                                          "blurred_vision"],
            age=52, temperature=37.0, heart_rate=80, blood_pressure="135/85"),
    ]


def main():
    banner()
    agent = build_system()
    planner = TreatmentPlanner()

    results = []
    for percept in demo_patients():
        results.append(run_patient(agent, planner, percept))

    section("📊 Session Summary")
    perf = agent.get_performance()
    print(f"  Patients processed : {perf['total_patients']}")
    print(f"  Diagnoses made      : {perf['diagnoses_made']}")
    print(f"  Performance score   : {perf['performance_score']}")

    agent.print_log()

    return results


if __name__ == "__main__":
    main()
