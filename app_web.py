# ============================================================
# WEB INTERFACE — Intelligent Healthcare Diagnostic Assistant
# Wraps the existing 7-module system in a Streamlit browser UI.
# Run with:  streamlit run app_web.py
# ============================================================

import streamlit as st

from modules.agent import HealthcareDiagnosticAgent, PatientPercept
from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.ml_classifier import MLDiagnosticClassifier
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner import TreatmentPlanner

try:
    from modules.neural_network import NeuralDiagnosticModel
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# ------------------------------------------------------------
# Disease-label normalization (same fix used in app.py)
# ------------------------------------------------------------
DIAGNOSIS_ALIASES = {'cardiac': 'cardiac_event', 'healthy': 'common_cold'}


def normalize_diagnosis(label: str) -> str:
    label = (label or '').lower().strip()
    for suffix in ('_suspected', '_confirmed'):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    return DIAGNOSIS_ALIASES.get(label, label)


# ------------------------------------------------------------
# Build the system once per server session (training is slow,
# so this is cached rather than re-run on every click).
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Starting up the diagnostic system "
                                 "(training models, first load only)...")
def build_system():
    agent = HealthcareDiagnosticAgent()
    modules = {
        'KnowledgeBase': MedicalKnowledgeBase(),
        'BayesianNet': SimpleBayesianDiagnostics(),
        'MLClassifier': MLDiagnosticClassifier(),
        'FuzzyLogic': FuzzySeverityAssessor(),
    }
    if HAS_TENSORFLOW:
        modules['NeuralNetwork'] = NeuralDiagnosticModel()

    for name, module in modules.items():
        agent.register_module(name, module)

    modules['MLClassifier'].train(verbose=False)
    if HAS_TENSORFLOW:
        modules['NeuralNetwork'].train(epochs=30, verbose=0)

    planner = TreatmentPlanner()
    return agent, planner


SYMPTOM_OPTIONS = [
    'fever', 'cough', 'fatigue', 'headache', 'body_aches', 'loss_of_smell',
    'chest_pain', 'rash', 'joint_pain', 'shortness_of_breath', 'sweating',
    'frequent_urination', 'excessive_thirst', 'blurred_vision',
    'night_sweats', 'weight_loss', 'stiff_neck', 'light_sensitivity',
]

URGENCY_COLORS = {
    'CRITICAL': '#c0392b',
    'HIGH': '#e67e22',
    'MEDIUM': '#f1c40f',
    'LOW': '#27ae60',
}

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Healthcare Diagnostic Assistant",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 Intelligent Healthcare Diagnostic Assistant")
st.caption(
    "AI capstone system combining an intelligent agent, logic-based inference, "
    "Bayesian reasoning, machine learning, deep learning, fuzzy logic, and "
    "automated treatment planning."
)

if not HAS_TENSORFLOW:
    st.warning(
        "TensorFlow isn't installed in this environment, so the Neural Network "
        "module is running with 4 of 5 diagnostic modules instead of 5. "
        "Run `pip install tensorflow` and restart to include it.",
        icon="⚠️",
    )

agent, planner = build_system()

st.divider()

# ------------------------------------------------------------
# Patient input form
# ------------------------------------------------------------
st.subheader("Patient Information")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID", value="P001")
        age = st.number_input("Age", min_value=0, max_value=120, value=35)
        temperature = st.slider("Temperature (°C)", 35.0, 42.0, 37.0, 0.1)
    with col2:
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30,
                                      max_value=220, value=80)
        blood_pressure = st.text_input("Blood Pressure", value="120/80")

    symptoms = st.multiselect(
        "Symptoms (select all that apply)",
        options=SYMPTOM_OPTIONS,
        help="These map directly to the 18 symptom features the ML and "
             "Neural Network modules were trained on.",
    )

    submitted = st.form_submit_button("Run Diagnosis", type="primary",
                                       use_container_width=True)

# ------------------------------------------------------------
# Run diagnosis
# ------------------------------------------------------------
if submitted:
    if not symptoms:
        st.error("Select at least one symptom before running a diagnosis.")
    else:
        percept = PatientPercept(
            patient_id=patient_id or "P001",
            symptoms=symptoms,
            age=int(age),
            temperature=float(temperature),
            heart_rate=int(heart_rate),
            blood_pressure=blood_pressure or "120/80",
        )

        with st.spinner("Running patient through all diagnostic modules..."):
            report = agent.run(percept)
            canonical_diagnosis = normalize_diagnosis(report['diagnosis'])
            plan = planner.create_treatment_plan(canonical_diagnosis, report['urgency'])

        st.divider()
        st.subheader(f"Report — Patient {report['patient_id']}")

        urgency = report['urgency']
        color = URGENCY_COLORS.get(urgency, '#7f8c8d')
        st.markdown(
            f"<div style='padding:12px 18px;border-radius:8px;background:{color};"
            f"color:white;font-weight:600;display:inline-block;'>"
            f"Urgency: {urgency}</div>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        c1.metric("Diagnosis", report['diagnosis'])
        c2.metric("Confidence", f"{report['confidence']:.1%}")

        st.markdown(f"**Next action:** {report['next_action']}")

        st.markdown("**Recommendations:**")
        for r in report['recommendations']:
            st.markdown(f"- {r}")

        st.markdown("**Treatment Plan:**")
        if plan and plan.get('plan'):
            st.table([
                {"Step": s['step'], "Action": s['action'], "Duration": s['duration']}
                for s in plan['plan']
            ])
        else:
            st.info(plan.get('error', 'No plan available') if plan else 'No plan available')

        with st.expander("See individual module opinions"):
            # agent.py stores each module's raw result in
            # agent.memory.diagnosis_history (populated by think()) rather
            # than in the final report dict itself.
            if agent.memory.diagnosis_history:
                for name, result in agent.memory.diagnosis_history[-1].items():
                    st.markdown(f"**{name}**")
                    st.json(result, expanded=False)

st.divider()
st.caption(
    "This system uses synthetic training data and is a capstone "
    "proof-of-concept, not a validated medical device."
)
