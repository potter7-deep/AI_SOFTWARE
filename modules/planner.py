"""
MODULE 7 — AI Treatment Planner
modules/planner.py

Generates a step-by-step treatment plan (a sequence of STRIPS actions)
from a patient's initial state to a goal state, using breadth-first
search over the action state-space.

Design note: instead of writing one-off actions per disease
("OrderPCRTest" only usable for COVID), the action library here is
GENERIC — it operates on disease-agnostic predicates (SUSPECTED,
CONTAGIOUS, DIAGNOSIS_CONFIRMED, ...). Each disease just sets different
flags in its initial state (contagious or not) and each urgency level
sets different flags too (needs ICU, needs a specialist). That's what
lets the same ~10 actions correctly plan for 8 diseases x 4 urgency
levels (32 combinations) instead of needing 32 hand-written paths.
"""

from collections import deque
from dataclasses import dataclass, field


# ---------------------------------------------------------------------- #
# STRIPS action library
# ---------------------------------------------------------------------- #

ACTIONS = [
    {
        'name': 'IsolatePatient',
        'precond': frozenset({'PATIENT_PRESENT', 'CONTAGIOUS'}),
        'delete':  frozenset({'CONTAGIOUS'}),
        'add':     frozenset({'ISOLATED'}),
        'cost': 1, 'duration': '14 days',
    },
    {
        'name': 'OrderDiagnosticTest',
        'precond': frozenset({'PATIENT_PRESENT', 'SUSPECTED'}),
        'delete':  frozenset({'SUSPECTED'}),
        'add':     frozenset({'TEST_PENDING'}),
        'cost': 1, 'duration': '24 hours',
    },
    {
        # Extension: alternate diagnostic route (see "Extending the
        # planner" below) — used by typhoid instead of the generic test.
        'name': 'OrderBloodCultureTest',
        'precond': frozenset({'PATIENT_PRESENT', 'SUSPECTED', 'NEEDS_BLOOD_CULTURE'}),
        'delete':  frozenset({'SUSPECTED', 'NEEDS_BLOOD_CULTURE'}),
        'add':     frozenset({'TEST_PENDING'}),
        'cost': 1, 'duration': '48-72 hours',
    },
    {
        'name': 'ReceiveTestResult',
        'precond': frozenset({'TEST_PENDING'}),
        'delete':  frozenset({'TEST_PENDING'}),
        'add':     frozenset({'DIAGNOSIS_CONFIRMED'}),
        'cost': 1, 'duration': '24 hours',
    },
    {
        'name': 'ConsultSpecialist',
        'precond': frozenset({'DIAGNOSIS_CONFIRMED', 'SPECIALIST_NEEDED'}),
        'delete':  frozenset({'SPECIALIST_NEEDED'}),
        'add':     frozenset({'SPECIALIST_CONSULTED'}),
        'cost': 2, 'duration': '2 hours',
    },
    {
        'name': 'AdmitToICU',
        'precond': frozenset({'DIAGNOSIS_CONFIRMED', 'CRITICAL_CARE_NEEDED'}),
        'delete':  frozenset({'CRITICAL_CARE_NEEDED'}),
        'add':     frozenset({'ICU_ADMITTED'}),
        'cost': 3, 'duration': 'Immediate',
    },
    {
        'name': 'PrescribeMedication',
        'precond': frozenset({'DIAGNOSIS_CONFIRMED'}),
        'delete':  frozenset(),
        'add':     frozenset({'MEDICATION_PRESCRIBED'}),
        'cost': 1, 'duration': '10 minutes',
    },
    {
        'name': 'StartTreatment',
        'precond': frozenset({'MEDICATION_PRESCRIBED'}),
        'delete':  frozenset(),
        'add':     frozenset({'TREATMENT_STARTED'}),
        'cost': 1, 'duration': '10 minutes',
    },
    {
        'name': 'MonitorVitals',
        'precond': frozenset({'TREATMENT_STARTED'}),
        'delete':  frozenset(),
        'add':     frozenset({'VITALS_MONITORED'}),
        'cost': 1, 'duration': 'Continuous',
    },
    {
        'name': 'ScheduleFollowUp',
        'precond': frozenset({'VITALS_MONITORED'}),
        'delete':  frozenset(),
        'add':     frozenset({'FOLLOWUP_SCHEDULED'}),
        'cost': 1, 'duration': '5 minutes',
    },
]

# ---------------------------------------------------------------------- #
# Disease + urgency -> initial state
# ---------------------------------------------------------------------- #

# contagious=True adds the CONTAGIOUS flag so IsolatePatient becomes
# reachable/necessary; needs_blood_culture routes through the alternate
# diagnostic action instead of the generic one.
DISEASE_PROFILES = {
    'covid19':       {'contagious': True,  'needs_blood_culture': False},
    'flu':           {'contagious': True,  'needs_blood_culture': False},
    'common_cold':   {'contagious': True,  'needs_blood_culture': False},
    'tuberculosis':  {'contagious': True,  'needs_blood_culture': False},
    'strep_throat':  {'contagious': True,  'needs_blood_culture': False},
    'dengue':        {'contagious': False, 'needs_blood_culture': False},
    'malaria':       {'contagious': False, 'needs_blood_culture': False},
    'pneumonia':     {'contagious': False, 'needs_blood_culture': False},
    # --- extension: new disease pathway, added on top of the original 8 ---
    'typhoid':       {'contagious': False, 'needs_blood_culture': True},
}

URGENCY_LEVELS = ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')


class TreatmentPlanner:
    """STRIPS-style treatment planner using BFS over the action state-space."""

    def __init__(self, actions=None):
        self.actions = actions if actions is not None else ACTIONS

    # ------------------------------------------------------------------ #
    # STRIPS action application
    # ------------------------------------------------------------------ #

    def _apply_action(self, state: frozenset, action: dict):
        if not action['precond'].issubset(state):
            return None  # preconditions not satisfied — cannot apply
        new_state = (state - action['delete']) | action['add']
        return frozenset(new_state)  # MUST be frozenset: it goes into a set/dict key

    # ------------------------------------------------------------------ #
    # BFS planner
    # ------------------------------------------------------------------ #

    def generate_plan(self, initial_state: frozenset, goal: frozenset, max_depth: int = 12):
        initial_state = frozenset(initial_state)
        goal = frozenset(goal)

        if goal.issubset(initial_state):
            return []

        queue = deque([(initial_state, [])])
        visited = {initial_state}

        while queue:
            state, plan = queue.popleft()
            if len(plan) >= max_depth:
                continue
            for action in self.actions:
                new_state = self._apply_action(state, action)
                if new_state is None or new_state in visited:
                    continue
                new_plan = plan + [action]
                if goal.issubset(new_state):
                    return new_plan
                visited.add(new_state)
                queue.append((new_state, new_plan))

        return None  # no plan found within max_depth

    # ------------------------------------------------------------------ #
    # Disease/urgency -> initial state + goal, then plan
    # ------------------------------------------------------------------ #

    def _build_initial_state(self, disease: str, urgency: str) -> frozenset:
        profile = DISEASE_PROFILES[disease]
        state = {'PATIENT_PRESENT', 'SUSPECTED'}
        if profile['contagious']:
            state.add('CONTAGIOUS')
        if profile['needs_blood_culture']:
            state.add('NEEDS_BLOOD_CULTURE')
        if urgency in ('HIGH', 'CRITICAL'):
            state.add('SPECIALIST_NEEDED')
        if urgency == 'CRITICAL':
            state.add('CRITICAL_CARE_NEEDED')
        return frozenset(state)

    def _build_goal(self, disease: str, urgency: str) -> frozenset:
        goal = {'TREATMENT_STARTED', 'VITALS_MONITORED', 'FOLLOWUP_SCHEDULED'}
        if DISEASE_PROFILES[disease]['contagious']:
            # Without this, BFS's shortest-plan bias skips IsolatePatient
            # entirely, since nothing else in the goal depends on it.
            goal.add('ISOLATED')
        if urgency in ('HIGH', 'CRITICAL'):
            goal.add('SPECIALIST_CONSULTED')
        if urgency == 'CRITICAL':
            goal.add('ICU_ADMITTED')
        return frozenset(goal)

    def create_treatment_plan(self, disease: str, urgency: str) -> dict:
        if disease not in DISEASE_PROFILES:
            raise ValueError(f"Unknown disease: {disease}")
        if urgency not in URGENCY_LEVELS:
            raise ValueError(f"Unknown urgency level: {urgency}")

        initial_state = self._build_initial_state(disease, urgency)
        goal = self._build_goal(disease, urgency)
        plan = self.generate_plan(initial_state, goal)

        if plan is None:
            return {
                'diagnosis': disease, 'urgency': urgency,
                'success': False, 'steps': 0, 'plan': [],
            }

        return {
            'diagnosis': disease,
            'urgency': urgency,
            'success': True,
            'steps': len(plan),
            'plan': [
                {'step': i + 1, 'action': a['name'], 'duration': a['duration']}
                for i, a in enumerate(plan)
            ],
        }


# ---------------------------------------------------------------------- #
# Exhaustive test: every disease x every urgency level
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    planner = TreatmentPlanner()

    print("=== Single example: covid19 / HIGH ===")
    result = planner.create_treatment_plan('covid19', 'HIGH')
    print(f"Diagnosis : {result['diagnosis']}")
    print(f"Plan Steps: {result['steps']}")
    for step in result['plan']:
        print(f"  Step {step['step']:2d}: {step['action']:<24} [{step['duration']}]")

    print("\n=== Exhaustive test: all diseases x all urgency levels ===")
    total, passed = 0, 0
    for disease in DISEASE_PROFILES:
        for urgency in URGENCY_LEVELS:
            total += 1
            result = planner.create_treatment_plan(disease, urgency)
            ok = result['success']
            passed += int(ok)
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {disease:<14} / {urgency:<8} -> {result['steps']} steps")

    print(f"\n{passed}/{total} diagnosis-urgency combinations produced a valid plan.")
