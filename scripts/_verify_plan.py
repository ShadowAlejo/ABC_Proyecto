import sys, numpy as np
sys.path.insert(0, '.')

print("=== A) Pipeline completo de inferencia ===")
from classification.svm_facial_model import SVMFacialModel
from classification.logistic_confidence_converter import decision_margin_to_probability
from decision_engine.threshold_acceptance_gate import ThresholdAcceptanceGate
from decision_engine.track_identity_state import TrackIdentityState
from decision_engine.unknown_labeler import label_unknown

m = SVMFacialModel()
m.load()
print(f"  Modelo cargado: {type(m.model).__name__}")
print(f"  Clases ({len(m.class_names)}): {m.class_names}")

rng = np.random.default_rng(42)
unknown_vector = rng.standard_normal(10188).astype(np.float32)
identity, gap = m.predict(unknown_vector)
prob = decision_margin_to_probability(gap)
print(f"  Persona desconocida (ruido): identity={identity}, gap={gap:.4f}, prob={prob:.4f}")

gate = ThresholdAcceptanceGate()
accepted = gate.accept(prob)
print(f"  Threshold={gate.t_aceptacion}, accepted={accepted}  (esperado: False)")

print()
print("=== B) Verificacion margin_gap con multiples clases ===")
decision_clear = np.array([3.8, 0.5, -1.2, 0.1, -0.3, 0.0, -0.8, 0.2, -0.5, -1.0, -0.7, 0.3, -0.2, -0.9, -0.4])
top2 = np.partition(decision_clear, -2)[-2:]
gap_clear = float(top2[-1] - top2[-2])
prob_clear = decision_margin_to_probability(gap_clear)
print(f"  Gap claro (3.8 vs 0.5):   gap={gap_clear:.4f}, prob={prob_clear:.4f}  (esperado: alta)")

decision_uncertain = np.array([2.3, 2.1, -0.8, 0.1, -0.3, 0.0, -0.8, 0.2, -0.5, -1.0, -0.7, 0.3, -0.2, -0.9, -0.4])
top2u = np.partition(decision_uncertain, -2)[-2:]
gap_uncertain = float(top2u[-1] - top2u[-2])
prob_uncertain = decision_margin_to_probability(gap_uncertain)
print(f"  Gap incierto (2.3 vs 2.1): gap={gap_uncertain:.4f}, prob={prob_uncertain:.4f}  (esperado: baja)")

print()
print("=== C) Bug fix en TrackIdentityState ===")
state = TrackIdentityState()
UNKNOWN = label_unknown()

result = state.resolve(track_id=1, branch="ID", candidate_identity="Alejandro_Torres", accepted=False)
print(f"  Sin historial, no aceptado: '{result}'  (esperado: '{UNKNOWN}')")
assert result == UNKNOWN, f"BUG NO RESUELTO: got '{result}'"

result2 = state.resolve(track_id=1, branch="ID", candidate_identity="Alejandro_Torres", accepted=True)
print(f"  Con aceptacion:             '{result2}'  (esperado: 'Alejandro_Torres')")
assert result2 == "Alejandro_Torres"

result3 = state.resolve(track_id=1, branch="ID", candidate_identity="Desconocido", accepted=False)
print(f"  Frame sin cara (historial): '{result3}'  (esperado: 'Alejandro_Torres')")
assert result3 == "Alejandro_Torres"

print()
print("TODOS LOS TESTS PASARON.")
