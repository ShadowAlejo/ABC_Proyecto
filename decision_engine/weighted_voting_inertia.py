"""Acumula predicciones de ambas ramas por Track-ID mediante votación ponderada por confianza [REQ-DEC-01]."""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class _TrackVotes:
    weighted_scores: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    total_weight: float = 0.0


class WeightedVotingInertia:
    """Mantiene la inercia de votación ponderada por Track-ID a lo largo del tiempo."""

    def __init__(self, decay: float = 0.98):
        """decay: factor de olvido aplicado en cada acumulación para dar más peso a votos recientes."""
        self.decay = decay
        self._votes: Dict[int, _TrackVotes] = {}

    def accumulate(self, track_id: int, branch: str, identity: str, confidence: float) -> None:
        track_votes = self._votes.setdefault(track_id, _TrackVotes())

        for key in track_votes.weighted_scores:
            track_votes.weighted_scores[key] *= self.decay
        track_votes.total_weight *= self.decay

        track_votes.weighted_scores[identity] += confidence
        track_votes.total_weight += confidence

    def get_winner(self, track_id: int) -> tuple[str, float]:
        """Devuelve (identidad_ganadora, confianza_normalizada_promedio)."""
        track_votes = self._votes.get(track_id)
        if track_votes is None or track_votes.total_weight == 0:
            return "Desconocido", 0.0

        winner_identity = max(track_votes.weighted_scores, key=track_votes.weighted_scores.get)
        winner_score = track_votes.weighted_scores[winner_identity]
        normalized_confidence = winner_score / (track_votes.total_weight + 0.05)

        return winner_identity, float(normalized_confidence)

    def reset_track(self, track_id: int) -> None:
        self._votes.pop(track_id, None)