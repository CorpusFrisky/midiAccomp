"""Session state management for the MIDI accompanist."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PositionHistory:
    """Record of a playback position for relative navigation."""
    measure: int
    tick: int
    timestamp: float  # When this position was recorded


@dataclass
class Session:
    """Manages the current state of a practice session."""

    midi_file_path: str | None = None
    current_measure: int = 1
    current_tick: int = 0
    base_tempo_bpm: float = 120.0
    tempo_multiplier: float = 1.0
    velocity_multiplier: float = 1.0
    position_history: list[PositionHistory] = field(default_factory=list)
    max_history: int = 100

    def record_position(self, measure: int, tick: int, timestamp: float) -> None:
        """Record the current position for 'go back' commands."""
        self.position_history.append(PositionHistory(measure, tick, timestamp))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        self.current_measure = measure
        self.current_tick = tick

    def get_position_measures_ago(self, measures_ago: int) -> int:
        """Get the measure number from N measures ago."""
        target = self.current_measure - measures_ago
        return max(1, target)

    def effective_tempo_bpm(self) -> float:
        """Get the current effective tempo in BPM."""
        return self.base_tempo_bpm * self.tempo_multiplier

    def reset(self) -> None:
        """Reset session state for a new file."""
        self.current_measure = 1
        self.current_tick = 0
        self.tempo_multiplier = 1.0
        self.velocity_multiplier = 1.0
        self.position_history = []

    def to_dict(self) -> dict:
        """Serialize session state to a dictionary."""
        return {
            "current_measure": self.current_measure,
            "tempo_multiplier": self.tempo_multiplier,
            "velocity_multiplier": self.velocity_multiplier,
        }

    @classmethod
    def from_dict(cls, data: dict, midi_file_path: str | None = None) -> "Session":
        """Create a session from a dictionary."""
        session = cls(midi_file_path=midi_file_path)
        session.current_measure = data.get("current_measure", 1)
        session.tempo_multiplier = data.get("tempo_multiplier", 1.0)
        session.velocity_multiplier = data.get("velocity_multiplier", 1.0)
        return session
