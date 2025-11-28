"""Session storage for saving and loading session snapshots."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .midi_engine import MidiEngine, GradualChange, TransitionType
from .session import Session


SCHEMA_VERSION = 1


@dataclass
class SaveInfo:
    """Information about a saved session."""
    name: str
    path: Path
    created_at: datetime
    current_measure: int
    tempo_multiplier: float
    velocity_multiplier: float
    has_gradual_changes: bool


class SessionStorage:
    """Handles saving and loading session snapshots."""

    def __init__(self, midi_file_path: str):
        """Initialize storage for a MIDI file.

        Args:
            midi_file_path: Path to the MIDI file
        """
        self.midi_path = Path(midi_file_path).resolve()
        self._storage_dir: Path | None = None

    def get_storage_dir(self) -> Path:
        """Get the storage directory for this MIDI file, creating if needed."""
        if self._storage_dir is None:
            # Create .midiAccomp/filename/ next to the MIDI file
            midi_dir = self.midi_path.parent
            midi_name = self.midi_path.stem
            self._storage_dir = midi_dir / ".midiAccomp" / midi_name

        return self._storage_dir

    def _ensure_storage_dir(self) -> Path:
        """Ensure storage directory exists and return it."""
        storage_dir = self.get_storage_dir()
        storage_dir.mkdir(parents=True, exist_ok=True)
        return storage_dir

    def compute_midi_hash(self) -> str:
        """Compute SHA256 hash of the MIDI file."""
        sha256 = hashlib.sha256()
        with open(self.midi_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"

    def _generate_timestamp_name(self) -> str:
        """Generate a timestamp-based save name."""
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a save name for use as a filename."""
        # Replace spaces with underscores, remove special chars
        sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return sanitized.strip("_") or "unnamed"

    def list_saves(self) -> list[SaveInfo]:
        """List all saved sessions for this MIDI file."""
        storage_dir = self.get_storage_dir()
        if not storage_dir.exists():
            return []

        saves = []
        for path in sorted(storage_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(path) as f:
                    data = json.load(f)

                created_at = datetime.fromisoformat(data.get("created_at", ""))
                state = data.get("state", {})
                gradual = data.get("gradual_changes", {})

                saves.append(SaveInfo(
                    name=path.stem,
                    path=path,
                    created_at=created_at,
                    current_measure=state.get("current_measure", 1),
                    tempo_multiplier=state.get("tempo_multiplier", 1.0),
                    velocity_multiplier=state.get("velocity_multiplier", 1.0),
                    has_gradual_changes=bool(gradual.get("tempo") or gradual.get("velocity")),
                ))
            except (json.JSONDecodeError, KeyError, ValueError):
                # Skip invalid files
                continue

        return saves

    def save(
        self,
        session: Session,
        engine: MidiEngine,
        name: str | None = None,
    ) -> str:
        """Save current session state.

        Args:
            session: Current session state
            engine: MIDI engine with playback state
            name: Optional custom name (auto-timestamp if not provided)

        Returns:
            The name of the saved session
        """
        storage_dir = self._ensure_storage_dir()

        # Generate name if not provided
        if name:
            save_name = self._sanitize_name(name)
        else:
            save_name = self._generate_timestamp_name()

        # Build save data
        data = {
            "version": SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(),
            "midi_file": {
                "name": self.midi_path.name,
                "hash": self.compute_midi_hash(),
                "measure_count": engine.get_measure_count(),
            },
            "state": {
                "current_measure": engine.state.current_measure,
                "tempo_multiplier": engine.state.base_tempo_multiplier,
                "velocity_multiplier": engine.state.base_velocity_multiplier,
            },
            "gradual_changes": engine.get_gradual_changes_dict(),
        }

        # Write file
        save_path = storage_dir / f"{save_name}.json"
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        return save_name

    def load(self, name: str) -> dict[str, Any]:
        """Load a saved session.

        Args:
            name: Name of the save to load

        Returns:
            Dictionary with loaded data and hash_match status

        Raises:
            FileNotFoundError: If save doesn't exist
            ValueError: If save is invalid
        """
        storage_dir = self.get_storage_dir()
        save_path = storage_dir / f"{name}.json"

        if not save_path.exists():
            raise FileNotFoundError(f"Save '{name}' not found")

        with open(save_path) as f:
            data = json.load(f)

        # Check version
        version = data.get("version", 0)
        if version > SCHEMA_VERSION:
            raise ValueError(f"Save version {version} is newer than supported ({SCHEMA_VERSION})")

        # Check MIDI file hash
        saved_hash = data.get("midi_file", {}).get("hash", "")
        current_hash = self.compute_midi_hash()
        hash_match = saved_hash == current_hash

        return {
            "data": data,
            "hash_match": hash_match,
            "saved_hash": saved_hash,
            "current_hash": current_hash,
        }

    def delete(self, name: str) -> bool:
        """Delete a saved session.

        Args:
            name: Name of the save to delete

        Returns:
            True if deleted, False if not found
        """
        storage_dir = self.get_storage_dir()
        save_path = storage_dir / f"{name}.json"

        if save_path.exists():
            save_path.unlink()
            return True
        return False

    def get_most_recent_save(self) -> str | None:
        """Get the name of the most recent save, or None if no saves exist."""
        saves = self.list_saves()
        return saves[0].name if saves else None


def apply_loaded_session(
    data: dict[str, Any],
    session: Session,
    engine: MidiEngine,
) -> None:
    """Apply loaded session data to session and engine.

    Args:
        data: The loaded session data (from SessionStorage.load()["data"])
        session: Session to update
        engine: MidiEngine to update
    """
    state = data.get("state", {})

    # Update session
    session.tempo_multiplier = state.get("tempo_multiplier", 1.0)
    session.velocity_multiplier = state.get("velocity_multiplier", 1.0)

    # Update engine state
    engine.state.current_measure = state.get("current_measure", 1)
    engine.state.current_tick = engine.get_measure_tick(engine.state.current_measure)
    engine.set_tempo_multiplier(session.tempo_multiplier)
    engine.set_velocity_multiplier(session.velocity_multiplier)

    # Clear existing gradual changes and apply saved ones
    engine.clear_gradual_changes()

    gradual = data.get("gradual_changes", {})

    # Apply tempo changes
    for tc in gradual.get("tempo", []):
        change_type = tc.get("type", "")
        start = tc.get("start", 1)
        end = tc.get("end", start + 4)

        if change_type == "ritardando":
            engine.add_ritardando(start, end)
        elif change_type == "accelerando":
            engine.add_accelerando(start, end)

    # Apply velocity changes
    for vc in gradual.get("velocity", []):
        change_type = vc.get("type", "")
        start = vc.get("start", 1)
        end = vc.get("end", start + 4)

        if change_type == "crescendo":
            engine.add_crescendo(start, end)
        elif change_type == "diminuendo":
            engine.add_diminuendo(start, end)
