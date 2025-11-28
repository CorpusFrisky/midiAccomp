"""MIDI file parsing, playback control, and real-time manipulation."""

from dataclasses import dataclass, field
from typing import Callable
import threading
import time

import mido


@dataclass
class MeasureInfo:
    """Information about a measure's location in the MIDI file."""
    number: int  # 1-indexed measure number
    tick: int  # Absolute tick position
    time_seconds: float  # Time in seconds from start
    time_signature: tuple[int, int]  # (numerator, denominator)
    tempo: int  # Microseconds per beat at this measure


@dataclass
class PlaybackState:
    """Current state of MIDI playback."""
    is_playing: bool = False
    current_tick: int = 0
    current_measure: int = 1
    tempo_multiplier: float = 1.0  # For ritardando/accelerando
    velocity_multiplier: float = 1.0  # For dynamics


class MidiEngine:
    """Handles MIDI file loading, parsing, and playback."""

    def __init__(self):
        self.midi_file: mido.MidiFile | None = None
        self.measures: list[MeasureInfo] = []
        self.ticks_per_beat: int = 480
        self.state = PlaybackState()
        self._playback_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._output_port: mido.ports.BaseOutput | None = None
        self._on_position_change: Callable[[int, int, float], None] | None = None

    def load_file(self, filepath: str) -> None:
        """Load a MIDI file and analyze its structure."""
        self.midi_file = mido.MidiFile(filepath)
        self.ticks_per_beat = self.midi_file.ticks_per_beat
        self._analyze_measures()
        self.state = PlaybackState()

    def _analyze_measures(self) -> None:
        """Analyze the MIDI file to find measure boundaries."""
        if not self.midi_file:
            return

        self.measures = []

        # Default time signature and tempo
        current_time_sig = (4, 4)
        current_tempo = 500000  # 120 BPM in microseconds per beat

        # Collect all tempo and time signature changes
        tempo_changes: list[tuple[int, int]] = [(0, current_tempo)]
        time_sig_changes: list[tuple[int, tuple[int, int]]] = [(0, current_time_sig)]

        # Merge all tracks to get meta events in order
        for msg in mido.merge_tracks(self.midi_file.tracks):
            # Note: merged messages have absolute time in msg.time
            pass

        # Actually, let's iterate through track 0 (usually has tempo/time sig)
        # and accumulate tick positions
        abs_tick = 0
        for track in self.midi_file.tracks:
            track_tick = 0
            for msg in track:
                track_tick += msg.time
                if msg.type == 'set_tempo':
                    tempo_changes.append((track_tick, msg.tempo))
                elif msg.type == 'time_signature':
                    time_sig_changes.append((track_tick, (msg.numerator, msg.denominator)))

        # Sort changes by tick
        tempo_changes.sort(key=lambda x: x[0])
        time_sig_changes.sort(key=lambda x: x[0])

        # Calculate measure boundaries
        measure_num = 1
        current_tick = 0
        current_time = 0.0
        tempo_idx = 0
        time_sig_idx = 0
        current_tempo = tempo_changes[0][1]
        current_time_sig = time_sig_changes[0][1]

        # Find the end of the file
        max_tick = 0
        for track in self.midi_file.tracks:
            track_tick = 0
            for msg in track:
                track_tick += msg.time
            max_tick = max(max_tick, track_tick)

        while current_tick <= max_tick:
            # Update tempo if needed
            while tempo_idx + 1 < len(tempo_changes) and tempo_changes[tempo_idx + 1][0] <= current_tick:
                tempo_idx += 1
                current_tempo = tempo_changes[tempo_idx][1]

            # Update time signature if needed
            while time_sig_idx + 1 < len(time_sig_changes) and time_sig_changes[time_sig_idx + 1][0] <= current_tick:
                time_sig_idx += 1
                current_time_sig = time_sig_changes[time_sig_idx][1]

            # Record this measure
            self.measures.append(MeasureInfo(
                number=measure_num,
                tick=current_tick,
                time_seconds=current_time,
                time_signature=current_time_sig,
                tempo=current_tempo,
            ))

            # Calculate ticks per measure
            numerator, denominator = current_time_sig
            # Ticks per measure = ticks_per_beat * (numerator * 4 / denominator)
            ticks_per_measure = int(self.ticks_per_beat * numerator * 4 / denominator)

            # Calculate time for this measure
            beats_per_measure = numerator
            seconds_per_beat = current_tempo / 1_000_000
            measure_duration = beats_per_measure * seconds_per_beat

            # Move to next measure
            current_tick += ticks_per_measure
            current_time += measure_duration
            measure_num += 1

        # Add a final measure marker if needed
        if self.measures and self.measures[-1].tick < max_tick:
            self.measures.append(MeasureInfo(
                number=measure_num,
                tick=max_tick,
                time_seconds=current_time,
                time_signature=current_time_sig,
                tempo=current_tempo,
            ))

    def get_measure_count(self) -> int:
        """Return the total number of measures."""
        return len(self.measures)

    def get_measure_tick(self, measure_num: int) -> int:
        """Get the tick position for a measure number (1-indexed)."""
        if measure_num < 1:
            return 0
        if measure_num > len(self.measures):
            return self.measures[-1].tick if self.measures else 0
        return self.measures[measure_num - 1].tick

    def tick_to_measure(self, tick: int) -> int:
        """Convert a tick position to a measure number."""
        for i, measure in enumerate(self.measures):
            if i + 1 < len(self.measures) and self.measures[i + 1].tick > tick:
                return measure.number
        return self.measures[-1].number if self.measures else 1

    def open_output(self, port_name: str | None = None) -> None:
        """Open a MIDI output port."""
        if port_name:
            self._output_port = mido.open_output(port_name)
        else:
            # Try to create a virtual port
            try:
                self._output_port = mido.open_output('midiAccomp', virtual=True)
            except (OSError, NotImplementedError):
                # Fall back to first available port
                ports = mido.get_output_names()
                if ports:
                    self._output_port = mido.open_output(ports[0])
                else:
                    raise RuntimeError("No MIDI output ports available")

    def close_output(self) -> None:
        """Close the MIDI output port."""
        if self._output_port:
            self._output_port.close()
            self._output_port = None

    def play(self, from_measure: int = 1) -> None:
        """Start playback from a specific measure."""
        if not self.midi_file or not self._output_port:
            return

        self.stop()

        self.state.current_measure = from_measure
        self.state.current_tick = self.get_measure_tick(from_measure)
        self.state.is_playing = True
        self._stop_event.clear()

        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

    def stop(self) -> None:
        """Stop playback."""
        self.state.is_playing = False
        self._stop_event.set()

        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)

        # Send all notes off
        if self._output_port:
            for channel in range(16):
                self._output_port.send(mido.Message('control_change', channel=channel, control=123, value=0))

    def set_tempo_multiplier(self, multiplier: float) -> None:
        """Set the tempo multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double)."""
        self.state.tempo_multiplier = max(0.1, min(4.0, multiplier))

    def set_velocity_multiplier(self, multiplier: float) -> None:
        """Set the velocity multiplier for dynamics."""
        self.state.velocity_multiplier = max(0.1, min(2.0, multiplier))

    def adjust_tempo(self, delta: float) -> None:
        """Adjust tempo by a relative amount (e.g., 0.1 for 10% faster)."""
        self.state.tempo_multiplier *= (1.0 + delta)
        self.state.tempo_multiplier = max(0.1, min(4.0, self.state.tempo_multiplier))

    def adjust_velocity(self, delta: float) -> None:
        """Adjust velocity by a relative amount."""
        self.state.velocity_multiplier *= (1.0 + delta)
        self.state.velocity_multiplier = max(0.1, min(2.0, self.state.velocity_multiplier))

    def set_position_callback(self, callback: Callable[[int, int, float], None]) -> None:
        """Set a callback for position updates: (measure, tick, time)."""
        self._on_position_change = callback

    def _playback_loop(self) -> None:
        """Main playback loop running in a separate thread."""
        if not self.midi_file or not self._output_port:
            return

        # Merge all tracks for playback
        merged = list(mido.merge_tracks(self.midi_file.tracks))

        # Find starting position
        start_tick = self.state.current_tick
        current_tick = 0
        msg_index = 0

        # Skip messages before start position
        while msg_index < len(merged) and current_tick < start_tick:
            msg = merged[msg_index]
            current_tick += msg.time
            msg_index += 1

        # Get initial tempo
        current_tempo = 500000  # Default 120 BPM
        for measure in self.measures:
            if measure.tick <= start_tick:
                current_tempo = measure.tempo

        last_time = time.perf_counter()

        while msg_index < len(merged) and not self._stop_event.is_set():
            msg = merged[msg_index]

            if msg.time > 0:
                # Calculate delay with tempo multiplier
                ticks_to_wait = msg.time
                seconds_per_tick = (current_tempo / 1_000_000) / self.ticks_per_beat
                delay = ticks_to_wait * seconds_per_tick / self.state.tempo_multiplier

                # Wait
                target_time = last_time + delay
                while time.perf_counter() < target_time:
                    if self._stop_event.is_set():
                        return
                    time.sleep(0.001)  # 1ms sleep for responsiveness

                last_time = time.perf_counter()

            current_tick += msg.time

            # Update tempo if this is a tempo message
            if msg.type == 'set_tempo':
                current_tempo = msg.tempo

            # Send the message (with velocity scaling for note_on)
            if msg.type == 'note_on' and msg.velocity > 0:
                scaled_velocity = int(msg.velocity * self.state.velocity_multiplier)
                scaled_velocity = max(1, min(127, scaled_velocity))
                out_msg = msg.copy(velocity=scaled_velocity)
                self._output_port.send(out_msg)
            elif not msg.is_meta:
                self._output_port.send(msg)

            # Update state
            self.state.current_tick = current_tick
            self.state.current_measure = self.tick_to_measure(current_tick)

            # Call position callback
            if self._on_position_change:
                elapsed = sum(m.time for m in merged[:msg_index+1])
                self._on_position_change(self.state.current_measure, current_tick, elapsed)

            msg_index += 1

        self.state.is_playing = False


def list_output_ports() -> list[str]:
    """List available MIDI output ports."""
    return mido.get_output_names()
