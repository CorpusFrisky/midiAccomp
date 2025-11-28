"""Voice input handling: push-to-talk, audio recording, and transcription."""

import queue
import threading
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd


@dataclass
class AudioConfig:
    """Configuration for audio recording."""
    sample_rate: int = 16000  # Whisper expects 16kHz
    channels: int = 1
    dtype: str = "float32"
    blocksize: int = 1024


class AudioRecorder:
    """Records audio while push-to-talk is active."""

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._is_recording = False
        self._stream: sd.InputStream | None = None
        self._recorded_audio: list[np.ndarray] = []

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")
        if self._is_recording:
            self._audio_queue.put(indata.copy())

    def start_recording(self) -> None:
        """Start recording audio."""
        self._recorded_audio = []
        self._is_recording = True

        # Clear any old data from queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        if self._stream is None:
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=self.config.blocksize,
                callback=self._audio_callback,
            )
            self._stream.start()

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the recorded audio."""
        self._is_recording = False

        # Collect all recorded audio from queue
        while not self._audio_queue.empty():
            try:
                self._recorded_audio.append(self._audio_queue.get_nowait())
            except queue.Empty:
                break

        if self._recorded_audio:
            audio = np.concatenate(self._recorded_audio, axis=0)
            return audio.flatten()
        return np.array([], dtype=np.float32)

    def close(self) -> None:
        """Close the audio stream."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class Transcriber:
    """Transcribes audio using faster-whisper."""

    def __init__(self, model_size: str = "base"):
        """Initialize the transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2)
                       Smaller = faster but less accurate
                       Recommended: "base" for good balance
        """
        self.model_size = model_size
        self._model = None

    def _ensure_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            from faster_whisper import WhisperModel

            # Use CPU with int8 quantization for efficiency
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio samples as float32 numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text
        """
        self._ensure_model()

        if len(audio) == 0:
            return ""

        # faster-whisper expects float32 audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Transcribe
        segments, info = self._model.transcribe(
            audio,
            beam_size=5,
            language="en",
            vad_filter=True,  # Filter out silence
        )

        # Combine all segments
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()


class PushToTalk:
    """Manages push-to-talk functionality with keyboard."""

    def __init__(
        self,
        key: str = "space",
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
    ):
        """Initialize push-to-talk.

        Args:
            key: The key to use for push-to-talk (e.g., "space", "ctrl")
            on_start: Callback when key is pressed
            on_stop: Callback when key is released
        """
        self.key = key
        self.on_start = on_start
        self.on_stop = on_stop
        self._listener = None
        self._is_pressed = False

    def _get_key_from_string(self, key_str: str):
        """Convert string key name to pynput Key."""
        from pynput import keyboard

        key_map = {
            "space": keyboard.Key.space,
            "ctrl": keyboard.Key.ctrl,
            "ctrl_l": keyboard.Key.ctrl_l,
            "ctrl_r": keyboard.Key.ctrl_r,
            "shift": keyboard.Key.shift,
            "shift_l": keyboard.Key.shift_l,
            "shift_r": keyboard.Key.shift_r,
            "alt": keyboard.Key.alt,
            "alt_l": keyboard.Key.alt_l,
            "alt_r": keyboard.Key.alt_r,
            "cmd": keyboard.Key.cmd,
            "tab": keyboard.Key.tab,
        }
        return key_map.get(key_str.lower(), keyboard.Key.space)

    def _on_press(self, key):
        """Handle key press."""
        from pynput import keyboard

        target_key = self._get_key_from_string(self.key)

        if key == target_key and not self._is_pressed:
            self._is_pressed = True
            if self.on_start:
                self.on_start()

    def _on_release(self, key):
        """Handle key release."""
        from pynput import keyboard

        target_key = self._get_key_from_string(self.key)

        if key == target_key and self._is_pressed:
            self._is_pressed = False
            if self.on_stop:
                self.on_stop()

    def start(self) -> None:
        """Start listening for key presses."""
        from pynput import keyboard

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop(self) -> None:
        """Stop listening for key presses."""
        if self._listener:
            self._listener.stop()
            self._listener = None

    @property
    def is_pressed(self) -> bool:
        """Check if the push-to-talk key is currently pressed."""
        return self._is_pressed


class VoiceInput:
    """High-level voice input manager combining recording and transcription."""

    def __init__(
        self,
        ptt_key: str = "space",
        whisper_model: str = "base",
        on_transcription: Callable[[str], None] | None = None,
    ):
        """Initialize voice input.

        Args:
            ptt_key: Key to use for push-to-talk
            whisper_model: Whisper model size to use
            on_transcription: Callback when transcription is complete
        """
        self.on_transcription = on_transcription
        self._recorder = AudioRecorder()
        self._transcriber = Transcriber(whisper_model)
        self._ptt = PushToTalk(
            key=ptt_key,
            on_start=self._start_recording,
            on_stop=self._stop_recording,
        )
        self._is_active = False
        self._transcription_thread: threading.Thread | None = None

    def _start_recording(self) -> None:
        """Called when PTT key is pressed."""
        if self._is_active:
            self._recorder.start_recording()

    def _stop_recording(self) -> None:
        """Called when PTT key is released."""
        if not self._is_active:
            return

        audio = self._recorder.stop_recording()

        if len(audio) > 0:
            # Transcribe in background thread to not block
            self._transcription_thread = threading.Thread(
                target=self._transcribe_async,
                args=(audio,),
                daemon=True,
            )
            self._transcription_thread.start()

    def _transcribe_async(self, audio: np.ndarray) -> None:
        """Transcribe audio in background thread."""
        text = self._transcriber.transcribe(audio)
        if text and self.on_transcription:
            self.on_transcription(text)

    def start(self) -> None:
        """Start voice input processing."""
        self._is_active = True
        self._ptt.start()

    def stop(self) -> None:
        """Stop voice input processing."""
        self._is_active = False
        self._ptt.stop()
        self._recorder.close()

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._ptt.is_pressed


def list_audio_devices() -> list[dict]:
    """List available audio input devices."""
    devices = sd.query_devices()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append({
                'index': i,
                'name': dev['name'],
                'channels': dev['max_input_channels'],
                'sample_rate': dev['default_samplerate'],
            })
    return input_devices
