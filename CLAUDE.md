# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

midiAccomp is a voice-controlled MIDI accompanist that allows musicians to control playback using natural language commands like "start at measure 33" or "slow down heading into that last measure."

## Build and Development Commands

```bash
# Install dependencies (use a virtual environment)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -e ".[dev]"

# Run the application
midi-accomp <path-to-midi-file>

# Run tests
pytest
```

## Architecture

The application is structured into these core modules:

- **midi_engine.py** - MIDI file parsing, playback control, tempo/velocity manipulation
- **voice_input.py** - Push-to-talk audio recording and Whisper transcription
- **command_parser.py** - Ollama-based NLU to convert speech to structured commands
- **session.py** - State management (current position, tempo, playback history)
- **main.py** - CLI entry point and event loop coordination

## Key Dependencies

- `mido` + `python-rtmidi` for MIDI I/O and virtual port creation
- `faster-whisper` for local speech-to-text
- `ollama` for local LLM command parsing
- `sounddevice` for microphone input
- `pynput` for push-to-talk key detection
- `rich` for terminal UI

## Prerequisites

1. Ollama with a model: `brew install ollama && ollama pull llama3.2`
2. Virtual MIDI setup (macOS: enable IAC Driver in Audio MIDI Setup)
3. A DAW or synth to receive MIDI output
