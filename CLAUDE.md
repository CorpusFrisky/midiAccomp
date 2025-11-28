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

# Run with voice control
midi-accomp <path-to-midi-file> -v

# Run with voice + specific Whisper model
midi-accomp <path-to-midi-file> -v --whisper-model small

# List available MIDI ports
midi-accomp --list-ports

# Run tests
pytest
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Voice Input    │────▶│  Command Parser  │────▶│  MIDI Engine    │
│  (Whisper STT)  │     │  (Ollama NLU)    │     │  (Playback)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Virtual MIDI   │
                                                 │  Output Port    │
                                                 └─────────────────┘
```

### Core Modules

- **midi_engine.py** - MIDI file parsing, measure tracking, playback with tempo/velocity control, gradual changes (ritardando, accelerando, crescendo, diminuendo)
- **voice_input.py** - Push-to-talk audio recording using `sounddevice`, Whisper transcription via `faster-whisper`
- **command_parser.py** - Ollama-based NLU with fast pattern matching fallback for simple commands
- **session.py** - State management (position history for relative references like "4 measures ago")
- **session_storage.py** - Save/load session snapshots to disk (JSON files stored next to MIDI files)
- **main.py** - CLI entry point, Rich terminal UI, command execution

### Key Data Structures

- `MeasureInfo` - Tick position, time signature, tempo for each measure
- `PlaybackState` - Current position, tempo/velocity multipliers
- `GradualChange` - Represents ritardando/accelerando/crescendo/diminuendo with start/end measures
- `ParsedCommand` - Structured command output from NLU

## Key Dependencies

- `mido` + `python-rtmidi` - MIDI I/O and virtual port creation
- `faster-whisper` - Local speech-to-text (runs on CPU)
- `ollama` - Local LLM for natural language command parsing
- `sounddevice` - Cross-platform microphone input
- `pynput` - Keyboard listener for push-to-talk
- `rich` - Terminal UI with colors and tables

## Prerequisites

1. **Ollama** with a model:
   ```bash
   brew install ollama
   ollama serve  # Run in background
   ollama pull llama3.2
   ```

2. **Virtual MIDI** setup:
   - macOS: Enable IAC Driver in Audio MIDI Setup
   - Windows: Install loopMIDI
   - Linux: Use ALSA virtual MIDI

3. **DAW or synth** to receive MIDI (GarageBand, Logic, Pianoteq, etc.)

## Supported Voice Commands

| Command | Example Phrases |
|---------|----------------|
| Play | "play", "start", "go", "begin" |
| Stop | "stop", "pause", "hold" |
| Navigate | "measure 33", "go back 4 measures", "from the top" |
| Tempo | "faster", "slower", "tempo 120" |
| Volume | "louder", "softer", "forte", "piano" |
| Ritardando | "ritardando into measure 24", "slow down heading into measure 16" |
| Accelerando | "accelerando to measure 20", "speed up into measure 32" |
| Crescendo | "crescendo over the next 4 bars", "build into measure 16" |
| Diminuendo | "diminuendo to measure 8", "fade over the next 2 measures" |
| Reset | "reset", "clear changes" |
| Save | "save", "save as slow practice" |
| Load | "load", "load slow practice", "load last save" |
| List Saves | "list saves", "show saves" |

## Session Storage

Sessions are saved as JSON files in a `.midiAccomp/` folder next to the MIDI file:

```
/path/to/music/
├── song.mid
└── .midiAccomp/
    └── song/
        ├── slow_practice.json
        └── 2024-11-27_14-30.json
```

Saved sessions include:
- Current measure position
- Tempo and velocity multipliers
- Gradual changes (ritardando, accelerando, crescendo, diminuendo)
- MIDI file hash for integrity checking (warns if file changed)
