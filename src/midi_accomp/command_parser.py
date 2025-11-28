"""Natural language command parsing using Ollama."""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import ollama


class CommandType(Enum):
    """Types of commands the user can give."""
    PLAY = "play"
    STOP = "stop"
    GOTO = "goto"
    TEMPO = "tempo"
    TEMPO_ADJUST = "tempo_adjust"
    VELOCITY = "velocity"
    VELOCITY_ADJUST = "velocity_adjust"
    RITARDANDO = "ritardando"
    ACCELERANDO = "accelerando"
    CRESCENDO = "crescendo"
    DIMINUENDO = "diminuendo"
    RESET = "reset"
    CLEAR_CHANGES = "clear_changes"
    SAVE = "save"
    LOAD = "load"
    LIST_SAVES = "list_saves"
    DELETE_SAVE = "delete_save"
    UNKNOWN = "unknown"


@dataclass
class PositionReference:
    """A reference to a position in the music."""
    type: str  # "measure", "relative", "beginning", "current"
    value: int | None = None  # Measure number or relative offset


@dataclass
class DynamicChange:
    """A change in dynamics (velocity)."""
    type: str  # "set", "adjust", "crescendo", "diminuendo"
    value: float | None = None  # Multiplier or target
    start: PositionReference | None = None
    end: PositionReference | None = None


@dataclass
class TempoChange:
    """A change in tempo."""
    type: str  # "set", "adjust", "ritardando", "accelerando"
    value: float | None = None  # BPM or multiplier
    start: PositionReference | None = None
    end: PositionReference | None = None


@dataclass
class SaveCommand:
    """A save or load command."""
    name: str | None = None  # Save name (None for auto-timestamp or most recent)


@dataclass
class ParsedCommand:
    """A parsed command from natural language."""
    command_type: CommandType
    position: PositionReference | None = None
    tempo_change: TempoChange | None = None
    dynamic_change: DynamicChange | None = None
    save_command: SaveCommand | None = None
    raw_text: str = ""
    confidence: float = 1.0


SYSTEM_PROMPT = """You are a command parser for a MIDI accompanist application. Musicians speak commands to control playback.

Parse the user's speech into a JSON command. Always respond with valid JSON only, no other text.

Command types:
- play: Start playback (optionally from a specific measure)
- stop: Stop playback
- goto: Move to a position without playing
- tempo: Set or adjust tempo
- velocity: Set or adjust volume/dynamics
- reset: Reset tempo and velocity to defaults
- clear_changes: Clear any gradual tempo/velocity changes
- save: Save current session state
- load: Load a saved session
- list_saves: List all saved sessions
- delete_save: Delete a saved session

Position references:
- "measure 33" -> {"type": "measure", "value": 33}
- "the beginning" / "the top" -> {"type": "beginning"}
- "4 measures ago" / "back 4 bars" -> {"type": "relative", "value": -4}
- "2 measures ahead" -> {"type": "relative", "value": 2}
- "here" / "current position" -> {"type": "current"}

Tempo changes:
- "tempo 120" -> {"type": "set", "value": 120}
- "faster" / "speed up" -> {"type": "adjust", "value": 1.1}
- "slower" / "slow down" -> {"type": "adjust", "value": 0.9}
- "much slower" -> {"type": "adjust", "value": 0.7}
- "ritardando" / "rit" / "slow down into" -> {"type": "ritardando", "start": ..., "end": ...}
- "accelerando" / "accel" / "speed up into" -> {"type": "accelerando", "start": ..., "end": ...}

Dynamic changes:
- "louder" / "more" -> {"type": "adjust", "value": 1.15}
- "softer" / "quieter" -> {"type": "adjust", "value": 0.85}
- "forte" / "strong" -> {"type": "set", "value": 1.3}
- "piano" / "soft" -> {"type": "set", "value": 0.7}
- "crescendo" / "build" -> {"type": "crescendo", "start": ..., "end": ...}
- "diminuendo" / "decrescendo" / "fade" -> {"type": "diminuendo", "start": ..., "end": ...}

For gradual changes (ritardando, accelerando, crescendo, diminuendo):
- If no start is specified, assume "current" position
- If no end is specified but a target measure is mentioned, use that as the end
- The "start" and "end" fields use the same position reference format

Examples:

Input: "start"
Output: {"command": "play"}

Input: "stop please"
Output: {"command": "stop"}

Input: "start at measure 33"
Output: {"command": "play", "position": {"type": "measure", "value": 33}}

Input: "take it from the top"
Output: {"command": "play", "position": {"type": "beginning"}}

Input: "go back 4 measures"
Output: {"command": "play", "position": {"type": "relative", "value": -4}}

Input: "let's try that again from 2 bars back"
Output: {"command": "play", "position": {"type": "relative", "value": -2}}

Input: "slower"
Output: {"command": "tempo", "tempo": {"type": "adjust", "value": 0.9}}

Input: "tempo 100"
Output: {"command": "tempo", "tempo": {"type": "set", "value": 100}}

Input: "louder please"
Output: {"command": "velocity", "velocity": {"type": "adjust", "value": 1.15}}

Input: "play it forte"
Output: {"command": "velocity", "velocity": {"type": "set", "value": 1.3}}

Input: "reset"
Output: {"command": "reset"}

Input: "ritardando into measure 24"
Output: {"command": "tempo", "tempo": {"type": "ritardando", "start": {"type": "current"}, "end": {"type": "measure", "value": 24}}}

Input: "slow down heading into measure 16"
Output: {"command": "tempo", "tempo": {"type": "ritardando", "start": {"type": "current"}, "end": {"type": "measure", "value": 16}}}

Input: "ritardando from measure 20 to measure 24"
Output: {"command": "tempo", "tempo": {"type": "ritardando", "start": {"type": "measure", "value": 20}, "end": {"type": "measure", "value": 24}}}

Input: "crescendo over the next 4 measures"
Output: {"command": "velocity", "velocity": {"type": "crescendo", "start": {"type": "current"}, "end": {"type": "relative", "value": 4}}}

Input: "build into measure 32"
Output: {"command": "velocity", "velocity": {"type": "crescendo", "start": {"type": "current"}, "end": {"type": "measure", "value": 32}}}

Input: "diminuendo to measure 8"
Output: {"command": "velocity", "velocity": {"type": "diminuendo", "start": {"type": "current"}, "end": {"type": "measure", "value": 8}}}

Input: "start at measure 20 and slow down going into measure 24"
Output: {"command": "play", "position": {"type": "measure", "value": 20}, "tempo": {"type": "ritardando", "start": {"type": "measure", "value": 20}, "end": {"type": "measure", "value": 24}}}

Input: "accelerando from here to measure 16"
Output: {"command": "tempo", "tempo": {"type": "accelerando", "start": {"type": "current"}, "end": {"type": "measure", "value": 16}}}

Input: "clear changes"
Output: {"command": "clear_changes"}

Input: "save"
Output: {"command": "save"}

Input: "save as slow practice"
Output: {"command": "save", "save_name": "slow practice"}

Input: "load slow practice"
Output: {"command": "load", "save_name": "slow practice"}

Input: "load last save"
Output: {"command": "load"}

Input: "list saves"
Output: {"command": "list_saves"}

Input: "show saves"
Output: {"command": "list_saves"}

Input: "delete slow practice"
Output: {"command": "delete_save", "save_name": "slow practice"}

Respond with only valid JSON. If you cannot understand the command, respond with {"command": "unknown"}.
"""


class CommandParser:
    """Parses natural language into structured commands using Ollama."""

    def __init__(self, model: str = "llama3.2"):
        """Initialize the parser.

        Args:
            model: Ollama model to use (e.g., "llama3.2", "mistral")
        """
        self.model = model
        self._client = ollama.Client()

    def parse(self, text: str, current_measure: int = 1) -> ParsedCommand:
        """Parse natural language text into a command.

        Args:
            text: The transcribed speech to parse
            current_measure: Current playback position for relative references

        Returns:
            ParsedCommand with the interpreted command
        """
        # First try simple pattern matching for common commands
        simple_result = self._try_simple_parse(text, current_measure)
        if simple_result:
            return simple_result

        # Fall back to LLM for complex commands
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                options={
                    "temperature": 0.1,  # Low temperature for consistent parsing
                },
            )

            response_text = response["message"]["content"].strip()

            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group()

            return self._parse_json_response(response_text, text, current_measure)

        except Exception as e:
            # If Ollama fails, return unknown command
            return ParsedCommand(
                command_type=CommandType.UNKNOWN,
                raw_text=text,
                confidence=0.0,
            )

    def _try_simple_parse(self, text: str, current_measure: int = 1) -> ParsedCommand | None:
        """Try to parse simple commands without using the LLM."""
        text = text.lower().strip()

        # Direct command matches
        if text in ("play", "start", "go", "begin"):
            return ParsedCommand(command_type=CommandType.PLAY, raw_text=text)

        if text in ("stop", "pause", "hold", "wait"):
            return ParsedCommand(command_type=CommandType.STOP, raw_text=text)

        if text in ("faster", "speed up", "quicker"):
            return ParsedCommand(
                command_type=CommandType.TEMPO_ADJUST,
                tempo_change=TempoChange(type="adjust", value=1.1),
                raw_text=text,
            )

        if text in ("slower", "slow down"):
            return ParsedCommand(
                command_type=CommandType.TEMPO_ADJUST,
                tempo_change=TempoChange(type="adjust", value=0.9),
                raw_text=text,
            )

        if text in ("louder", "more volume", "louder please"):
            return ParsedCommand(
                command_type=CommandType.VELOCITY_ADJUST,
                dynamic_change=DynamicChange(type="adjust", value=1.15),
                raw_text=text,
            )

        if text in ("softer", "quieter", "less volume"):
            return ParsedCommand(
                command_type=CommandType.VELOCITY_ADJUST,
                dynamic_change=DynamicChange(type="adjust", value=0.85),
                raw_text=text,
            )

        if text in ("reset", "reset tempo", "reset volume", "default"):
            return ParsedCommand(command_type=CommandType.RESET, raw_text=text)

        if text in ("clear changes", "clear", "clear all"):
            return ParsedCommand(command_type=CommandType.CLEAR_CHANGES, raw_text=text)

        # "play at measure X" or "start at measure X"
        match = re.match(r"(?:play|start|go|begin)\s+(?:at\s+)?measure\s+(\d+)", text)
        if match:
            measure = int(match.group(1))
            return ParsedCommand(
                command_type=CommandType.PLAY,
                position=PositionReference(type="measure", value=measure),
                raw_text=text,
            )

        # "measure X" alone
        match = re.match(r"measure\s+(\d+)", text)
        if match:
            measure = int(match.group(1))
            return ParsedCommand(
                command_type=CommandType.PLAY,
                position=PositionReference(type="measure", value=measure),
                raw_text=text,
            )

        # "tempo X"
        match = re.match(r"tempo\s+(\d+)", text)
        if match:
            bpm = int(match.group(1))
            return ParsedCommand(
                command_type=CommandType.TEMPO,
                tempo_change=TempoChange(type="set", value=bpm),
                raw_text=text,
            )

        # "from the top" / "from the beginning"
        if "from the top" in text or "from the beginning" in text:
            return ParsedCommand(
                command_type=CommandType.PLAY,
                position=PositionReference(type="beginning"),
                raw_text=text,
            )

        # Simple ritardando/accelerando patterns
        # "ritardando into measure X" or "rit to measure X"
        match = re.search(r"(?:ritardando|rit|slow down).*(?:into|to|at)\s+measure\s+(\d+)", text)
        if match:
            end_measure = int(match.group(1))
            return ParsedCommand(
                command_type=CommandType.RITARDANDO,
                tempo_change=TempoChange(
                    type="ritardando",
                    start=PositionReference(type="current"),
                    end=PositionReference(type="measure", value=end_measure),
                ),
                raw_text=text,
            )

        # "accelerando into measure X"
        match = re.search(r"(?:accelerando|accel|speed up).*(?:into|to|at)\s+measure\s+(\d+)", text)
        if match:
            end_measure = int(match.group(1))
            return ParsedCommand(
                command_type=CommandType.ACCELERANDO,
                tempo_change=TempoChange(
                    type="accelerando",
                    start=PositionReference(type="current"),
                    end=PositionReference(type="measure", value=end_measure),
                ),
                raw_text=text,
            )

        # "crescendo into measure X"
        match = re.search(r"(?:crescendo|cresc|build).*(?:into|to|at)\s+measure\s+(\d+)", text)
        if match:
            end_measure = int(match.group(1))
            return ParsedCommand(
                command_type=CommandType.CRESCENDO,
                dynamic_change=DynamicChange(
                    type="crescendo",
                    start=PositionReference(type="current"),
                    end=PositionReference(type="measure", value=end_measure),
                ),
                raw_text=text,
            )

        # "diminuendo into measure X"
        match = re.search(r"(?:diminuendo|dim|decrescendo|fade).*(?:into|to|at)\s+measure\s+(\d+)", text)
        if match:
            end_measure = int(match.group(1))
            return ParsedCommand(
                command_type=CommandType.DIMINUENDO,
                dynamic_change=DynamicChange(
                    type="diminuendo",
                    start=PositionReference(type="current"),
                    end=PositionReference(type="measure", value=end_measure),
                ),
                raw_text=text,
            )

        # "X measures" patterns for gradual changes
        # "crescendo over the next X measures"
        match = re.search(r"(?:crescendo|cresc|build).*(?:next|over)\s+(\d+)\s+(?:measures|bars)", text)
        if match:
            num_measures = int(match.group(1))
            return ParsedCommand(
                command_type=CommandType.CRESCENDO,
                dynamic_change=DynamicChange(
                    type="crescendo",
                    start=PositionReference(type="current"),
                    end=PositionReference(type="relative", value=num_measures),
                ),
                raw_text=text,
            )

        # "diminuendo over the next X measures"
        match = re.search(r"(?:diminuendo|dim|decrescendo|fade).*(?:next|over)\s+(\d+)\s+(?:measures|bars)", text)
        if match:
            num_measures = int(match.group(1))
            return ParsedCommand(
                command_type=CommandType.DIMINUENDO,
                dynamic_change=DynamicChange(
                    type="diminuendo",
                    start=PositionReference(type="current"),
                    end=PositionReference(type="relative", value=num_measures),
                ),
                raw_text=text,
            )

        # Save/Load commands
        # "save" - save with auto-timestamp
        if text == "save":
            return ParsedCommand(
                command_type=CommandType.SAVE,
                save_command=SaveCommand(),
                raw_text=text,
            )

        # "save as <name>" or "save <name>"
        match = re.match(r"save\s+(?:as\s+)?(.+)", text)
        if match:
            save_name = match.group(1).strip()
            return ParsedCommand(
                command_type=CommandType.SAVE,
                save_command=SaveCommand(name=save_name),
                raw_text=text,
            )

        # "load last save" or "load" - load most recent
        if text in ("load", "load last save", "load last", "load recent"):
            return ParsedCommand(
                command_type=CommandType.LOAD,
                save_command=SaveCommand(),
                raw_text=text,
            )

        # "load <name>"
        match = re.match(r"load\s+(.+)", text)
        if match:
            save_name = match.group(1).strip()
            return ParsedCommand(
                command_type=CommandType.LOAD,
                save_command=SaveCommand(name=save_name),
                raw_text=text,
            )

        # "list saves" or "show saves"
        if text in ("list saves", "show saves", "saves", "list sessions", "show sessions"):
            return ParsedCommand(
                command_type=CommandType.LIST_SAVES,
                raw_text=text,
            )

        # "delete <name>"
        match = re.match(r"delete\s+(?:save\s+)?(.+)", text)
        if match:
            save_name = match.group(1).strip()
            return ParsedCommand(
                command_type=CommandType.DELETE_SAVE,
                save_command=SaveCommand(name=save_name),
                raw_text=text,
            )

        return None

    def _parse_json_response(self, json_str: str, original_text: str, current_measure: int = 1) -> ParsedCommand:
        """Parse the JSON response from Ollama into a ParsedCommand."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return ParsedCommand(
                command_type=CommandType.UNKNOWN,
                raw_text=original_text,
                confidence=0.0,
            )

        command_str = data.get("command", "unknown")

        # Map command string to CommandType
        command_map = {
            "play": CommandType.PLAY,
            "stop": CommandType.STOP,
            "goto": CommandType.GOTO,
            "tempo": CommandType.TEMPO,
            "velocity": CommandType.VELOCITY,
            "reset": CommandType.RESET,
            "clear_changes": CommandType.CLEAR_CHANGES,
            "save": CommandType.SAVE,
            "load": CommandType.LOAD,
            "list_saves": CommandType.LIST_SAVES,
            "delete_save": CommandType.DELETE_SAVE,
            "unknown": CommandType.UNKNOWN,
        }

        command_type = command_map.get(command_str, CommandType.UNKNOWN)

        # Parse save command
        save_command = None
        if command_type in (CommandType.SAVE, CommandType.LOAD, CommandType.DELETE_SAVE):
            save_name = data.get("save_name")
            save_command = SaveCommand(name=save_name)

        # Parse position
        position = None
        if "position" in data:
            pos_data = data["position"]
            position = PositionReference(
                type=pos_data.get("type", "measure"),
                value=pos_data.get("value"),
            )

        # Parse tempo change
        tempo_change = None
        if "tempo" in data:
            tempo_data = data["tempo"]
            tempo_type = tempo_data.get("type", "set")

            # Parse start and end positions for gradual changes
            start_pos = None
            end_pos = None
            if "start" in tempo_data:
                start_data = tempo_data["start"]
                start_pos = PositionReference(
                    type=start_data.get("type", "current"),
                    value=start_data.get("value"),
                )
            if "end" in tempo_data:
                end_data = tempo_data["end"]
                end_pos = PositionReference(
                    type=end_data.get("type", "measure"),
                    value=end_data.get("value"),
                )

            tempo_change = TempoChange(
                type=tempo_type,
                value=tempo_data.get("value"),
                start=start_pos,
                end=end_pos,
            )

            # Adjust command type based on tempo type
            if tempo_type == "adjust":
                command_type = CommandType.TEMPO_ADJUST
            elif tempo_type == "ritardando":
                command_type = CommandType.RITARDANDO
            elif tempo_type == "accelerando":
                command_type = CommandType.ACCELERANDO

        # Parse dynamic change
        dynamic_change = None
        if "velocity" in data:
            vel_data = data["velocity"]
            vel_type = vel_data.get("type", "set")

            # Parse start and end positions for gradual changes
            start_pos = None
            end_pos = None
            if "start" in vel_data:
                start_data = vel_data["start"]
                start_pos = PositionReference(
                    type=start_data.get("type", "current"),
                    value=start_data.get("value"),
                )
            if "end" in vel_data:
                end_data = vel_data["end"]
                end_pos = PositionReference(
                    type=end_data.get("type", "measure"),
                    value=end_data.get("value"),
                )

            dynamic_change = DynamicChange(
                type=vel_type,
                value=vel_data.get("value"),
                start=start_pos,
                end=end_pos,
            )

            # Adjust command type based on velocity type
            if vel_type == "adjust":
                command_type = CommandType.VELOCITY_ADJUST
            elif vel_type == "crescendo":
                command_type = CommandType.CRESCENDO
            elif vel_type == "diminuendo":
                command_type = CommandType.DIMINUENDO

        return ParsedCommand(
            command_type=command_type,
            position=position,
            tempo_change=tempo_change,
            dynamic_change=dynamic_change,
            save_command=save_command,
            raw_text=original_text,
            confidence=0.9 if command_type != CommandType.UNKNOWN else 0.0,
        )


def check_ollama_available(model: str = "llama3.2") -> tuple[bool, str]:
    """Check if Ollama is available and the model is installed.

    Returns:
        (is_available, message)
    """
    try:
        client = ollama.Client()
        models = client.list()
        model_names = [m["name"].split(":")[0] for m in models.get("models", [])]

        if model in model_names or any(model in name for name in model_names):
            return True, f"Ollama ready with {model}"
        else:
            available = ", ".join(model_names) if model_names else "none"
            return False, f"Model '{model}' not found. Available: {available}. Run: ollama pull {model}"

    except Exception as e:
        return False, f"Ollama not available: {e}. Is Ollama running? Try: ollama serve"
