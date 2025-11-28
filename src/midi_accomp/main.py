"""CLI entry point for the MIDI accompanist."""

import argparse
import sys
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .midi_engine import MidiEngine, list_output_ports
from .session import Session
from .command_parser import CommandParser, CommandType, ParsedCommand, check_ollama_available


console = Console()


def create_status_display(
    session: Session,
    engine: MidiEngine,
    voice_enabled: bool = False,
    nlu_enabled: bool = False,
    is_recording: bool = False,
    last_transcription: str = "",
    last_interpreted: str = "",
) -> Panel:
    """Create a rich panel showing current status."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="bold")
    table.add_column("Value")

    status = "Playing" if engine.state.is_playing else "Stopped"
    status_style = "green" if engine.state.is_playing else "yellow"

    table.add_row("Status", f"[{status_style}]{status}[/]")
    table.add_row("File", session.midi_file_path or "None")
    table.add_row("Measure", f"{engine.state.current_measure} / {engine.get_measure_count()}")
    table.add_row("Tempo", f"{session.effective_tempo_bpm():.0f} BPM ({session.tempo_multiplier:.0%})")
    table.add_row("Velocity", f"{session.velocity_multiplier:.0%}")

    # Show active gradual changes
    active_changes = engine.get_active_changes()
    if active_changes["tempo_changes"] or active_changes["velocity_changes"]:
        changes_str = []
        for tc in active_changes["tempo_changes"]:
            changes_str.append(f"{tc['type']} m{tc['start']}-{tc['end']}")
        for vc in active_changes["velocity_changes"]:
            changes_str.append(f"{vc['type']} m{vc['start']}-{vc['end']}")
        table.add_row("Changes", f"[magenta]{', '.join(changes_str)}[/]")

    if voice_enabled:
        voice_status = "[red]Recording...[/]" if is_recording else "[green]Ready (hold SPACE)[/]"
        table.add_row("Voice", voice_status)
        if last_transcription:
            display_text = last_transcription[:50] + "..." if len(last_transcription) > 50 else last_transcription
            table.add_row("Heard", f"[dim]{display_text}[/]")
        if nlu_enabled and last_interpreted:
            table.add_row("Interpreted", f"[cyan]{last_interpreted}[/]")

    return Panel(table, title="[bold blue]midiAccomp[/]", border_style="blue")


def print_help(voice_enabled: bool = False, nlu_enabled: bool = False):
    """Print available commands."""
    console.print("\n[bold]Commands:[/]")
    console.print("  [cyan]play[/] [measure]     - Start playback (optionally from measure)")
    console.print("  [cyan]stop[/]               - Stop playback")
    console.print("  [cyan]goto[/] <measure>     - Go to measure number")
    console.print("  [cyan]tempo[/] <bpm>        - Set tempo in BPM")
    console.print("  [cyan]faster[/]             - Increase tempo 10%")
    console.print("  [cyan]slower[/]             - Decrease tempo 10%")
    console.print("  [cyan]louder[/]             - Increase volume 15%")
    console.print("  [cyan]softer[/]             - Decrease volume 15%")
    console.print("  [cyan]reset[/]              - Reset tempo, volume, and gradual changes")
    console.print("  [cyan]clear[/]              - Clear gradual tempo/velocity changes")
    console.print("  [cyan]ports[/]              - List MIDI output ports")
    console.print("  [cyan]help[/]               - Show this help")
    console.print("  [cyan]quit[/]               - Exit the program")
    if voice_enabled:
        console.print("\n[bold]Voice Control:[/]")
        console.print("  Hold [cyan]SPACE[/] and speak a command, then release")
        if nlu_enabled:
            console.print("  [bold]Natural language enabled![/] Try phrases like:")
            console.print("    - \"take it from the top\"")
            console.print("    - \"go back 4 measures\"")
            console.print("    - \"start at measure 33\"")
            console.print("    - \"ritardando into measure 24\"")
            console.print("    - \"crescendo over the next 4 bars\"")
            console.print("    - \"slow down heading into measure 16\"")
        else:
            console.print("  Examples: \"play\", \"stop\", \"start at measure 33\", \"slower\"")
    console.print()


def execute_parsed_command(
    parsed: ParsedCommand,
    engine: MidiEngine,
    session: Session,
) -> tuple[bool, str]:
    """Execute a parsed command.

    Returns:
        (should_quit, message)
    """
    cmd_type = parsed.command_type

    if cmd_type == CommandType.PLAY:
        # Determine starting measure
        measure = engine.state.current_measure
        if parsed.position:
            if parsed.position.type == "measure" and parsed.position.value:
                measure = parsed.position.value
            elif parsed.position.type == "beginning":
                measure = 1
            elif parsed.position.type == "relative" and parsed.position.value:
                measure = max(1, engine.state.current_measure + parsed.position.value)
            elif parsed.position.type == "current":
                measure = engine.state.current_measure

        if measure < 1:
            measure = 1

        engine.play(from_measure=measure)
        return False, f"Playing from measure {measure}"

    elif cmd_type == CommandType.STOP:
        engine.stop()
        return False, "Stopped"

    elif cmd_type == CommandType.GOTO:
        if parsed.position and parsed.position.value:
            measure = parsed.position.value
            engine.state.current_measure = measure
            engine.state.current_tick = engine.get_measure_tick(measure)
            return False, f"Position set to measure {measure}"
        return False, "No position specified"

    elif cmd_type == CommandType.TEMPO:
        if parsed.tempo_change and parsed.tempo_change.value:
            target_bpm = parsed.tempo_change.value
            session.tempo_multiplier = target_bpm / session.base_tempo_bpm
            engine.set_tempo_multiplier(session.tempo_multiplier)
            return False, f"Tempo set to {target_bpm:.0f} BPM"
        return False, f"Current tempo: {session.effective_tempo_bpm():.0f} BPM"

    elif cmd_type == CommandType.TEMPO_ADJUST:
        if parsed.tempo_change and parsed.tempo_change.value:
            session.tempo_multiplier *= parsed.tempo_change.value
            session.tempo_multiplier = max(0.1, min(4.0, session.tempo_multiplier))
            engine.set_tempo_multiplier(session.tempo_multiplier)
        return False, f"Tempo: {session.effective_tempo_bpm():.0f} BPM ({session.tempo_multiplier:.0%})"

    elif cmd_type == CommandType.VELOCITY:
        if parsed.dynamic_change and parsed.dynamic_change.value:
            session.velocity_multiplier = parsed.dynamic_change.value
            session.velocity_multiplier = max(0.1, min(2.0, session.velocity_multiplier))
            engine.set_velocity_multiplier(session.velocity_multiplier)
        return False, f"Velocity: {session.velocity_multiplier:.0%}"

    elif cmd_type == CommandType.VELOCITY_ADJUST:
        if parsed.dynamic_change and parsed.dynamic_change.value:
            session.velocity_multiplier *= parsed.dynamic_change.value
            session.velocity_multiplier = max(0.1, min(2.0, session.velocity_multiplier))
            engine.set_velocity_multiplier(session.velocity_multiplier)
        return False, f"Velocity: {session.velocity_multiplier:.0%}"

    elif cmd_type == CommandType.RESET:
        session.tempo_multiplier = 1.0
        session.velocity_multiplier = 1.0
        engine.set_tempo_multiplier(1.0)
        engine.set_velocity_multiplier(1.0)
        engine.clear_gradual_changes()
        return False, "Tempo and velocity reset"

    elif cmd_type == CommandType.CLEAR_CHANGES:
        engine.clear_gradual_changes()
        return False, "Cleared all gradual changes"

    elif cmd_type == CommandType.RITARDANDO:
        if parsed.tempo_change:
            start_measure, end_measure = _resolve_range(
                parsed.tempo_change.start,
                parsed.tempo_change.end,
                engine.state.current_measure,
            )
            engine.add_ritardando(start_measure, end_measure)
            return False, f"Ritardando from measure {start_measure} to {end_measure}"
        return False, "Ritardando needs a target measure"

    elif cmd_type == CommandType.ACCELERANDO:
        if parsed.tempo_change:
            start_measure, end_measure = _resolve_range(
                parsed.tempo_change.start,
                parsed.tempo_change.end,
                engine.state.current_measure,
            )
            engine.add_accelerando(start_measure, end_measure)
            return False, f"Accelerando from measure {start_measure} to {end_measure}"
        return False, "Accelerando needs a target measure"

    elif cmd_type == CommandType.CRESCENDO:
        if parsed.dynamic_change:
            start_measure, end_measure = _resolve_range(
                parsed.dynamic_change.start,
                parsed.dynamic_change.end,
                engine.state.current_measure,
            )
            engine.add_crescendo(start_measure, end_measure)
            return False, f"Crescendo from measure {start_measure} to {end_measure}"
        return False, "Crescendo needs a target measure"

    elif cmd_type == CommandType.DIMINUENDO:
        if parsed.dynamic_change:
            start_measure, end_measure = _resolve_range(
                parsed.dynamic_change.start,
                parsed.dynamic_change.end,
                engine.state.current_measure,
            )
            engine.add_diminuendo(start_measure, end_measure)
            return False, f"Diminuendo from measure {start_measure} to {end_measure}"
        return False, "Diminuendo needs a target measure"

    elif cmd_type == CommandType.UNKNOWN:
        return False, f"Didn't understand: {parsed.raw_text}"

    return False, ""


def _resolve_range(
    start: "PositionReference | None",
    end: "PositionReference | None",
    current_measure: int,
) -> tuple[int, int]:
    """Resolve start and end position references to actual measure numbers."""
    from .command_parser import PositionReference

    # Resolve start measure
    start_measure = current_measure
    if start:
        if start.type == "measure" and start.value:
            start_measure = start.value
        elif start.type == "relative" and start.value:
            start_measure = current_measure + start.value
        elif start.type == "beginning":
            start_measure = 1
        # "current" uses the default

    # Resolve end measure
    end_measure = start_measure + 4  # Default: 4 measures
    if end:
        if end.type == "measure" and end.value:
            end_measure = end.value
        elif end.type == "relative" and end.value:
            end_measure = current_measure + end.value
        elif end.type == "beginning":
            end_measure = 1

    # Ensure start < end
    if start_measure >= end_measure:
        end_measure = start_measure + 4

    return max(1, start_measure), max(1, end_measure)


def execute_text_command(
    cmd: str,
    engine: MidiEngine,
    session: Session,
    parser: CommandParser | None = None,
) -> tuple[bool, str, str]:
    """Execute a text command (typed or from voice).

    Returns:
        (should_quit, message, interpreted_as)
    """
    cmd = cmd.strip()

    if not cmd:
        return False, "", ""

    # Check for quit commands first (don't send to NLU)
    if cmd.lower() in ('quit', 'exit', 'q'):
        return True, "Quitting...", "quit"

    if cmd.lower() == 'help':
        print_help(voice_enabled=True, nlu_enabled=parser is not None)
        return False, "", "help"

    if cmd.lower() == 'ports':
        ports = list_output_ports()
        console.print("[bold]Available MIDI ports:[/]")
        for p in ports:
            console.print(f"  - {p}")
        return False, "", "ports"

    if cmd.lower() == 'status':
        return False, "", "status"

    # Use NLU parser if available
    if parser:
        parsed = parser.parse(cmd, engine.state.current_measure)
        interpreted = f"{parsed.command_type.value}"
        if parsed.position:
            if parsed.position.type == "measure":
                interpreted += f" @ measure {parsed.position.value}"
            elif parsed.position.type == "beginning":
                interpreted += " @ beginning"
            elif parsed.position.type == "relative":
                interpreted += f" {parsed.position.value:+d} measures"

        should_quit, message = execute_parsed_command(parsed, engine, session)
        return should_quit, message, interpreted

    # Fall back to simple command parsing
    parts = cmd.lower().split()
    command = parts[0]
    args = parts[1:]

    if command in ('play', 'start'):
        measure = int(args[0]) if args else engine.state.current_measure
        if measure < 1:
            measure = 1
        engine.play(from_measure=measure)
        return False, f"Playing from measure {measure}", "play"

    elif command == 'stop':
        engine.stop()
        return False, "Stopped", "stop"

    elif command == 'goto':
        if not args:
            return False, "Usage: goto <measure>", "goto"
        measure = int(args[0])
        engine.state.current_measure = measure
        engine.state.current_tick = engine.get_measure_tick(measure)
        return False, f"Position set to measure {measure}", "goto"

    elif command == 'tempo':
        if not args:
            return False, f"Current tempo: {session.effective_tempo_bpm():.0f} BPM", "tempo"
        target_bpm = float(args[0])
        session.tempo_multiplier = target_bpm / session.base_tempo_bpm
        engine.set_tempo_multiplier(session.tempo_multiplier)
        return False, f"Tempo set to {target_bpm:.0f} BPM", "tempo"

    elif command == 'faster':
        session.tempo_multiplier *= 1.1
        engine.set_tempo_multiplier(session.tempo_multiplier)
        return False, f"Tempo: {session.effective_tempo_bpm():.0f} BPM", "faster"

    elif command == 'slower':
        session.tempo_multiplier *= 0.9
        engine.set_tempo_multiplier(session.tempo_multiplier)
        return False, f"Tempo: {session.effective_tempo_bpm():.0f} BPM", "slower"

    elif command == 'louder':
        session.velocity_multiplier *= 1.15
        session.velocity_multiplier = min(2.0, session.velocity_multiplier)
        engine.set_velocity_multiplier(session.velocity_multiplier)
        return False, f"Velocity: {session.velocity_multiplier:.0%}", "louder"

    elif command == 'softer':
        session.velocity_multiplier *= 0.85
        session.velocity_multiplier = max(0.1, session.velocity_multiplier)
        engine.set_velocity_multiplier(session.velocity_multiplier)
        return False, f"Velocity: {session.velocity_multiplier:.0%}", "softer"

    elif command == 'reset':
        session.tempo_multiplier = 1.0
        session.velocity_multiplier = 1.0
        engine.set_tempo_multiplier(1.0)
        engine.set_velocity_multiplier(1.0)
        return False, "Tempo and velocity reset", "reset"

    else:
        return False, f"Unknown command: {command}", ""


def run_cli(
    midi_file: str,
    port_name: str | None = None,
    enable_voice: bool = False,
    whisper_model: str = "base",
    ptt_key: str = "space",
    ollama_model: str = "llama3.2",
    disable_nlu: bool = False,
):
    """Run the interactive CLI."""
    engine = MidiEngine()
    session = Session()
    voice_input = None
    command_parser = None
    last_transcription = ""
    last_interpreted = ""
    is_recording = False
    nlu_enabled = False

    # Load MIDI file
    console.print(f"Loading [cyan]{midi_file}[/]...")
    try:
        engine.load_file(midi_file)
        session.midi_file_path = midi_file
    except Exception as e:
        console.print(f"[red]Error loading MIDI file:[/] {e}")
        return 1

    console.print(f"Loaded {engine.get_measure_count()} measures")

    # Open MIDI output
    console.print("Opening MIDI output...")
    try:
        engine.open_output(port_name)
        port_info = port_name or "midiAccomp (virtual)"
        console.print(f"Output: [green]{port_info}[/]")
    except Exception as e:
        console.print(f"[red]Error opening MIDI output:[/] {e}")
        console.print("Available ports:", list_output_ports())
        return 1

    # Set up position callback
    def on_position(measure: int, tick: int, elapsed: float):
        session.record_position(measure, tick, elapsed)

    engine.set_position_callback(on_position)

    # Get base tempo from first measure
    if engine.measures:
        tempo_us = engine.measures[0].tempo
        session.base_tempo_bpm = 60_000_000 / tempo_us

    # Set up Ollama NLU if not disabled
    if not disable_nlu:
        console.print(f"Checking Ollama ([cyan]{ollama_model}[/])... ", end="")
        available, ollama_msg = check_ollama_available(ollama_model)
        if available:
            command_parser = CommandParser(model=ollama_model)
            nlu_enabled = True
            console.print("[green]Ready![/]")
        else:
            console.print("[yellow]Not available[/]")
            console.print(f"[dim]{ollama_msg}[/]")
            console.print("[dim]Continuing with basic command parsing...[/]")

    # Set up voice input if enabled
    if enable_voice:
        console.print(f"Loading Whisper model ([cyan]{whisper_model}[/])... ", end="")
        try:
            from .voice_input import VoiceInput

            def on_transcription(text: str):
                nonlocal last_transcription, last_interpreted
                last_transcription = text
                console.print(f"\n[dim]Heard:[/] {text}")

                # Execute command
                should_quit, message, interpreted = execute_text_command(
                    text, engine, session, command_parser
                )
                last_interpreted = interpreted

                if interpreted:
                    console.print(f"[cyan]Interpreted as:[/] {interpreted}")
                if message:
                    if "Unknown" in message or "Didn't understand" in message:
                        console.print(f"[yellow]{message}[/]")
                    else:
                        console.print(f"[green]{message}[/]")
                console.print("[bold]>[/] ", end="")

            voice_input = VoiceInput(
                ptt_key=ptt_key,
                whisper_model=whisper_model,
                on_transcription=on_transcription,
            )
            # Force model load now
            voice_input._transcriber._ensure_model()
            voice_input.start()
            console.print("[green]Ready![/]")
            console.print(f"Hold [cyan]{ptt_key.upper()}[/] to speak commands")
        except Exception as e:
            console.print(f"[red]Failed![/]")
            console.print(f"[red]Voice input error:[/] {e}")
            console.print("Continuing without voice control...")
            enable_voice = False
            voice_input = None

    console.print(create_status_display(
        session, engine, enable_voice, nlu_enabled, False, last_transcription, last_interpreted
    ))
    print_help(enable_voice, nlu_enabled)

    try:
        while True:
            try:
                cmd = console.input("[bold]>[/] ").strip()
            except EOFError:
                break

            if not cmd:
                is_rec = voice_input.is_recording() if voice_input else False
                console.print(create_status_display(
                    session, engine, enable_voice, nlu_enabled, is_rec, last_transcription, last_interpreted
                ))
                continue

            should_quit, message, interpreted = execute_text_command(
                cmd, engine, session, command_parser
            )

            if message:
                if "Unknown" in message or "Usage" in message or "Didn't understand" in message:
                    console.print(f"[red]{message}[/]")
                else:
                    console.print(f"[green]{message}[/]")

            if should_quit:
                break

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")

    finally:
        if voice_input:
            voice_input.stop()
        engine.stop()
        engine.close_output()
        console.print("[dim]Goodbye![/]")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Voice-controlled MIDI accompanist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("midi_file", nargs="?", help="Path to MIDI file to play")
    parser.add_argument("-p", "--port", help="MIDI output port name")
    parser.add_argument("--list-ports", action="store_true", help="List available MIDI ports and exit")
    parser.add_argument(
        "-v", "--voice",
        action="store_true",
        help="Enable voice control (push-to-talk with SPACE key)"
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v2"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--ptt-key",
        default="space",
        help="Push-to-talk key (default: space)"
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model for NLU (default: llama3.2)"
    )
    parser.add_argument(
        "--no-nlu",
        action="store_true",
        help="Disable natural language understanding (use simple commands only)"
    )

    args = parser.parse_args()

    if args.list_ports:
        ports = list_output_ports()
        print("Available MIDI output ports:")
        for p in ports:
            print(f"  - {p}")
        return 0

    if not args.midi_file:
        parser.error("midi_file is required unless using --list-ports")

    return run_cli(
        args.midi_file,
        port_name=args.port,
        enable_voice=args.voice,
        whisper_model=args.whisper_model,
        ptt_key=args.ptt_key,
        ollama_model=args.ollama_model,
        disable_nlu=args.no_nlu,
    )


if __name__ == "__main__":
    sys.exit(main())
