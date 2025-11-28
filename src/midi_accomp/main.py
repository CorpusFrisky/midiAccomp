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


console = Console()


def create_status_display(
    session: Session,
    engine: MidiEngine,
    voice_enabled: bool = False,
    is_recording: bool = False,
    last_transcription: str = "",
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

    if voice_enabled:
        voice_status = "[red]Recording...[/]" if is_recording else "[green]Ready (hold SPACE)[/]"
        table.add_row("Voice", voice_status)
        if last_transcription:
            # Truncate long transcriptions
            display_text = last_transcription[:50] + "..." if len(last_transcription) > 50 else last_transcription
            table.add_row("Last heard", f"[dim]{display_text}[/]")

    return Panel(table, title="[bold blue]midiAccomp[/]", border_style="blue")


def print_help(voice_enabled: bool = False):
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
    console.print("  [cyan]reset[/]              - Reset tempo and volume to default")
    console.print("  [cyan]ports[/]              - List MIDI output ports")
    console.print("  [cyan]help[/]               - Show this help")
    console.print("  [cyan]quit[/]               - Exit the program")
    if voice_enabled:
        console.print("\n[bold]Voice Control:[/]")
        console.print("  Hold [cyan]SPACE[/] and speak a command, then release")
        console.print("  Examples: \"play\", \"stop\", \"start at measure 33\", \"slower\"")
    console.print()


def execute_command(
    cmd: str,
    engine: MidiEngine,
    session: Session,
) -> tuple[bool, str]:
    """Execute a command string.

    Returns:
        (should_quit, message)
    """
    cmd = cmd.strip().lower()

    if not cmd:
        return False, ""

    parts = cmd.split()
    command = parts[0]
    args = parts[1:]

    if command in ('quit', 'exit', 'q'):
        return True, "Quitting..."

    elif command == 'help':
        print_help(voice_enabled=True)
        return False, ""

    elif command == 'play' or command == 'start':
        measure = int(args[0]) if args else engine.state.current_measure
        if measure < 1:
            measure = 1
        engine.play(from_measure=measure)
        return False, f"Playing from measure {measure}"

    elif command == 'stop':
        engine.stop()
        return False, "Stopped"

    elif command == 'goto':
        if not args:
            return False, "Usage: goto <measure>"
        else:
            measure = int(args[0])
            engine.state.current_measure = measure
            engine.state.current_tick = engine.get_measure_tick(measure)
            return False, f"Position set to measure {measure}"

    elif command == 'tempo':
        if not args:
            return False, f"Current tempo: {session.effective_tempo_bpm():.0f} BPM"
        else:
            target_bpm = float(args[0])
            session.tempo_multiplier = target_bpm / session.base_tempo_bpm
            engine.set_tempo_multiplier(session.tempo_multiplier)
            return False, f"Tempo set to {target_bpm:.0f} BPM"

    elif command == 'faster':
        session.tempo_multiplier *= 1.1
        engine.set_tempo_multiplier(session.tempo_multiplier)
        return False, f"Tempo: {session.effective_tempo_bpm():.0f} BPM"

    elif command == 'slower':
        session.tempo_multiplier *= 0.9
        engine.set_tempo_multiplier(session.tempo_multiplier)
        return False, f"Tempo: {session.effective_tempo_bpm():.0f} BPM"

    elif command == 'louder':
        session.velocity_multiplier *= 1.15
        session.velocity_multiplier = min(2.0, session.velocity_multiplier)
        engine.set_velocity_multiplier(session.velocity_multiplier)
        return False, f"Velocity: {session.velocity_multiplier:.0%}"

    elif command == 'softer':
        session.velocity_multiplier *= 0.85
        session.velocity_multiplier = max(0.1, session.velocity_multiplier)
        engine.set_velocity_multiplier(session.velocity_multiplier)
        return False, f"Velocity: {session.velocity_multiplier:.0%}"

    elif command == 'reset':
        session.tempo_multiplier = 1.0
        session.velocity_multiplier = 1.0
        engine.set_tempo_multiplier(1.0)
        engine.set_velocity_multiplier(1.0)
        return False, "Tempo and velocity reset"

    elif command == 'ports':
        ports = list_output_ports()
        console.print("[bold]Available MIDI ports:[/]")
        for p in ports:
            console.print(f"  - {p}")
        return False, ""

    elif command == 'status':
        return False, ""

    else:
        return False, f"Unknown command: {command}"


def run_cli(
    midi_file: str,
    port_name: str | None = None,
    enable_voice: bool = False,
    whisper_model: str = "base",
    ptt_key: str = "space",
):
    """Run the interactive CLI."""
    engine = MidiEngine()
    session = Session()
    voice_input = None
    last_transcription = ""
    is_recording = False

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

    # Set up voice input if enabled
    if enable_voice:
        console.print(f"Loading Whisper model ([cyan]{whisper_model}[/])... ", end="")
        try:
            from .voice_input import VoiceInput

            def on_transcription(text: str):
                nonlocal last_transcription
                last_transcription = text
                console.print(f"\n[dim]Heard:[/] {text}")

                # Try to execute as command
                should_quit, message = execute_command(text, engine, session)
                if message:
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

    console.print(create_status_display(session, engine, enable_voice, False, last_transcription))
    print_help(enable_voice)

    try:
        while True:
            try:
                cmd = console.input("[bold]>[/] ").strip()
            except EOFError:
                break

            if not cmd:
                is_rec = voice_input.is_recording() if voice_input else False
                console.print(create_status_display(session, engine, enable_voice, is_rec, last_transcription))
                continue

            should_quit, message = execute_command(cmd, engine, session)
            if message:
                if "Unknown" in message or "Usage" in message:
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
    )


if __name__ == "__main__":
    sys.exit(main())
