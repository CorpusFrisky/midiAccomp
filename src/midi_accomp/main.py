"""CLI entry point for the MIDI accompanist."""

import argparse
import sys
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .midi_engine import MidiEngine, list_output_ports
from .session import Session


console = Console()


def create_status_display(session: Session, engine: MidiEngine) -> Panel:
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

    return Panel(table, title="[bold blue]midiAccomp[/]", border_style="blue")


def print_help():
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
    console.print()


def run_cli(midi_file: str, port_name: str | None = None):
    """Run the interactive CLI."""
    engine = MidiEngine()
    session = Session()

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
        # Convert microseconds per beat to BPM
        tempo_us = engine.measures[0].tempo
        session.base_tempo_bpm = 60_000_000 / tempo_us

    console.print(create_status_display(session, engine))
    print_help()

    try:
        while True:
            try:
                cmd = console.input("[bold]>[/] ").strip().lower()
            except EOFError:
                break

            if not cmd:
                console.print(create_status_display(session, engine))
                continue

            parts = cmd.split()
            command = parts[0]
            args = parts[1:]

            if command in ('quit', 'exit', 'q'):
                break

            elif command == 'help':
                print_help()

            elif command == 'play':
                measure = int(args[0]) if args else engine.state.current_measure
                if measure < 1:
                    measure = 1
                engine.play(from_measure=measure)
                console.print(f"[green]Playing from measure {measure}[/]")

            elif command == 'stop':
                engine.stop()
                console.print("[yellow]Stopped[/]")

            elif command == 'goto':
                if not args:
                    console.print("[red]Usage: goto <measure>[/]")
                else:
                    measure = int(args[0])
                    engine.state.current_measure = measure
                    engine.state.current_tick = engine.get_measure_tick(measure)
                    console.print(f"Position set to measure {measure}")

            elif command == 'tempo':
                if not args:
                    console.print(f"Current tempo: {session.effective_tempo_bpm():.0f} BPM")
                else:
                    target_bpm = float(args[0])
                    session.tempo_multiplier = target_bpm / session.base_tempo_bpm
                    engine.set_tempo_multiplier(session.tempo_multiplier)
                    console.print(f"Tempo set to {target_bpm:.0f} BPM")

            elif command == 'faster':
                session.tempo_multiplier *= 1.1
                engine.set_tempo_multiplier(session.tempo_multiplier)
                console.print(f"Tempo: {session.effective_tempo_bpm():.0f} BPM ({session.tempo_multiplier:.0%})")

            elif command == 'slower':
                session.tempo_multiplier *= 0.9
                engine.set_tempo_multiplier(session.tempo_multiplier)
                console.print(f"Tempo: {session.effective_tempo_bpm():.0f} BPM ({session.tempo_multiplier:.0%})")

            elif command == 'louder':
                session.velocity_multiplier *= 1.15
                session.velocity_multiplier = min(2.0, session.velocity_multiplier)
                engine.set_velocity_multiplier(session.velocity_multiplier)
                console.print(f"Velocity: {session.velocity_multiplier:.0%}")

            elif command == 'softer':
                session.velocity_multiplier *= 0.85
                session.velocity_multiplier = max(0.1, session.velocity_multiplier)
                engine.set_velocity_multiplier(session.velocity_multiplier)
                console.print(f"Velocity: {session.velocity_multiplier:.0%}")

            elif command == 'reset':
                session.tempo_multiplier = 1.0
                session.velocity_multiplier = 1.0
                engine.set_tempo_multiplier(1.0)
                engine.set_velocity_multiplier(1.0)
                console.print("Tempo and velocity reset to default")

            elif command == 'ports':
                ports = list_output_ports()
                console.print("[bold]Available MIDI ports:[/]")
                for p in ports:
                    console.print(f"  - {p}")

            elif command == 'status':
                console.print(create_status_display(session, engine))

            else:
                console.print(f"[red]Unknown command:[/] {command}")
                console.print("Type 'help' for available commands")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")

    finally:
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
    parser.add_argument("midi_file", help="Path to MIDI file to play")
    parser.add_argument("-p", "--port", help="MIDI output port name")
    parser.add_argument("--list-ports", action="store_true", help="List available MIDI ports and exit")

    args = parser.parse_args()

    if args.list_ports:
        ports = list_output_ports()
        print("Available MIDI output ports:")
        for p in ports:
            print(f"  - {p}")
        return 0

    return run_cli(args.midi_file, args.port)


if __name__ == "__main__":
    sys.exit(main())
