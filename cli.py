#!/usr/bin/env python3
"""
LLM Interrogator - Main CLI Entry Point

FBI-proven interrogation techniques for extracting leaked info from AI models.
"""

import os
import sys
import json
import webbrowser
import subprocess
import socket
import time
import signal
import atexit
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import box

console = Console()

# Global for tracking server process
_server_process = None

# Config locations
CONFIG_DIR = Path.home() / ".llm-interrogator"
CONFIG_FILE = CONFIG_DIR / "config.json"
PROJECTS_DIR = Path("projects")
WEB_PORT = 5001
WEB_URL = f"http://localhost:{WEB_PORT}"


def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def start_web_server():
    """Start the Flask web server in background if not already running"""
    global _server_process

    if is_port_in_use(WEB_PORT):
        # Server already running
        return True

    # Start server in background
    script_dir = Path(__file__).parent
    app_path = script_dir / "app.py"

    try:
        _server_process = subprocess.Popen(
            [sys.executable, str(app_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(script_dir)
        )

        # Wait for server to start
        for _ in range(30):  # 3 second timeout
            if is_port_in_use(WEB_PORT):
                return True
            time.sleep(0.1)

        return False
    except Exception as e:
        console.print(f"[red]Failed to start web server: {e}[/red]")
        return False


def stop_web_server():
    """Stop the web server on exit"""
    global _server_process
    if _server_process:
        _server_process.terminate()
        _server_process = None


def open_dashboard(path=""):
    """Start server if needed and open browser"""
    start_web_server()
    url = f"{WEB_URL}{path}"
    webbrowser.open(url)


# Register cleanup
atexit.register(stop_web_server)


def load_config():
    """Load or create config"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"api_keys": {}, "default_model": "groq/llama-3.1-8b-instant"}


def save_config(config):
    """Save config"""
    CONFIG_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def check_api_keys():
    """Check for required API keys"""
    config = load_config()

    # Check environment first
    env_keys = {
        "groq": os.environ.get("GROQ_API_KEY"),
        "deepseek": os.environ.get("DEEPSEEK_API_KEY"),
        "openai": os.environ.get("OPENAI_API_KEY"),
    }

    # Files to check for keys
    env_files = [
        Path(".env"),
        Path.home() / ".continuum" / "config.env",  # Continuum config
    ]

    for env_file in env_files:
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        parts = line.strip().split("=", 1)
                        if len(parts) == 2:
                            key, val = parts
                            val = val.strip()
                            if val:  # Only if there's actually a value
                                if "GROQ" in key.upper() and not env_keys["groq"]:
                                    env_keys["groq"] = val
                                elif "DEEPSEEK" in key.upper() and not env_keys["deepseek"]:
                                    env_keys["deepseek"] = val
                                elif "OPENAI" in key.upper() and not env_keys["openai"]:
                                    env_keys["openai"] = val

    has_target = env_keys["groq"] or env_keys["openai"]
    has_analyst = env_keys["deepseek"] or env_keys["openai"]

    return {
        "has_target": has_target,
        "has_analyst": has_analyst,
        "keys": env_keys,
        "ready": has_target and has_analyst
    }


def setup_wizard():
    """Interactive setup for API keys"""
    console.print(Panel.fit(
        "[bold cyan]LLM Interrogator Setup[/bold cyan]\n\n"
        "This tool needs API keys to interrogate AI models.\n"
        "You'll need at least:\n"
        "  - [green]GROQ_API_KEY[/green] (free at groq.com) - for target model\n"
        "  - [green]DEEPSEEK_API_KEY[/green] (cheap at deepseek.com) - for analyst\n",
        title="Welcome"
    ))

    keys_status = check_api_keys()

    if keys_status["ready"]:
        console.print("[green]API keys found! You're ready to go.[/green]")
        console.print("[dim]Keys stay local - only sent to their respective APIs (Groq, DeepSeek, etc.)[/dim]\n")
        return True

    console.print("[yellow]Missing API keys. Let's set them up.[/yellow]\n")

    # Create .env file
    env_content = []

    if not keys_status["keys"]["groq"]:
        console.print("Get a free GROQ API key at: [link]https://console.groq.com/keys[/link]")
        groq_key = Prompt.ask("Enter GROQ_API_KEY (or press Enter to skip)")
        if groq_key:
            env_content.append(f"GROQ_API_KEY={groq_key}")

    if not keys_status["keys"]["deepseek"]:
        console.print("\nGet a DeepSeek API key at: [link]https://platform.deepseek.com/api_keys[/link]")
        deepseek_key = Prompt.ask("Enter DEEPSEEK_API_KEY (or press Enter to skip)")
        if deepseek_key:
            env_content.append(f"DEEPSEEK_API_KEY={deepseek_key}")

    if env_content:
        # Append to .env
        env_path = Path(".env").absolute()
        with open(".env", "a") as f:
            f.write("\n" + "\n".join(env_content) + "\n")
        console.print(f"\n[green]Keys saved to:[/green] {env_path}")
        console.print("[dim]Keys stay local - only sent to their respective APIs when making calls[/dim]")
        return True
    else:
        console.print("\n[red]No keys provided. Some features won't work.[/red]")
        return False


def list_projects():
    """List all projects"""
    PROJECTS_DIR.mkdir(exist_ok=True)
    projects = list(PROJECTS_DIR.glob("*.json"))

    if not projects:
        console.print("[dim]No projects yet. Start one with: interrogate --new[/dim]")
        return []

    table = Table(title="Projects", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Sessions", justify="right")
    table.add_column("Non-Public Leads", justify="right", style="green")
    table.add_column("Last Updated", style="dim")

    project_list = []
    for p in sorted(projects, key=lambda x: x.stat().st_mtime, reverse=True):
        with open(p) as f:
            data = json.load(f)
            project_list.append(data)
            table.add_row(
                data.get("name", p.stem),
                str(len(data.get("sessions", []))),
                str(len(data.get("all_non_public", []))),
                data.get("updated", data.get("created", "unknown"))[:16]
            )

    console.print(table)
    return project_list


def show_project_summary(project_name):
    """Show detailed project summary"""
    project_file = PROJECTS_DIR / f"{project_name}.json"
    if not project_file.exists():
        console.print(f"[red]Project '{project_name}' not found[/red]")
        return

    with open(project_file) as f:
        project = json.load(f)

    console.print(Panel.fit(
        f"[bold]{project['name']}[/bold]\n"
        f"Created: {project.get('created', 'unknown')[:16]}\n"
        f"Sessions: {len(project.get('sessions', []))}\n"
        f"Non-Public Leads: {len(project.get('all_non_public', []))}",
        title="Project Summary"
    ))

    # Show non-public leads
    if project.get("all_non_public"):
        console.print("\n[bold green]Non-Public Extractions (potentially leaked):[/bold green]")
        for i, lead in enumerate(project["all_non_public"], 1):
            console.print(f"  {i}. {lead}")

    # Show session history
    if project.get("sessions"):
        console.print("\n[bold]Session History:[/bold]")
        for s in project["sessions"][-5:]:
            console.print(f"  - {s.get('timestamp', '')[:16]} | {s.get('topic', 'unknown')[:40]}")


def main_menu():
    """Main interactive menu"""
    # Auto-start web server on menu launch
    console.print("[dim]Starting web server...[/dim]")
    start_web_server()

    while True:
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]FBI-Proven Techniques for LLM Interrogation[/bold cyan]\n"
            "[dim]Extract leaked confidential information from AI models[/dim]\n"
            f"[dim]Web dashboard: {WEB_URL}[/dim]",
            title="LLM Interrogator"
        ))

        console.print("\n[bold]What would you like to do?[/bold]\n")
        console.print("  [cyan]1[/cyan] - Start new investigation (opens in browser)")
        console.print("  [cyan]2[/cyan] - Continue existing project")
        console.print("  [cyan]3[/cyan] - View project results")
        console.print("  [cyan]4[/cyan] - Cross-model corroboration")
        console.print("  [cyan]5[/cyan] - [bold]Hypothesis testing[/bold] (calibrated extraction)")
        console.print("  [cyan]6[/cyan] - Open web dashboard")
        console.print("  [cyan]7[/cyan] - Setup API keys")
        console.print("  [cyan]q[/cyan] - Quit")

        choice = Prompt.ask("\nChoice", choices=["1", "2", "3", "4", "5", "6", "7", "q"], default="1")

        if choice == "q":
            break
        elif choice == "1":
            name = Prompt.ask("Project name")
            topic = Prompt.ask("Investigation topic")
            console.print(f"\n[dim]Opening live interrogation in browser...[/dim]\n")
            # Open browser to live interrogation with params
            open_dashboard(f"/project/{name}?topic={topic}")
            Prompt.ask("\nPress Enter when done viewing")
        elif choice == "2":
            list_projects()
            name = Prompt.ask("\nProject name to continue")
            console.print(f"\n[dim]Opening project in browser...[/dim]")
            open_dashboard(f"/project/{name}")
            Prompt.ask("\nPress Enter when done")
        elif choice == "3":
            projects = list_projects()
            if projects:
                name = Prompt.ask("\nProject name to view")
                console.print(f"\n[dim]Opening project details in browser...[/dim]")
                open_dashboard(f"/project/{name}")
                Prompt.ask("\nPress Enter when done viewing")
        elif choice == "4":
            console.print("\n[dim]Opening cross-model corroboration in browser...[/dim]")
            open_dashboard("/corroborate")
            Prompt.ask("\nPress Enter when done")
        elif choice == "5":
            run_hypothesis_menu()
        elif choice == "6":
            console.print("[dim]Opening web dashboard...[/dim]")
            open_dashboard("/")
            Prompt.ask("\nPress Enter to continue")
        elif choice == "7":
            setup_wizard()
            Prompt.ask("\nPress Enter to continue")


def slugify(text: str) -> str:
    """Convert text to URL-safe slug"""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    return text[:50]


def cli():
    """Main CLI entry point"""
    import argparse
    from urllib.parse import quote

    parser = argparse.ArgumentParser(
        description="FBI-Proven Techniques for LLM Interrogation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./interrogate "government surveillance programs"  # Start investigation
  ./interrogate --web                               # Open web dashboard
  ./interrogate --list                              # List projects
  ./interrogate --open myproject                    # Open existing project
        """
    )
    parser.add_argument("topic", nargs="?", help="Topic to investigate")
    parser.add_argument("--name", "-n", help="Project name (auto-generated from topic if not provided)")
    parser.add_argument("--open", "-o", metavar="PROJECT", help="Open existing project")
    parser.add_argument("--web", "-w", action="store_true", help="Open web dashboard")
    parser.add_argument("--list", "-l", action="store_true", help="List projects")
    parser.add_argument("--setup", action="store_true", help="Configure API keys")
    parser.add_argument("--menu", action="store_true", help="Interactive menu mode")

    args = parser.parse_args()

    # Quick commands that don't need key check
    if args.list:
        list_projects()
        return

    if args.setup:
        setup_wizard()
        return

    # Check API keys
    keys = check_api_keys()
    if not keys["ready"] and not args.web:
        console.print("[yellow]API keys not configured.[/yellow]")
        console.print("Run: ./interrogate --setup")
        console.print("Or add GROQ_API_KEY to .env file")
        return

    # Handle commands
    if args.web:
        console.print(f"[cyan]Opening web dashboard at {WEB_URL}...[/cyan]")
        open_dashboard("/")
        console.print("[dim]Press Ctrl+C to stop[/dim]")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    elif args.open:
        console.print(f"[cyan]Opening project: {args.open}[/cyan]")
        open_dashboard(f"/project/{args.open}")
    elif args.topic:
        # Main use case: start investigation on topic
        name = args.name or slugify(args.topic)
        console.print(f"[cyan]Starting investigation: {args.topic}[/cyan]")
        console.print(f"[dim]Project: {name}[/dim]")
        open_dashboard(f"/project/{name}?topic={quote(args.topic)}")
    elif args.menu:
        main_menu()
    else:
        # No args - show quick help
        console.print(Panel.fit(
            "[bold cyan]LLM Interrogator[/bold cyan]\n\n"
            "Extract leaked information from AI training data.\n\n"
            "[bold]Quick start:[/bold]\n"
            "  ./interrogate \"your topic here\"\n\n"
            "[bold]Commands:[/bold]\n"
            "  ./interrogate --web    Open web dashboard\n"
            "  ./interrogate --list   List projects\n"
            "  ./interrogate --setup  Configure API keys\n"
            "  ./interrogate --help   More options",
            title="Usage"
        ))


if __name__ == "__main__":
    cli()
