"""Command-line interface for the LangGraph demo application."""
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

from .workflow import LangGraphWorkflow

app = typer.Typer(help="LangGraph Demo CLI")
console = Console()


def display_welcome():
    """Display welcome message."""
    console.print(
        "[bold blue]Welcome to LangGraph Demo![/bold blue]"
        "\nType 'exit' or 'quit' to end the session.\n"
    )


def display_response(response: str):
    """Display the AI's response."""
    console.print("\n[bold green]AI:[/bold green]")
    console.print(Markdown(response))
    console.print("\n" + "-" * 50 + "\n")


@app.command()
def chat():
    """Start an interactive chat session with the AI."""
    try:
        workflow = LangGraphWorkflow()
        display_welcome()

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ("exit", "quit"):
                    console.print("\n[bold blue]Goodbye![/bold blue]")
                    break

                if not user_input:
                    continue

                # Show a spinner while processing
                with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                    response = workflow.process_message(user_input)
                
                display_response(response)

            except KeyboardInterrupt:
                console.print("\n\n[bold yellow]Use 'exit' or 'quit' to end the session.[/bold yellow]\n")
            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")
                break

    except ImportError as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        console.print("\nPlease make sure you have installed all required dependencies.")
        console.print("Run: [bold]uv pip install -e '.[dev]'[/bold]")
        sys.exit(1)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Default entrypoint: start chat if no subcommand is provided."""
    # If no subcommand was provided, run chat by default.
    if ctx.invoked_subcommand is None:
        chat()


if __name__ == "__main__":
    app()
