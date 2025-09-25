"""Example script to demonstrate using the LangGraph workflow."""
import asyncio
from rich.console import Console
from rich.markdown import Markdown

from langgraph_demo.workflow import LangGraphWorkflow

console = Console()


def display_response(response: str):
    """Display the AI's response in a formatted way."""
    console.print("\n[bold green]AI:[/bold green]")
    console.print(Markdown(response))
    console.print("\n" + "-" * 50 + "\n")


async def main():
    """Run the chat demo."""
    console.print(
        "[bold blue]LangGraph Chat Demo[/bold blue]"
        "\nType 'exit' or 'quit' to end the session.\n"
    )

    try:
        # Initialize the workflow
        workflow = LangGraphWorkflow()

        # Example conversation
        messages = [
            "Hello, who are you?",
            "What can you help me with?",
            "Tell me a joke"
        ]

        for message in messages:
            console.print(f"[bold]You:[/bold] {message}")
            
            # Process the message
            with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                response = workflow.process_message(message)
            
            display_response(response)
            
            # Small delay to simulate thinking time
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        console.print("\n\n[bold yellow]Demo interrupted by user.[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
    finally:
        console.print("\n[bold blue]Demo completed. Goodbye![/bold blue]")


if __name__ == "__main__":
    asyncio.run(main())
