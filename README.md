# LangGraph Demo Application

A simple yet powerful demo application showcasing LangGraph workflows for building AI agents.

## Features

- Interactive chat interface with rich text formatting
- Configurable LLM settings via environment variables
- Well-structured, modular codebase
- Unit tests for core functionality
- Example scripts for programmatic usage

## Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) - Python package installer
- OpenAI API key

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd langgraph-demo
   ```

2. Install dependencies using uv:
   ```bash
   uv pip install -e '.[dev]'
   ```

3. Copy the example environment file and add your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

## Usage

### Interactive Chat

Start an interactive chat session:

```bash
python -m langgraph_demo.cli chat
```

### Run the Demo Script

Run the example script that demonstrates the workflow:

```bash
python examples/chat_demo.py
```

### Run Tests

Run the test suite:

```bash
uv run pytest tests/
```

## Configuration

Edit the `.env` file to configure the application:

```env
# API Keys
OPENAI_API_KEY=your_api_key_here

# Model Configuration
MODEL_NAME=gpt-3.5-turbo
TEMPERATURE=0.7
MAX_TOKENS=1000
```

## Project Structure

```
langgraph_demo/
├── __init__.py           # Package initialization
├── config.py            # Configuration settings
├── workflow.py          # LangGraph workflow definition
├── cli.py               # Command-line interface
├── tests/               # Test files
│   └── test_workflow.py
└── examples/            # Example scripts
    └── chat_demo.py
```

## License

MIT
