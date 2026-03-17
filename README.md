# LlamaIndex ReAct Agent Example

This is a simple example of a ReAct Agent built with LlamaIndex and Python 3.12, managed by `uv`.

## Features

- **ReAct Agent**: Uses the ReAct (Reasoning + Acting) pattern to solve problems.
- **Calculator Tools**: Basic `add` and `multiply` functions.
- **Mock Knowledge Tool**: A simple dictionary-based search tool simulating a knowledge base.
- **OpenAI Integration**: Uses OpenAI's GPT-3.5-turbo model.

## Prerequisites

- Python 3.12+
- `uv` (a fast Python package installer and resolver)
- OpenAI API Key

## Setup

1.  **Clone/Navigate to the directory**:
    ```bash
    cd llama_agent_example
    ```

2.  **Install Dependencies**:
    ```bash
    uv sync
    ```

3.  **Configure Environment**:
    Create a `.env` file from the example:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and add your OpenAI API Key:
    ```env
    OPENAI_API_KEY=sk-proj-...
    ```

## Run

To run the agent:

```bash
uv run main.py
```

## Usage

Once running, you can ask questions like:
- "What is 20 + 5 * 3?" (Wait for the agent to calculate step-by-step)
- "What is LlamaIndex?" (Wait for the agent to use the search tool)
- Type `exit` to quit.
