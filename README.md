# SW-KG Prototype

A research prototype for testing whether a **Small-World Knowledge Graph (SW-KG)** can reduce token usage in multi-agent LLM workflows compared with a baseline that keeps full conversation history.

## Overview

This repository compares two approaches:

- **Baseline**: each agent receives the full conversation history every turn.
- **SW-KG**: each agent retrieves only the most relevant prior insights from a graph-backed memory structure.

The goal is to measure whether graph-structured retrieval can lower prompt token cost while preserving useful context.

## Features

- Baseline vs SW-KG token-efficiency comparison
- Small-world knowledge graph with:
  - Watts-Strogatz topology
  - hub promotion
  - token-value-based retrieval
- Multiple ablation experiments:
  - no hubs
  - no token economy
  - no topology
- CSV output for token comparisons
- Optional plots and analysis outputs

## Project Structure

- `swkg_prototype.py` — main experiment runner
- `graph.py` — standalone knowledge graph implementation
- `Ablation-no-hubs.py` — tests SW-KG without hub promotion
- `Ablation-no-economy.py` — tests SW-KG without token-value retrieval
- `AblationNoTopology.py` — tests retrieval without graph topology
- `AblationCompare.py` — compares ablation results
- `analyze_results.py` — analyzes CSV outputs and generates plots

## Requirements

Install the Python dependencies:

```bash
pip install anthropic networkx pandas numpy matplotlib
```

## Setup

Before running the scripts, set your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

On Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY="your_api_key_here"
```

## Usage

Run the main experiment:

```bash
python swkg_prototype.py
```

Run the graph module example:

```bash
python graph.py
```

Run the ablation studies:

```bash
python Ablation-no-hubs.py
python Ablation-no-economy.py
python AblationNoTopology.py
```

Run the results analysis:

```bash
python analyze_results.py
```

## Outputs

The main script writes results to the `results/` directory, including:

- `token_comparison.csv`
- `summary.txt`
- `token_curve.png`

The ablation scripts also save CSV files and comparison plots.

## How It Works

The prototype simulates multiple agents discussing global trade patterns across many turns.

### Baseline
Each agent sends its entire conversation history back into the model on every turn. This is simple, but token costs grow quickly.

### SW-KG
Each agent queries the knowledge graph for a small set of relevant prior insights instead of sending full history. The graph prioritizes nodes using:

- betweenness centrality
- degree centrality
- token value
- clustering penalty

Frequently read nodes can become hubs, which makes future retrieval cheaper and more useful.

## Research Goal

This project explores whether small-world graph structure and selective retrieval can improve efficiency in long-running multi-agent LLM systems.

## Security Note

Do **not** commit API keys into the repository. Use environment variables instead of hard-coding secrets in the source code.

## License

Add a license here if you plan to share or publish the project publicly.
