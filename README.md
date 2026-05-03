# SW-KG — Small-World Knowledge Graph

Prototype comparing **SW-KG** (structured graph memory) to a **full-history baseline** on token usage for multi-turn LLM agents.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
set ANTHROPIC_API_KEY=your_key_here
```

## Run

```bash
python swkg_prototype.py
python analyze_results.py
```

Ablation scripts (after editing paths if needed):

```bash
python "Ablation no hubs.py"
python "Ablation no economy.py"
python AblationNoTopology.py
python AblationCompare.py
```

Outputs are written under `results/` (gitignored).

## License

Add a license file if you publish publicly.
