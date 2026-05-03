import os, time, csv, math
from datetime import datetime
from collections import defaultdict
import networkx as nx
import anthropic

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 300
N_AGENTS = 3
N_TURNS = 15
EPOCH_SIZE = 5
HUB_THRESHOLD = 3
SW_K = 4
SW_P = 0.3
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

_api_key = os.environ.get("ANTHROPIC_API_KEY")
if not _api_key:
    raise RuntimeError("Set environment variable ANTHROPIC_API_KEY before running.")
client = anthropic.Anthropic(api_key=_api_key)

print("Testing API key...")
try:
    test = client.messages.create(model=MODEL, max_tokens=10, messages=[{"role":"user","content":"Hi"}])
    print(f"KEY WORKS. Starting experiment.")
except Exception as e:
    print(f"KEY FAILED: {e}")
    exit(1)
