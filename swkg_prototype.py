"""
SW-KG Minimal Prototype — Token Efficiency Validator
=====================================================
Tests SW-KG vs Baseline (full history) on the same task.
Tracks every token. Produces a comparison CSV.

Setup:
    pip install anthropic networkx pandas numpy

Run:
    ANTHROPIC_API_KEY=your_key 
    python swkg_prototype.py

Output:
    results/token_comparison.csv
    results/summary.txt
"""

import os, json, time, csv, math, random
from datetime import datetime
from collections import defaultdict
import networkx as nx
import anthropic

MOCK_MODE = False  # ensure variable always exists
# ── Config ────────────────────────────────────────────────────────────────────
MODEL          = "claude-haiku-4-5-20251001"   # cheapest model — use for testing
MAX_TOKENS     = 300
N_AGENTS       = 10
N_TURNS        = 100                            # increase to 50+ to see SW-KG win
EPOCH_SIZE     = 10                             # turns per epoch
HUB_THRESHOLD  = 3                             # reads before hub promotion
SW_K           = 4                             # Watts-Strogatz neighbors
SW_P           = 0.3                           # rewiring probability
RESULTS_DIR    = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

_api_key = os.environ.get("ANTHROPIC_API_KEY")
if not _api_key:
    raise RuntimeError("Set environment variable ANTHROPIC_API_KEY before running.")
client = anthropic.Anthropic(api_key=_api_key)

# ── Token Counter ─────────────────────────────────────────────────────────────
class TokenLedger:
    def __init__(self, label):
        self.label      = label
        self.records    = []
        self.total_in   = 0
        self.total_out  = 0

    def log(self, agent_id, turn, prompt_tokens, completion_tokens, source="llm"):
        self.total_in  += prompt_tokens
        self.total_out += completion_tokens
        self.records.append({
            "label": self.label,
            "agent": agent_id,
            "turn": turn,
            "source": source,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total": prompt_tokens + completion_tokens,
            "cumulative": self.total_in + self.total_out,
        })

    @property
    def total(self):
        return self.total_in + self.total_out

# ── Knowledge Graph ───────────────────────────────────────────────────────────
# Composite SW score weights
SW_ALPHA = 0.40   # betweenness centrality  (cross-cluster connectors)
SW_BETA  = 0.30   # degree centrality       (hub-ness)
SW_GAMMA = 0.20   # token_value             (existing economy)
SW_DELTA = 0.10   # clustering coefficient  (penalise local-only nodes)

class KnowledgeGraph:
    def __init__(self):
        # smaller initial WS graph to reduce overhead but keep topology
        self.G        = nx.watts_strogatz_graph(30, SW_K, SW_P)
        self.nodes    = {}   # node_id → {content, token_cost, token_value, type, epoch, produced_by, reads}
        self.hubs     = set()
        self.n_id     = 0
        self.epoch    = 0
        self._sw_scores = {}  # node_id → composite SW score (cached at epoch boundary)

    def write(self, content: str, node_type: str, agent_id: str,
              token_cost: int, epoch: int) -> str:
        nid = f"n{self.n_id}"
        self.n_id += 1
        self.nodes[nid] = {
            "content": content,
            "type": node_type,
            "token_cost": token_cost,
            "token_value": 0,
            "epoch": epoch,
            "produced_by": agent_id,
            "reads": 0,
        }
        # Add node to graph
        self.G.add_node(nid)
        # Connect to recent nodes (simulate small-world growth)
        recent = list(self.nodes.keys())[-min(SW_K, len(self.nodes)):]
        for r in recent:
            if r != nid:
                self.G.add_edge(nid, r)
        return nid

    def read(self, query_type: str, limit: int = 3) -> list:
        """
        Retrieve relevant nodes using Small-World composite score.

        Scoring (after first epoch when SW scores are available):
            score = α·betweenness + β·degree + γ·token_value_norm - δ·clustering
        Before the first epoch: falls back to token_value (cold-start).
        Hubs are always included if capacity allows.
        """
        candidates = []
        for nid, node in self.nodes.items():
            if node["type"] == query_type or nid in self.hubs:
                candidates.append((nid, node))

        if self._sw_scores:
            # Use composite SW score — the graph topology is now actually used
            candidates.sort(
                key=lambda x: (
                    x[0] in self.hubs,          # hubs always bubble to top tier
                    self._sw_scores.get(x[0], 0.0),
                ),
                reverse=True,
            )
        else:
            # Cold-start fallback: plain token_value sort (no SW scores yet)
            candidates.sort(
                key=lambda x: (x[0] in self.hubs, x[1]["token_value"]),
                reverse=True,
            )

        selected = candidates[:limit]
        # Update token_value on read
        for nid, node in selected:
            node["reads"] += 1
            node["token_value"] += node["token_cost"]
            if node["reads"] >= HUB_THRESHOLD and nid not in self.hubs:
                self.hubs.add(nid)
        return [(nid, node["content"]) for nid, node in selected]

    def promote_hubs(self):
        """Called at epoch boundary — also recomputes SW scores."""
        for nid, node in self.nodes.items():
            if node["reads"] >= HUB_THRESHOLD:
                self.hubs.add(nid)
        self._recompute_sw_scores()

    def _recompute_sw_scores(self):
        """
        Compute composite Small-World score for every knowledge node.
        Uses NetworkX graph metrics (cached — only called at epoch boundaries).

        Score = α·betweenness + β·degree + γ·token_value_norm - δ·clustering
        All components normalised to [0, 1] across the current node set.
        """
        knowledge_nids = list(self.nodes.keys())
        if len(knowledge_nids) < 2:
            # Not enough nodes to compute meaningful centrality
            self._sw_scores = {nid: 0.0 for nid in knowledge_nids}
            return

        # Build subgraph containing only the actual knowledge nodes
        subgraph = self.G.subgraph([n for n in self.G.nodes if n in self.nodes])

        # NetworkX centrality (approximated betweenness for speed)
        k_sample = min(10, len(subgraph))
        try:
            betweenness = nx.betweenness_centrality(subgraph, normalized=True, k=k_sample)
        except Exception:
            betweenness = {n: 0.0 for n in subgraph.nodes}

        degree = nx.degree_centrality(subgraph)
        try:
            clustering = nx.clustering(subgraph)
        except Exception:
            clustering = {n: 0.0 for n in subgraph.nodes}

        # token_value — normalise across all knowledge nodes
        tv_vals = [self.nodes[nid]["token_value"] for nid in knowledge_nids]
        tv_max  = max(tv_vals) if max(tv_vals) > 0 else 1.0

        def _norm(d, nid):
            """Safely get a normalised value from a centrality dict."""
            return d.get(nid, 0.0)

        for nid in knowledge_nids:
            tv_norm = self.nodes[nid]["token_value"] / tv_max
            score = (
                SW_ALPHA * _norm(betweenness, nid)
                + SW_BETA  * _norm(degree,      nid)
                + SW_GAMMA * tv_norm
                - SW_DELTA * _norm(clustering,  nid)
            )
            self._sw_scores[nid] = score

    def context_tokens_for(self, query_type: str, cached_nodes: list = None) -> int:
        """Estimate tokens needed to load relevant context.
        Accepts pre-fetched nodes to avoid a second read() call."""
        nodes = cached_nodes if cached_nodes is not None else self.read(query_type)
        total = sum(len(c.split()) * 1.3 for _, c in nodes)  # rough token estimate
        return max(int(total), 75)

# ── LLM Call ─────────────────────────────────────────────────────────────────
def call_llm(system: str, messages: list, ledger: TokenLedger,
             agent_id: int, turn: int) -> str:
    if MOCK_MODE:
        # Mock mode: no real API call, approximate token usage
        mock = f"Agent {agent_id} response for turn {turn+1}: Analysis complete (mock)."
        est_prompt = len(system.split()) + sum(len(m["content"].split()) for m in messages)
        est_completion = 40
        ledger.log(agent_id, turn, est_prompt, est_completion, "mock")
        return mock

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
        )
        ledger.log(
            agent_id,
            turn,
            resp.usage.input_tokens,
            resp.usage.output_tokens,
            "llm",
        )
        return resp.content[0].text
    except Exception as e:
        print(f"  [API error] {e} — falling back to mock response.")
        mock = f"Agent {agent_id} response for turn {turn+1}: Analysis complete (fallback)."
        est_prompt = len(system.split()) + sum(len(m["content"].split()) for m in messages)
        est_completion = 40
        ledger.log(agent_id, turn, est_prompt, est_completion, "mock_error")
        return mock

# ── Task Definition ───────────────────────────────────────────────────────────
TASK = """You are analyzing global trade patterns. Each turn, identify one 
key insight about trade network resilience, hub countries, or cascade effects. 
Be concise (2-3 sentences max). Reference previous findings if relevant."""

AGENT_ROLES = [
    "Trade Network Analyst: focus on network topology and centrality",
    "Risk Assessment Agent: focus on cascade failures and resilience",
    "Geospatial Agent: focus on regional clustering and geographic patterns",
]

# ── Baseline: Full History Agent ──────────────────────────────────────────────
def run_baseline(n_agents: int, n_turns: int) -> TokenLedger:
    print(f"\n{'='*60}")
    print("BASELINE: Full conversation history (no KG)")
    print(f"{'='*60}")

    ledger = TokenLedger("baseline")
    histories = {i: [] for i in range(n_agents)}

    for turn in range(n_turns):
        for agent_id in range(n_agents):
            role   = AGENT_ROLES[agent_id % len(AGENT_ROLES)]
            system = f"{TASK}\n\nYour role: {role}\nTurn: {turn+1}/{n_turns}"

            # Full history in context — this is the expensive part
            messages = histories[agent_id].copy()
            messages.append({
                "role": "user",
                "content": f"Turn {turn+1}: Provide your analysis.",
            })

            output = call_llm(system, messages, ledger, agent_id, turn)

            # Append to full history
            histories[agent_id].append({
                "role": "user",
                "content": f"Turn {turn+1}: Provide your analysis.",
            })
            histories[agent_id].append({
                "role": "assistant",
                "content": output,
            })

            # Soft cap on history length to avoid Python/memory blowup
            if len(histories[agent_id]) > 500:
                histories[agent_id] = histories[agent_id][-500:]

            print(
                f"  [Baseline] Agent {agent_id} Turn {turn+1}: "
                f"{ledger.records[-1]['prompt_tokens']}+{ledger.records[-1]['completion_tokens']} tokens "
                f"(cumulative: {ledger.total:,})"
            )

        # Small delay to avoid rate limits
        time.sleep(0.1)

    return ledger

# ── SW-KG: Knowledge Graph Agent ──────────────────────────────────────────────
def run_swkg(n_agents: int, n_turns: int) -> TokenLedger:
    print(f"\n{'='*60}")
    print("SW-KG: Small-World Knowledge Graph")
    print(f"{'='*60}")

    ledger = TokenLedger("swkg")
    kg     = KnowledgeGraph()

    for turn in range(n_turns):
        epoch = turn // EPOCH_SIZE

        # Epoch boundary — promote hubs
        if turn > 0 and turn % EPOCH_SIZE == 0:
            kg.promote_hubs()
            print(
                f"  [Epoch {epoch}] Hubs promoted: {len(kg.hubs)} hubs, "
                f"{len(kg.nodes)} total nodes"
            )

        for agent_id in range(n_agents):
            role = AGENT_ROLES[agent_id % len(AGENT_ROLES)]

            # Query graph for relevant context (not full history)
            relevant_nodes = kg.read("insight", limit=3)
            context_parts  = [c for _, c in relevant_nodes]
            context_str    = "\n".join(context_parts) if context_parts else "No prior context yet."

            # Estimate graph query tokens — pass cached nodes to avoid a second read()
            graph_query_tokens = kg.context_tokens_for("insight", cached_nodes=relevant_nodes)
            ledger.log(agent_id, turn, graph_query_tokens, 0, "graph_query")

            system = f"{TASK}\n\nYour role: {role}"
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Turn {turn+1}: Provide your analysis.\n\n"
                        f"Relevant prior knowledge (from graph):\n{context_str}\n\n"
                        f"Note: This is a targeted excerpt, not full history."
                    ),
                }
            ]

            output = call_llm(system, messages, ledger, agent_id, turn)

            # Write output to graph
            output_tokens = ledger.records[-1]["completion_tokens"]
            kg.write(
                content    = output,
                node_type  = "insight",
                agent_id   = agent_id,
                token_cost = output_tokens,
                epoch      = epoch,
            )

            print(
                f"  [SW-KG]    Agent {agent_id} Turn {turn+1}: "
                f"{ledger.records[-1]['prompt_tokens']}+{ledger.records[-1]['completion_tokens']} tokens "
                f"(cumulative: {ledger.total:,}) | hubs={len(kg.hubs)} nodes={len(kg.nodes)}"
            )

        time.sleep(0.1)

    return ledger

# ── Analysis ──────────────────────────────────────────────────────────────────
def analyze(baseline: TokenLedger, swkg: TokenLedger):
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    # Per-turn cumulative comparison
    baseline_by_turn = defaultdict(int)
    swkg_by_turn     = defaultdict(int)

    for r in baseline.records:
        baseline_by_turn[r["turn"]] += r["total"]
    for r in swkg.records:
        swkg_by_turn[r["turn"]] += r["total"]

    # Cumulative
    baseline_cum, swkg_cum = 0, 0
    crossover_turn = None
    rows = []

    for turn in range(N_TURNS):
        baseline_cum += baseline_by_turn.get(turn, 0)
        swkg_cum     += swkg_by_turn.get(turn, 0)
        ratio         = baseline_cum / swkg_cum if swkg_cum > 0 else 1.0

        if crossover_turn is None and swkg_cum < baseline_cum:
            crossover_turn = turn + 1  # human-readable (1-indexed)

        rows.append({
            "turn": turn + 1,
            "baseline_cumulative": baseline_cum,
            "swkg_cumulative": swkg_cum,
            "swkg_vs_baseline_ratio": round(ratio, 2),
            "swkg_cheaper": swkg_cum < baseline_cum,
        })

    # Save CSV
    csv_path = f"{RESULTS_DIR}/token_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    reduction = (1 - swkg.total / baseline.total) * 100 if baseline.total > 0 else 0
    final_ratio = baseline.total / swkg.total if swkg.total > 0 else 1.0
    summary = f"""
SW-KG VALIDATION SUMMARY
=========================
Date:              {datetime.now().strftime('%Y-%m-%d %H:%M')}
Model:             {MODEL}
Agents:            {N_AGENTS}
Turns:             {N_TURNS}
Epoch size:        {EPOCH_SIZE}
Hub threshold:     {HUB_THRESHOLD} reads

BASELINE
  Total tokens:    {baseline.total:,}
  Prompt tokens:   {baseline.total_in:,}
  Completion:      {baseline.total_out:,}

SW-KG
  Total tokens:    {swkg.total:,}
  Prompt tokens:   {swkg.total_in:,}
  Completion:      {swkg.total_out:,}

COMPARISON
  Token reduction: {reduction:.1f}%
  Crossover turn:  {crossover_turn if crossover_turn else 'Not yet (increase N_TURNS)'}
  Final ratio:     {final_ratio:.2f}x

INTERPRETATION
  - If reduction > 0%: SW-KG is more efficient overall
  - If crossover_turn is None: increase N_TURNS (cold start not resolved)
  - If crossover_turn < 20: strong result, hub formation is fast
  - If crossover_turn 20-40: expected for this task type
  - Token reduction < 30%: normal for short tasks (SW-KG advantages are asymptotic)
  - Token reduction > 50%: strong result, publish-worthy at this scale

NEXT STEPS
  1. Increase N_TURNS to 75, 100 to see scaling behavior
  2. Increase N_AGENTS to 5, 10 to test multi-agent scaling
  3. Plot baseline_cumulative vs swkg_cumulative from CSV
  4. Fit O(T^2) to baseline, O(T*log^2(N)) to SW-KG
  5. Report R^2 of fit — this validates the complexity claim

CSV saved to: {csv_path}
"""
    print(summary)

    txt_path = f"{RESULTS_DIR}/summary.txt"
    with open(txt_path, "w") as f:
        f.write(summary)

    print(f"Results saved to {RESULTS_DIR}/")
    return rows

# ── Plot (optional, if matplotlib available) ──────────────────────────────────
def try_plot(rows):
    try:
        import matplotlib.pyplot as plt
        turns     = [r["turn"] for r in rows]
        baseline  = [r["baseline_cumulative"] for r in rows]
        swkg      = [r["swkg_cumulative"] for r in rows]

        plt.figure(figsize=(10, 6))
        plt.plot(turns, baseline, color="#f97316", linewidth=2.5, label="Baseline O(A·T²)")
        plt.plot(turns, swkg,     color="#38bdf8", linewidth=2.5, label="SW-KG O(A·T·log²N)")
        plt.xlabel("Turn Number")
        plt.ylabel("Cumulative Tokens")
        plt.title(f"SW-KG vs Baseline — {N_AGENTS} agents, {N_TURNS} turns")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{RESULTS_DIR}/token_curve.png"
        plt.savefig(path, dpi=150)
        print(f"Plot saved to {path}")
    except ImportError:
        print("matplotlib not installed — skipping plot. pip install matplotlib")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("SW-KG Token Efficiency Validator")
    print(f"Config: {N_AGENTS} agents, {N_TURNS} turns, model={MODEL}")
    est_cost = N_AGENTS * N_TURNS * 0.0003
    print(f"Estimated cost (very rough): ~${est_cost:.3f} USD\n")

    if MOCK_MODE:
        print("Running in MOCK MODE — results are structurally correct but not from real LLM calls.\n")

    # Run both systems on identical task
    baseline_ledger = run_baseline(N_AGENTS, N_TURNS)
    swkg_ledger     = run_swkg(N_AGENTS, N_TURNS)

    # Compare
    rows = analyze(baseline_ledger, swkg_ledger)
    try_plot(rows)