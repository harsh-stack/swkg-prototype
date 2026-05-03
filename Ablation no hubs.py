"""
Ablation Study 1: No Hub Compression
=====================================
HUB_THRESHOLD set to 999 so hubs never form.
Tests: what does small-world topology alone contribute without hub promotion?

Run: python ablation_no_hubs.py
"""

import os, time, csv
from datetime import datetime
from collections import defaultdict
import networkx as nx
import anthropic

MODEL          = "claude-haiku-4-5-20251001"
MAX_TOKENS     = 300
N_AGENTS       = 10
N_TURNS        = 100
EPOCH_SIZE     = 5
HUB_THRESHOLD  = 999          # ← KEY CHANGE: hubs never form
SW_K           = 4
SW_P           = 0.3
RESULTS_DIR    = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

_api_key = os.environ.get("ANTHROPIC_API_KEY")
if not _api_key:
    raise RuntimeError("Set environment variable ANTHROPIC_API_KEY before running.")
client = anthropic.Anthropic(api_key=_api_key)

TASK = """You are analyzing global trade patterns. Each turn, identify one 
key insight about trade network resilience, hub countries, or cascade effects. 
Be concise (2-3 sentences max). Reference previous findings if relevant."""

AGENT_ROLES = [
    "Trade Network Analyst: focus on network topology and centrality",
    "Risk Assessment Agent: focus on cascade failures and resilience",
    "Geospatial Agent: focus on regional clustering and geographic patterns"
]

class TokenLedger:
    def __init__(self, label):
        self.label = label
        self.records = []
        self.total_in = 0
        self.total_out = 0

    def log(self, agent_id, turn, prompt_tokens, completion_tokens, source="llm"):
        self.total_in += prompt_tokens
        self.total_out += completion_tokens
        self.records.append({
            "label": self.label, "agent": agent_id, "turn": turn,
            "source": source, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total": prompt_tokens + completion_tokens,
            "cumulative": self.total_in + self.total_out
        })

    @property
    def total(self):
        return self.total_in + self.total_out

class KnowledgeGraphNoHubs:
    def __init__(self):
        self.G = nx.watts_strogatz_graph(50, SW_K, SW_P)
        self.nodes = {}
        self.hubs = set()   # will always be empty
        self.n_id = 0

    def write(self, content, node_type, agent_id, token_cost, epoch):
        nid = f"n{self.n_id}"
        self.n_id += 1
        self.nodes[nid] = {
            "content": content, "type": node_type,
            "token_cost": token_cost, "token_value": 0,
            "epoch": epoch, "produced_by": agent_id, "reads": 0
        }
        self.G.add_node(nid)
        recent = list(self.nodes.keys())[-min(SW_K, len(self.nodes)):]
        for r in recent:
            if r != nid:
                self.G.add_edge(nid, r)
        return nid

    def read(self, query_type, limit=3):
        candidates = [(nid, node) for nid, node in self.nodes.items()
                      if node["type"] == query_type]
        candidates.sort(key=lambda x: x[1]["token_value"], reverse=True)
        selected = candidates[:limit]
        for nid, node in selected:
            node["reads"] += 1
            node["token_value"] += node["token_cost"]
            # HUB_THRESHOLD=999 so this never triggers
            if node["reads"] >= HUB_THRESHOLD:
                self.hubs.add(nid)
        return [(nid, node["content"]) for nid, node in selected]

    def context_tokens_for(self, query_type):
        nodes = self.read(query_type)
        total = sum(len(c.split()) * 1.3 for _, c in nodes)
        return max(int(total), 50)

    def promote_hubs(self):
        pass  # never promotes

def call_llm(system, messages, ledger, agent_id, turn):
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS,
            system=system, messages=messages
        )
        ledger.log(agent_id, turn, resp.usage.input_tokens, resp.usage.output_tokens)
        return resp.content[0].text
    except Exception as e:
        print(f"  [API error] {e}")
        mock = f"Agent {agent_id} response for turn {turn}: Analysis complete."
        ledger.log(agent_id, turn, len(system.split()) + len(str(messages)), 20, "mock")
        return mock

def run_ablation_no_hubs(n_agents, n_turns):
    print(f"\n{'='*60}")
    print("ABLATION: SW-KG WITHOUT HUB COMPRESSION (HUB_THRESHOLD=999)")
    print(f"{'='*60}")

    ledger = TokenLedger("no_hubs")
    kg = KnowledgeGraphNoHubs()

    for turn in range(n_turns):
        epoch = turn // EPOCH_SIZE
        if turn > 0 and turn % EPOCH_SIZE == 0:
            kg.promote_hubs()
            print(f"  [Epoch {epoch}] Hubs: {len(kg.hubs)} (always 0), nodes={len(kg.nodes)}")

        for agent_id in range(n_agents):
            role = AGENT_ROLES[agent_id % len(AGENT_ROLES)]
            relevant_nodes = kg.read("insight", limit=3)
            context_parts = [c for _, c in relevant_nodes]
            context_str = "\n".join(context_parts) if context_parts else "No prior context yet."

            graph_query_tokens = kg.context_tokens_for("insight")
            ledger.log(agent_id, turn, graph_query_tokens, 0, "graph_query")

            system = f"{TASK}\n\nYour role: {role}"
            messages = [{
                "role": "user",
                "content": (
                    f"Turn {turn+1}: Provide your analysis.\n\n"
                    f"Relevant prior knowledge (from graph):\n{context_str}\n\n"
                    f"Note: This is a targeted excerpt, not full history."
                )
            }]

            output = call_llm(system, messages, ledger, agent_id, turn)
            output_tokens = ledger.records[-1]["completion_tokens"]
            kg.write(output, "insight", agent_id, output_tokens, epoch)

            print(f"  [NoHubs]   Agent {agent_id} Turn {turn+1}: "
                  f"{ledger.records[-1]['prompt_tokens']}+{ledger.records[-1]['completion_tokens']} tokens "
                  f"(cumulative: {ledger.total:,}) | hubs=0 nodes={len(kg.nodes)}")

        time.sleep(0.5)

    return ledger

def save_results(ledger, filename):
    by_turn = defaultdict(int)
    for r in ledger.records:
        by_turn[r["turn"]] += r["total"]

    cum = 0
    rows = []
    for turn in range(N_TURNS):
        cum += by_turn.get(turn, 0)
        rows.append({"turn": turn+1, "cumulative": cum})

    path = f"{RESULTS_DIR}/{filename}"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nABLATION 1 RESULTS (No Hubs)")
    print(f"Total tokens: {ledger.total:,}")
    print(f"CSV saved: {path}")
    return rows

if __name__ == "__main__":
    print("Ablation Study 1: No Hub Compression")
    print(f"Config: {N_AGENTS} agents, {N_TURNS} turns, HUB_THRESHOLD={HUB_THRESHOLD}")
    ledger = run_ablation_no_hubs(N_AGENTS, N_TURNS)
    save_results(ledger, "ablation_no_hubs.csv")
