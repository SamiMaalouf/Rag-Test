import time
import json
import math
import re
import itertools
import os
from collections import defaultdict, deque

import nest_asyncio
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

nest_asyncio.apply()

# ================== CONFIG ==================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "Missing GEMINI_API_KEY. Create a .env file with GEMINI_API_KEY=... or set it in your environment."
    )

MODEL_NAME = "models/gemini-2.5-flash"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

NUM_QUESTIONS = 15
VANILLA_TOP_K = 5

# GRAG-style parameters
EGO_HOPS = 1           # start with 1; try 2 later if you want
TOP_N_EGOGRAPHS = 3    # number of ego-graphs retrieved
MAX_PRUNED_NODES = 6   # heuristic pruning budget
EDGE_OVERLAP_THRESHOLD = 1

Settings.llm = Gemini(
    model=MODEL_NAME,
    api_key=GEMINI_API_KEY
)
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

# ================== HELPERS ==================

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_keywords(text: str):
    """
    Very simple keyword extractor:
    - keep words with letters/numbers
    - lowercase
    - remove very short tokens
    """
    tokens = re.findall(r"[A-Za-z0-9\-']+", text.lower())
    stop = {
        "the","a","an","and","or","of","to","in","on","for","with","is","was","were",
        "are","be","by","that","this","what","which","who","when","where","how","from",
        "as","at","it","its","his","her","their","both","did","do","does","had","has",
        "have","held","name","known"
    }
    return {t for t in tokens if len(t) > 2 and t not in stop}

def extract_capitalized_phrases(text: str):
    """
    Lightweight proxy for named entities.
    """
    phrases = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    return {p.strip() for p in phrases if len(p.strip()) > 2}

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ================== GRAPH STRUCTURES ==================

class Node:
    def __init__(self, node_id, title, text):
        self.id = node_id
        self.title = title
        self.text = normalize_text(text)
        self.full_text = f"Title: {self.title}\nText: {self.text}"
        self.keywords = extract_keywords(self.full_text)
        self.entities = extract_capitalized_phrases(self.full_text)

class TextGraph:
    def __init__(self):
        self.nodes = {}                # node_id -> Node
        self.adj = defaultdict(set)    # node_id -> set(neighbor_ids)
        self.edges = set()             # set of tuple(sorted(i,j))

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, a, b):
        if a == b:
            return
        e = tuple(sorted((a, b)))
        if e not in self.edges:
            self.edges.add(e)
            self.adj[a].add(b)
            self.adj[b].add(a)

# ================== GRAPH BUILDING ==================

def build_text_graph_from_hotpot_item(item):
    """
    Build one graph per question context.
    Each context paragraph becomes one node.
    Edge if entity/keyword overlap exists.
    """
    graph = TextGraph()
    context = item["context"]

    # HotpotQA distractor context format is typically [titles, sentences] pairs or list-like entries
    # We'll handle robustly.
    node_counter = 0
    raw_entries = []

    if isinstance(context, list):
        for entry in context:
            # entry often looks like [title, [sent1, sent2, ...]]
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                title = str(entry[0])
                body = entry[1]
                if isinstance(body, list):
                    text = " ".join(map(str, body))
                else:
                    text = str(body)
                raw_entries.append((title, text))
            else:
                raw_entries.append((f"Context_{node_counter}", str(entry)))
                node_counter += 1
    else:
        raw_entries.append(("Context_0", str(context)))

    # Add nodes
    for idx, (title, text) in enumerate(raw_entries):
        graph.add_node(Node(idx, title, text))

    # Add edges by overlap
    node_ids = list(graph.nodes.keys())
    for i, j in itertools.combinations(node_ids, 2):
        ni = graph.nodes[i]
        nj = graph.nodes[j]

        entity_overlap = ni.entities & nj.entities
        keyword_overlap = ni.keywords & nj.keywords

        score = 0
        if entity_overlap:
            score += len(entity_overlap)
        if keyword_overlap:
            score += min(len(keyword_overlap), 3)

        if score >= EDGE_OVERLAP_THRESHOLD:
            graph.add_edge(i, j)

    return graph

# ================== EGO-GRAPHS ==================

def get_k_hop_ego_graph(graph: TextGraph, center_id: int, k: int):
    """
    Returns set of node ids in k-hop ego graph.
    """
    visited = {center_id}
    q = deque([(center_id, 0)])

    while q:
        node_id, dist = q.popleft()
        if dist == k:
            continue
        for nb in graph.adj[node_id]:
            if nb not in visited:
                visited.add(nb)
                q.append((nb, dist + 1))
    return visited

def ego_graph_to_text(graph: TextGraph, node_ids):
    """
    Flatten ego-graph for embedding / retrieval.
    """
    parts = []
    for nid in sorted(node_ids):
        n = graph.nodes[nid]
        parts.append(f"[NODE {nid}] {n.title}: {n.text}")
    return "\n".join(parts)

def build_all_ego_graphs(graph: TextGraph, k: int):
    ego_graphs = []
    for nid in graph.nodes:
        node_ids = get_k_hop_ego_graph(graph, nid, k)
        text_repr = ego_graph_to_text(graph, node_ids)
        ego_graphs.append({
            "center": nid,
            "node_ids": node_ids,
            "text": text_repr
        })
    return ego_graphs

# ================== EMBEDDING ==================

def embed_text(text: str):
    return Settings.embed_model.get_text_embedding(text)

def batch_embed_texts(texts):
    return Settings.embed_model.get_text_embedding_batch(texts)

# ================== PRUNING ==================

def score_node_relevance(node: Node, question: str, q_emb):
    """
    Heuristic soft-pruning proxy:
    combine embedding similarity + keyword overlap.
    """
    node_emb = embed_text(node.full_text)
    sim = cosine_similarity(q_emb, node_emb)

    q_keywords = extract_keywords(question)
    overlap = len(q_keywords & node.keywords)

    # weighted score
    return 0.8 * sim + 0.2 * min(overlap / 5.0, 1.0)

def prune_merged_subgraph(graph: TextGraph, merged_node_ids, question, q_emb, max_nodes=6):
    scored = []
    for nid in merged_node_ids:
        node = graph.nodes[nid]
        score = score_node_relevance(node, question, q_emb)
        scored.append((nid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    keep = [nid for nid, _ in scored[:max_nodes]]

    return set(keep), scored

# ================== HIERARCHICAL TEXT VIEW ==================

def build_hierarchical_graph_description(graph: TextGraph, kept_node_ids, question):
    """
    Approximate the paper's hierarchical text description.
    We pick the most connected / central kept node as root.
    """
    if not kept_node_ids:
        return "No relevant graph context found."

    kept_node_ids = set(kept_node_ids)

    # choose root by degree within kept subgraph
    root = max(
        kept_node_ids,
        key=lambda nid: len(graph.adj[nid] & kept_node_ids)
    )

    visited = set()
    lines = []

    def dfs(nid, depth):
        visited.add(nid)
        node = graph.nodes[nid]
        indent = "  " * depth
        lines.append(f"{indent}- NODE: {node.title}")
        lines.append(f"{indent}  TEXT: {node.text}")

        neighbors = sorted(
            list(graph.adj[nid] & kept_node_ids),
            key=lambda x: graph.nodes[x].title
        )

        for nb in neighbors:
            if nb not in visited:
                lines.append(f"{indent}  RELATION: connected_to -> {graph.nodes[nb].title}")
                dfs(nb, depth + 1)

    dfs(root, 0)

    # add disconnected kept nodes if any
    remaining = kept_node_ids - visited
    for nid in sorted(remaining):
        node = graph.nodes[nid]
        lines.append(f"- NODE: {node.title}")
        lines.append(f"  TEXT: {node.text}")

    return "\n".join(lines)

# ================== CUSTOM GRAG-STYLE ENGINE ==================

class GRAGStyleEngine:
    def __init__(self, documents_by_question):
        """
        Prebuild per-question graphs and ego-graph indices.
        documents_by_question: list of raw HotpotQA items
        """
        self.items = documents_by_question
        self.cache = []

        print("🔨 Building GRAG-style ego-graph cache...")
        for item in tqdm(self.items):
            graph = build_text_graph_from_hotpot_item(item)
            ego_graphs = build_all_ego_graphs(graph, EGO_HOPS)

            ego_texts = [eg["text"] for eg in ego_graphs]
            ego_embs = batch_embed_texts(ego_texts) if ego_texts else []

            for eg, emb in zip(ego_graphs, ego_embs):
                eg["embedding"] = emb

            self.cache.append({
                "graph": graph,
                "ego_graphs": ego_graphs
            })

    def query(self, item_idx, question):
        """
        Query against graph built from same HotpotQA sample context.
        """
        entry = self.cache[item_idx]
        graph = entry["graph"]
        ego_graphs = entry["ego_graphs"]

        if not ego_graphs:
            return "No graph evidence available."

        q_emb = embed_text(question)

        # Rank ego-graphs
        ranked = []
        for eg in ego_graphs:
            sim = cosine_similarity(q_emb, eg["embedding"])
            ranked.append((eg, sim))
        ranked.sort(key=lambda x: x[1], reverse=True)

        top_egs = [eg for eg, _ in ranked[:TOP_N_EGOGRAPHS]]

        # Merge top ego-graphs
        merged_node_ids = set()
        for eg in top_egs:
            merged_node_ids |= eg["node_ids"]

        # Heuristic soft pruning
        kept_node_ids, scored_nodes = prune_merged_subgraph(
            graph,
            merged_node_ids,
            question,
            q_emb,
            max_nodes=MAX_PRUNED_NODES
        )

        # Build hierarchical text description
        graph_description = build_hierarchical_graph_description(
            graph,
            kept_node_ids,
            question
        )

        prompt = f"""
You are answering a question using graph-structured retrieved evidence.

Question:
{question}

Retrieved Graph Context (hierarchical text view):
{graph_description}

Instructions:
- Answer only using the retrieved graph context.
- If the answer is not clearly supported, say so.
- Prefer concise, factual answers.
- For multi-hop questions, reason across connected nodes before answering.
"""
        response = Settings.llm.complete(prompt)
        return str(response)

# ================== LOAD DATA ==================

print(f"📥 Loading {NUM_QUESTIONS} HotpotQA questions...")
dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
data = dataset.select(range(NUM_QUESTIONS))
data = [item for item in data]

documents = []
for item in data:
    ctx = item["context"]
    text = "\n\n".join([str(c) for c in ctx]) if isinstance(ctx, list) else str(ctx)
    documents.append(Document(text=text))

print(f"✅ Created {len(documents)} documents")

# ===================== VANILLA RAG =====================

print("🔨 Building Vanilla RAG...")
vanilla_index = VectorStoreIndex.from_documents(documents)
vanilla_engine = vanilla_index.as_query_engine(similarity_top_k=VANILLA_TOP_K)

# ===================== GRAG-STYLE =====================

grag_engine = GRAGStyleEngine(data)

# ========================= EVALUATION =========================

results = []
print(f"🚀 Running comparison on {NUM_QUESTIONS} questions...")

for idx, item in enumerate(tqdm(data)):
    question = item["question"]
    gold = item["answer"].strip()

    # Vanilla
    start = time.time()
    v_resp = vanilla_engine.query(question)
    v_time = time.time() - start
    v_ans = str(v_resp).strip()

    # GRAG-style
    start = time.time()
    g_ans = grag_engine.query(idx, question)
    g_time = time.time() - start

    results.append({
        "question": question,
        "gold": gold,
        "vanilla_answer": v_ans,
        "grag_style_answer": g_ans,
        "vanilla_time": round(v_time, 2),
        "grag_style_time": round(g_time, 2),
    })

with open("grag_style_comparison.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

avg_v = sum(r["vanilla_time"] for r in results) / len(results)
avg_g = sum(r["grag_style_time"] for r in results) / len(results)

print("\n✅ FINISHED!")
print(f"Average Vanilla Latency   : {avg_v:.2f} seconds")
print(f"Average GRAG-Style Latency: {avg_g:.2f} seconds")
print("Results saved to grag_style_comparison.json")