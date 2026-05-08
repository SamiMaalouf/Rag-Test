import time
import json
import os
import nest_asyncio
from tqdm import tqdm
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, PropertyGraphIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from datasets import load_dataset

nest_asyncio.apply()   # ← This fixes the nested async error

# ================== CONFIG ==================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "Missing GEMINI_API_KEY. Create a .env file with GEMINI_API_KEY=... or set it in your environment."
    )

Settings.llm = Gemini(
    model="models/gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

print("📥 Loading 15 HotpotQA questions...")
dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
data = dataset.select(range(15))
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
vanilla_engine = vanilla_index.as_query_engine(similarity_top_k=5)

# ===================== GRAPH RAG =====================
print("🔨 Building GraphRAG (this may take 1-3 minutes)...")
graph_index = PropertyGraphIndex.from_documents(
    documents,
    include_embeddings=True,
    kg_triplet_extract_fn=None,
    use_llm_for_triplet_extraction=False
)
graph_engine = graph_index.as_query_engine(similarity_top_k=5)

# ========================= EVALUATION =========================
results = []
print("🚀 Running comparison on 15 questions...")

for item in tqdm(data):
    question = item["question"]
    gold = item["answer"].strip()
    
    # Vanilla
    start = time.time()
    v_resp = vanilla_engine.query(question)
    v_time = time.time() - start
    v_ans = str(v_resp).strip()
    
    # GraphRAG
    start = time.time()
    g_resp = graph_engine.query(question)
    g_time = time.time() - start
    g_ans = str(g_resp).strip()
    
    results.append({
        "question": question,
        "gold": gold,
        "vanilla_answer": v_ans,
        "graph_answer": g_ans,
        "vanilla_time": round(v_time, 2),
        "graph_time": round(g_time, 2),
    })

with open("graphrag_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

avg_v = sum(r["vanilla_time"] for r in results) / len(results)
avg_g = sum(r["graph_time"] for r in results) / len(results)

print(f"\n✅ FINISHED!")
print(f"Average Vanilla Latency : {avg_v:.2f} seconds")
print(f"Average GraphRAG Latency: {avg_g:.2f} seconds")
print("Results saved!")