# RAG + GraphRAG (Gemini) Test

This repo contains two Python scripts that compare retrieval approaches on a small HotpotQA sample using Gemini via `llama-index`.

## Prerequisites

- Python 3.10+ recommended
- A Gemini API key

## Setup (Windows PowerShell)

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Configure environment variables

Create a `.env` file in the project root (same folder as the scripts):

```env
GEMINI_API_KEY=your_real_key_here
```

Notes:
- `.env` is ignored by git.
- You can copy from `.env.example`.

## Run

Vanilla RAG vs GraphRAG (property graph index):

```powershell
python .\gemini_rag_test.py
```

Vanilla RAG vs a custom “GRAG-style” ego-graph approach:

```powershell
python .\grag_test.py
```

## Output

- `graphrag_comparison.json`: output from `gemini_rag_test.py`
- `grag_style_comparison.json`: output from `grag_test.py`

"# Rag-Test" 
