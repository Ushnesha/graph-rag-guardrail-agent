#!/usr/bin/env python3
import os
import subprocess

def get_recent_commits():
    try:
        # Fetch the last 5 commits formatted for a markdown table
        cmd = ["git", "log", "-n", "5", "--pretty=format:%h|%an|%ad|%s", "--date=short"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = res.stdout.strip().split("\n")
        if not lines or not lines[0]:
            return "*No commit history found.*"
        
        table = "| Commit | Author | Date | Message |\n| --- | --- | --- | --- |\n"
        for line in lines:
            parts = line.split("|")
            if len(parts) >= 4:
                hash_val, author, date, msg = parts[0], parts[1], parts[2], "|".join(parts[3:])
                table += f"| `{hash_val}` | {author} | {date} | {msg} |\n"
        return table
    except Exception as e:
        return f"*Error retrieving git logs:* {str(e)}"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    readme_path = os.path.join(root_dir, "README.md")
    
    commits_table = get_recent_commits()
    
    content = f"""# GraphRAG Security & Multi-Model Routing Sandbox

An autonomous, secure GraphRAG analytical platform built with LangGraph, Neo4j, Qdrant, Redis, and local Ollama LLMs. It features a modern light-themed chat console, dynamic LLM model selection, real-time token utilization metrics, and input guardrail security auditing.

---

## 🚀 Key Features

1. **Dual-Plane Retrieval (GraphRAG):**
   - Combines semantic vector database queries (Qdrant) and structural knowledge graph queries (Neo4j driver).
   - Melds dense embeddings and BM25 indexing via Reciprocal Rank Fusion (RRF).
2. **Dynamic Downstream Model Selection:**
   - Sidebar selector allows choosing any installed Ollama model on-the-fly.
   - Fetches available models dynamically via the host's Ollama registry.
3. **Execution Caching:**
   - Caches agent states and results in Redis to bypass live execution.
   - Caching is partitioned per user ID and LLM model.
4. **Token Utilization Metrics:**
   - Automatically parses prompt and completion token statistics from model metadata.
   - Displays real-time token tracking in the dashboard.
5. **Robust Security Guardrails:**
   - An auditing node filters prompt injection attempts, database bypasses, and system prompt overrides.

---

## 🛠️ Tech Stack & Architecture

- **Orchestrator:** LangGraph state graph.
- **Backend:** FastAPI (Python 3.11).
- **Frontend:** HTML5, Tailwind-free Glassmorphic Light CSS, Vanilla JavaScript.
- **Data Planes:** Neo4j (Graph), Qdrant (Vector).
- **Cache:** Redis.
- **LLM Engine:** Local Ollama service.

---

## 📥 Getting Started

### Prerequisites
- Docker & Docker Compose
- Ollama running locally on the host machine (accessible at `http://host.docker.internal:11434`)

### Run the Stack
Rebuild and start the application container stack:
```bash
docker compose up -d --build
```

### Access Points
- **Web UI:** [http://localhost:8000](http://localhost:8000)
- **FastAPI Endpoints:**
  - `GET /api/v1/models` - Lists available reasoning models.
  - `POST /api/v1/query` - Submits RAG query payload.

---

## ⏱️ Recent Activity (Auto-Updated)

{commits_table}

---
*Note: This README is automatically updated and committed before pushes via a Git pre-push hook.*
"""
    
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Successfully updated README.md at: {readme_path}")

if __name__ == "__main__":
    main()
