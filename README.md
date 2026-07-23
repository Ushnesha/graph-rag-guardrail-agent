# GraphRAG Security & Multi-Model Routing Sandbox

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

| Commit | Author | Date | Message |
| --- | --- | --- | --- |
| `dd9fb2c` | Ushnesha Daripa | 2026-07-22 | added time logs for audit |
| `85b2f56` | Ushnesha Daripa | 2026-07-22 | docs: auto-update README [skip ci] |
| `204013e` | Ushnesha Daripa | 2026-07-22 | switching to mps embedding calls from ollama call, implemented batch qdrant upserting |
| `685f3c9` | Ushnesha Daripa | 2026-07-20 | docs: auto-update README [skip ci] |
| `7aa365b` | Ushnesha Daripa | 2026-07-20 | load_data.py added for adding FinQA and TatQA dataset |


---
*Note: This README is automatically updated and committed before pushes via a Git pre-push hook.*
