# app/main.py
import os
import json
import urllib.request
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from agentic import StateAgent
from app.cache import RedisCache

app = FastAPI(title="Autonomous Enterprise Analyst API")

# Enable CORS for cross-origin requests (e.g. if running UI in another container or local dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = StateAgent()
cache = RedisCache()

class QueryRequest(BaseModel):
    user_id: str
    query: str
    model: str = "llama3"

def get_ollama_models():
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    url = f"{ollama_base}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
            # Filter out non-text/embedding-only models if desired, or return all except embedding
            models = [m["name"] for m in data.get("models", []) if "embed" not in m["name"]]
            # Clean up tag names (e.g., if there's mistral:latest and mistral, keep the name as is)
            return models if models else ["llama3:latest", "llama3.2:3b", "gemma3:4b"]
    except Exception:
        return ["llama3:latest", "llama3.2:3b", "gemma3:4b", "mistral:latest", "deepseek-r1:8b"]

@app.get("/api/v1/models")
async def list_available_models():
    models = get_ollama_models()
    return {"models": models}

@app.post("/api/v1/query")
async def handle_analyst_query(payload: QueryRequest):
    # Include selected model in cache key to partition cache by model
    cache_key = f"user:{payload.user_id}:model:{payload.model}:query:{hash(payload.query.strip().lower())}"
    
    # 1. Redis Caching Guard
    cached_val = cache.get(cache_key)
    if cached_val:
        if isinstance(cached_val, dict):
            return {
                "source": "redis_cache",
                "result": cached_val.get("result"),
                "tokens": cached_val.get("tokens", {"prompt_tokens": 0, "completion_tokens": 0})
            }
        return {
            "source": "redis_cache",
            "result": cached_val,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0}
        }

    # 2. Pipeline Execution
    try:
        execution_state = agent.run(payload.query, payload.model)
        tokens = execution_state.get("tokens", {"prompt_tokens": 0, "completion_tokens": 0})
        
        # If blocked by safety boundaries
        if not execution_state["is_safe"]:
            return {
                "source": "guardrail_shield",
                "result": "Request blocked by safety policy.",
                "tokens": tokens
            }
            
        output = execution_state["final_output"]
        
        # 3. Hydrate cache
        cache_data = {"result": output, "tokens": tokens}
        cache.set(cache_key, cache_data, ttl=300) # Cache for 5 minutes
        
        return {"source": "live_compute_nodes", "result": output, "tokens": tokens}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Pipeline Error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="UI Template index.html not found. Make sure it is placed under app/templates/index.html.")
