# app/main.py
import os
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

@app.post("/api/v1/query")
async def handle_analyst_query(payload: QueryRequest):
    cache_key = f"user:{payload.user_id}:query:{hash(payload.query.strip().lower())}"
    
    # 1. Redis Caching Guard
    cached_val = cache.get(cache_key)
    if cached_val:
        return {"source": "redis_cache", "result": cached_val}

    # 2. Pipeline Execution
    try:
        execution_state = agent.run(payload.query)
        
        # If blocked by safety boundaries
        if not execution_state["is_safe"]:
            return {"source": "guardrail_shield", "result": "Request blocked by safety policy."}
            
        output = execution_state["final_output"]
        
        # 3. Hydrate cache
        cache.set(cache_key, output, ttl=300) # Cache for 5 minutes
        
        return {"source": "live_compute_nodes", "result": output}
        
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