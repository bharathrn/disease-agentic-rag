# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph_workflow import graph, MedState
import logging
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

"""  
FastAPI layer that routes UI calls to the single LangGraph tools_node.
Two endpoints kept for UI simplicity:
  - POST /get_diseases   { "symptoms": "..." }
  - POST /get_treatments { "disease": "..." }
Both endpoints call the same graph node; the graph decides which underlying tool to call.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from agent import app as lg_app  # compiled LangGraph app
# (agent_graph also exports call_tools_node if you want to call directly)

app = FastAPI(title="Disease Agent API (single tools node)")

# Enable CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request models ----------
class SymptomsIn(BaseModel):
    symptoms: str

class DiseaseIn(BaseModel):
    disease: str

# ---------- Endpoints ----------
@app.post("/get_diseases")
def get_diseases(req: SymptomsIn):
    state = {"symptoms": req.symptoms}
    # invoke the compiled graph — entry point is tools_node
    try:
        res = lg_app.invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if res is None:
        raise HTTPException(status_code=500, detail="Graph returned no result")
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    # res should contain 'diseases'
    diseases = res.get("diseases", [])
    return {"diseases": diseases}


@app.post("/get_treatments")
def get_treatments(req: DiseaseIn):
    state = {"disease": req.disease}
    try:
        res = lg_app.invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if res is None:
        raise HTTPException(status_code=500, detail="Graph returned no result")
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    treatments = res.get("treatments", [])
    return {"treatments": treatments}


# Optional health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}


# Mount a "static" directory for frontend files
app.mount("/", StaticFiles(directory="static", html=True), name="static")


