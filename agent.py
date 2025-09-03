from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from tools import disease_to_treatment_tool,symptom_to_disease_tool
# -------------------------
# Graph: single tools_node
# -------------------------
class State(dict):
    symptoms: str
    disease: str
    diseases: List[Dict]
    treatments: List[str]
    error: str

workflow = StateGraph(State)

def tools_node(state: State):
    """
    Single decision node. Decides which tool to call:
      - if 'symptoms' in state -> call symptom_to_disease_tool -> populate 'diseases'
      - elif 'disease' in state -> call disease_to_treatment_tool -> populate 'treatments'
      - else -> return error message
    The node returns a dict of updated fields for the state.
    """
    # Normalize keys (strip, lower)
    if "symptoms" in state and state.get("symptoms"):
        symptoms = state["symptoms"]
        # OPTIONAL: you can add preprocessing/prompts here (e.g., expand synonyms)
        diseases = symptom_to_disease_tool(symptoms)
        return {"diseases": diseases}

    if "disease" in state and state.get("disease"):
        disease = state["disease"]
        # OPTIONAL: normalize disease string (map aliases to canonical name)
        treatments = disease_to_treatment_tool(disease)
        return {"treatments": treatments}

    return {"error": "Invalid input. Provide either 'symptoms' or 'disease' in the request."}

workflow.add_node("tools_node", tools_node)

# single node graph: set entry point to tools_node
workflow.set_entry_point("tools_node")
workflow.add_edge("tools_node", END)

# compile
app = workflow.compile()