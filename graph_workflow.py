# graph_workflow.py
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import logging

# Your RAG query functions
from tools import symptom_to_disease_tool,disease_to_treatment_tool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MedState(TypedDict):
    user_query: str
    diseases: List[dict]
    treatments: List[str]
    final_answer: str


def disease_tool(state: MedState) -> Dict:
    diseases = symptom_to_disease_tool(state["user_query"])
    logger.info(f"Disease candidates: {diseases[0]}")
    return {"diseases": diseases}


def treatment_tool(state: MedState) -> Dict:
    print(f"state {state}")
    top_disease = state["diseases"]
    treatments = disease_to_treatment_tool(top_disease)
    logger.info(f"Suggested treatments for {top_disease}: {treatments}")
    return {"treatments": treatments}


def doctor_validation(state: MedState) -> Dict:
    """
    In production → push this request to Doctor Dashboard via DB/queue/API.
    """
    logger.warning("Doctor review required.")
    return {"final_answer": "PENDING_DOCTOR_REVIEW"}


# ---- Build LangGraph workflow ----
workflow = StateGraph(MedState)
workflow.add_node("disease_tool", disease_tool)
workflow.add_node("treatment_tool", treatment_tool)
workflow.add_node("doctor_validation", doctor_validation)

workflow.set_entry_point("disease_tool")
workflow.add_edge("disease_tool", "treatment_tool")
workflow.add_edge("treatment_tool", "doctor_validation")
workflow.add_edge("doctor_validation", END)

graph = workflow.compile()
