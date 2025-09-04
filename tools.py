# tools.py
from rag.symptoms2disease_retriever import query_and_aggregate
from rag.disease2treatement_retriever import retrieve_treatments
from langchain_core.tools import tool

# --- Tool 1: Symptoms → Disease ---
def symptom_to_disease_tool(query: str) -> str:
    """
    Given a natural language symptom description,
    retrieve possible matching diseases with supporting evidence.
    """
    results = query_and_aggregate(query)
    # output = "Top disease candidates:\n"
    # for r in results:
    #     output += f"- {r['name']} ({r['disease_id']}) score={r['score']:.4f}\n"
    return results

# --- Tool 2: Disease → Treatment ---
def disease_to_treatment_tool(disease_query: str) -> str:
    results = retrieve_treatments(disease_query)
    if not results:
        return f"No treatments found for {disease_query}."

    # response = f"Possible treatments for **{results[0]['name']}**:\n"
    # for t in results[0]["treatments"]:
    #     response += f"- {t}\n"
    return results