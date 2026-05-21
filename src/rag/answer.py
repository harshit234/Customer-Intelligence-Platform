"""
answer.py â€” Grounded Q&A answering with Google Gemini API using retrieved evidence.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.rag.retrieve import retrieve
load_dotenv()
PROMPT_VERSION = "v1.0"
def answer_question(
    query: str,
    filters: dict | None = None,
    top_k: int = 5,
    threshold: float = 0.45,
    index=None,
    metadata=None,
    model=None,
) -> dict:
    """
    Retrieve evidence and generate a grounded answer using Gemini.
    Parameters
    ----------
    query : str
        The user's question.
    filters : dict, optional
        Metadata filters to pass to retrieve().
    top_k : int
        Number of evidence documents to retrieve.
    threshold : float
        Similarity threshold for retrieval.
    index : faiss.Index, optional
        Pre-loaded FAISS index.
    metadata : list of dict, optional
        Pre-loaded metadata.
    model : SentenceTransformer, optional
        Pre-loaded embedding model.
    Returns
    -------
    dict
        Response with answer, evidence_ids, evidence_sufficiency, and prompt_version.
    """
    evidence = retrieve(
        query=query,
        top_k=top_k,
        threshold=threshold,
        filters=filters,
        index=index,
        metadata=metadata,
        model=model,
    )
    if not evidence:
        return {
            "answer": "Insufficient evidence to answer.",
            "evidence_ids": [],
            "evidence_sufficiency": "NONE",
            "prompt_version": PROMPT_VERSION,
        }
    context_parts = []
    evidence_ids = []
    for item in evidence:
        evidence_ids.append(item["complaint_id"])
        part = (
            f"Complaint ID: {item['complaint_id']}\n"
            f"Date: {item['date']}\n"
            f"Product: {item['product']}\n"
            f"Company: {item['company']}\n"
            f"Issue: {item['issue']}\n"
            f"Narrative: {item['narrative']}\n"
            f"----------------------------------------"
        )
        context_parts.append(part)
    context_str = "\n".join(context_parts)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your-gemini-api-key-here":
        print("WARNING: GEMINI_API_KEY is not set. Running in simulated fallback mode.")
        top_narrative = evidence[0]["narrative"]
        snippet = top_narrative[:300] + "..." if len(top_narrative) > 300 else top_narrative
        simulated_answer = (
            f"[Simulated response based on Complaint ID {evidence[0]['complaint_id']}]: "
            f"Regarding your query, the retrieved complaint from {evidence[0]['date']} about {evidence[0]['product']} "
            f"for company {evidence[0]['company']} states: \"{snippet}\""
        )
        return {
            "answer": simulated_answer,
            "evidence_ids": evidence_ids,
            "evidence_sufficiency": "HIGH" if evidence[0]["score"] >= 0.55 else "MEDIUM",
            "prompt_version": f"{PROMPT_VERSION}-simulated",
        }
    GEMINI_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-001",
        "gemini-1.5-pro",
    ]
    llm = None
    used_model = None
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            candidate = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.0,
                google_api_key=api_key,
            )
            llm = candidate
            used_model = model_name
            print(f"Using Gemini model: {used_model}")
            break
        except Exception as e:
            last_error = e
            print(f"Model '{model_name}' could not be initialised: {e}. Trying next...")
            continue
    if llm is None:
        print(f"Error: No Gemini model available. Last error: {last_error}")
        return {
            "answer": f"Gemini LLM unavailable. Evidence retrieved: {evidence_ids}",
            "evidence_ids": evidence_ids,
            "evidence_sufficiency": "LOW",
            "prompt_version": PROMPT_VERSION,
        }
    try:
        system_message = (
            "You are an expert customer intelligence analyst at Meridian Financial.\n"
            "Your task is to answer the user's question about customer complaints using ONLY the provided complaints context.\n"
            "Do not use any external knowledge. If the provided context is not sufficient to answer the question, "
            "say 'Insufficient evidence to answer.' and set evidence_sufficiency to 'NONE'.\n"
            "Format your output as a valid JSON object with the following keys:\n"
            "{\n"
            "  \"answer\": \"Your detailed answer here, citing specific complaint IDs and dates if relevant.\",\n"
            "  \"evidence_sufficiency\": \"HIGH\" | \"MEDIUM\" | \"LOW\" | \"NONE\"\n"
            "}\n"
            "Do not include any markdown formatting like ```json or ```. Return raw JSON only."
        )
        user_message = (
            f"User Question: {query}\n\n"
            f"Retrieved Complaints Context:\n"
            f"{context_str}"
        )
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]
        response = llm.invoke(messages)
        response_text = response.content.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1)
        if response_text.endswith("```"):
            response_text = response_text.rsplit("```", 1)[0]
        response_text = response_text.strip()
        try:
            data = json.loads(response_text)
            return {
                "answer": data.get("answer", ""),
                "evidence_ids": evidence_ids,
                "evidence_sufficiency": data.get("evidence_sufficiency", "MEDIUM"),
                "prompt_version": PROMPT_VERSION,
            }
        except json.JSONDecodeError:
            return {
                "answer": response_text,
                "evidence_ids": evidence_ids,
                "evidence_sufficiency": "HIGH",
                "prompt_version": PROMPT_VERSION,
            }
    except Exception as e:
        print(f"Error calling Gemini LLM: {e}")
        return {
            "answer": f"Error calling Gemini LLM. Retaining evidence ids: {evidence_ids}",
            "evidence_ids": evidence_ids,
            "evidence_sufficiency": "LOW",
            "prompt_version": PROMPT_VERSION,
        }
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Answer customer complaints query.")
    parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()
    res = answer_question(args.query)
    print(json.dumps(res, indent=2))
