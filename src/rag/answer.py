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
PROMPT_VERSION = "v2.0"
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
            "You are a senior customer intelligence analyst at Meridian Financial with deep expertise in "
            "complaint pattern analysis, root cause identification, and regulatory risk assessment.\n\n"
            "TASK: Answer the user's question comprehensively using ONLY the provided complaint evidence below. "
            "Do NOT use external knowledge or assumptions beyond what the evidence states.\n\n"
            "RESPONSE REQUIREMENTS:\n"
            "1. CITE EVIDENCE IDs EXPLICITLY: Every claim you make must be backed by citing the specific "
            "Complaint ID(s) inline, e.g. '[Complaint #1234567]' or '[Complaints #111, #222]'.\n"
            "2. DETAILED & STRUCTURED: Your answer must be thorough and well-organised. Use the following sections:\n"
            "   a) Executive Summary (2-3 sentences summarising the key finding)\n"
            "   b) Evidence Analysis (analyse EACH retrieved complaint individually — its date, product, "
            "company, issue, and what the narrative reveals; cite the ID for each)\n"
            "   c) Patterns & Root Causes (identify recurring themes, common products/companies/issues "
            "across the evidence; quantify where possible, e.g. '3 out of 5 complaints mention...')\n"
            "   d) Risk & Impact Assessment (assess severity, customer impact, or regulatory risk implied "
            "by the evidence)\n"
            "   e) Recommendations (data-driven suggestions grounded strictly in the evidence)\n"
            "3. MINIMUM LENGTH: Each section must be at least 3-5 sentences. Do not give one-line answers.\n"
            "4. EVIDENCE SUFFICIENCY: After your analysis, honestly assess whether the retrieved evidence "
            "is sufficient to fully answer the question:\n"
            "   - HIGH: Evidence directly and fully answers the question with strong signal.\n"
            "   - MEDIUM: Evidence partially answers the question; some gaps remain.\n"
            "   - LOW: Evidence is tangentially related; answer is mostly inferred.\n"
            "   - NONE: Evidence is irrelevant or absent; cannot answer.\n\n"
            "OUTPUT FORMAT: Return a valid JSON object with exactly these keys (no markdown, no ```json):\n"
            "{\n"
            "  \"answer\": \"<Your full structured answer with inline Evidence ID citations>\",\n"
            "  \"evidence_sufficiency\": \"HIGH\" | \"MEDIUM\" | \"LOW\" | \"NONE\",\n"
            "  \"cited_ids\": [\"<id1>\", \"<id2>\", ...]\n"
            "}\n"
            "The 'cited_ids' list must contain every Complaint ID you explicitly referenced in your answer."
        )
        evidence_id_list = ", ".join(f"#{eid}" for eid in evidence_ids)
        user_message = (
            f"User Question: {query}\n\n"
            f"Available Evidence IDs: {evidence_id_list}\n"
            f"Total Evidence Documents Retrieved: {len(evidence)}\n\n"
            f"=== Retrieved Complaint Evidence ===\n"
            f"{context_str}\n"
            f"=== End of Evidence ===\n\n"
            f"Now provide your comprehensive, evidence-grounded answer following the required structure. "
            f"Cite every Complaint ID inline wherever you reference it."
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
            # Merge model-returned cited_ids with retrieved evidence_ids for completeness
            model_cited = data.get("cited_ids", [])
            all_ids = list(dict.fromkeys(evidence_ids + [str(c) for c in model_cited]))
            return {
                "answer": data.get("answer", ""),
                "evidence_ids": all_ids,
                "cited_ids_in_answer": model_cited,
                "evidence_sufficiency": data.get("evidence_sufficiency", "MEDIUM"),
                "prompt_version": PROMPT_VERSION,
            }
        except json.JSONDecodeError:
            return {
                "answer": response_text,
                "evidence_ids": evidence_ids,
                "cited_ids_in_answer": [],
                "evidence_sufficiency": "MEDIUM",
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
