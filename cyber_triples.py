import os
import re
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "local-model")
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")


llm = ChatOpenAI(model_name=LLM_MODEL, base_url=LMSTUDIO_BASE_URL, api_key="not-needed")


def refine_cyber_query(user_question: str) -> str:
    """Uses the LLM to rewrite a vague cybersecurity question into a focused one."""
    prompt = f"""
You are a cybersecurity expert and query generator.

Rewrite the following vague question into a focused and specific query suitable for searching cybersecurity logs, threat reports, or indicators of compromise.

Original question: "{user_question}"

Output the improved query only.
"""
    refined_query = llm.invoke(prompt).strip()
    return refined_query


def extract_cyber_triples(question: str, answer: str) -> list:
    """
    Uses the LLM to extract subject-predicate-object triples from a given Q&A.
    Returns a list of dictionaries with 'subject', 'predicate', 'object'.
    """
    prompt = f"""Extract all cybersecurity-relevant semantic triples from the following Q&A as a JSON array. 
Each triple should have "subject", "predicate", and "object" fields. Output only the JSON array.

Q: {question}
A: {answer}

JSON:
"""
    print("🧠 LLM Prompt:\n", prompt)
    response = llm.invoke(prompt)
    print("🧾 Raw LLM response:\n", response)

    try:
        json_array_match = re.search(r"\[\s*\{.*?\}\s*\]", response, re.DOTALL)
        if not json_array_match:
            raise ValueError("No JSON array found in response.")
        json_str = json_array_match.group(0)
        triples = json.loads(json_str)
        if isinstance(triples, list) and all(isinstance(t, dict) for t in triples):
            return triples
    except Exception as e:
        print(f"[ERROR] Failed to parse triples: {e}")
        print(f"[DEBUG] Raw LLM output: {response}")
    return []



# Optional test entry point
if __name__ == "__main__":
    user_q = "Is something wrong with my firewall?"
    refined = refine_cyber_query(user_q)
    print("🔍 Refined Query:", refined)

    fake_answer = "The firewall logs show repeated SSH brute-force attempts from a Russian IP address."
    triples = extract_cyber_triples(refined, fake_answer)
    print("📊 Extracted Triples:")
    for t in triples:
        print("-", t)
