#!/usr/bin/env python3
"""
build_kg_from_txt.py

Usage:
  python build_kg_from_txt.py path/to/input.txt

What it does:
  1) Reads a text file.
  2) Uses an OpenAI-compatible LLM (LocalAI/LM Studio/OpenAI) to extract triples.
  3) Creates/updates nodes & relationships in Neo4j with MERGE.

Environment variables you should set:
  # LLM (OpenAI-compatible: LocalAI/LM Studio/OpenAI)
  OPENAI_API_KEY=anything_nonempty_for_localai
  OPENAI_BASE_URL=http://localhost:8082/v1        # e.g., LocalAI mapped port
  LLM_MODEL=gpt-4o-mini                           # or your local model name

  # Neo4j
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=yourpassword

Optional:
  MAX_CHARS=12000          # max characters sent to the LLM (simple truncation)
  CHUNK_SIZE=2000          # chunk text and extract in passes
  CHUNK_OVERLAP=200
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

# -------- LLM (OpenAI-compatible) ----------
try:
    from openai import OpenAI
except Exception:
    print("Please install openai:  pip install openai>=1.40")
    raise

# -------- Neo4j driver ----------
try:
    from neo4j import GraphDatabase
except Exception:
    print("Please install neo4j driver:  pip install neo4j")
    raise


# ------------------ Config ------------------

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "local-key")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8082/v1")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

MAX_CHARS     = int(os.getenv("MAX_CHARS", "12000"))
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --------------- Helpers --------------------

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    if size <= 0:
        return [text]
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i:i+size])
        i += max(1, size - overlap)
    return chunks

def sanitize_label(label: str) -> str:
    """Neo4j labels: letters, digits, underscore; must not start with a digit."""
    if not label:
        return "Thing"
    s = re.sub(r"[^A-Za-z0-9]+", "_", label.strip())
    s = s.strip("_")
    if not s:
        s = "Thing"
    if s[0].isdigit():
        s = "_" + s
    return s

def sanitize_rel_type(rel: str) -> str:
    """Relationship types must be uppercase letters, digits, underscore; no spaces."""
    if not rel:
        return "RELATED_TO"
    s = re.sub(r"[^A-Za-z0-9]+", "_", rel.strip()).upper()
    s = s.strip("_") or "RELATED_TO"
    if s[0].isdigit():
        s = "_" + s
    return s

def dedupe_triples(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for t in triples:
        key = (
            t.get("subject","").strip(),
            t.get("predicate","").strip(),
            t.get("object","").strip(),
            t.get("subject_type","").strip(),
            t.get("object_type","").strip(),
        )
        if key not in seen:
            out.append(t)
            seen.add(key)
    return out

# --------------- LLM extraction --------------------

SYSTEM_PROMPT = """You are an information extraction assistant. From the user text, extract clean triples of factual relationships.

Return STRICT JSON with this schema:

{
  "triples": [
    {
      "subject": "string",
      "subject_type": "string",     // Person, Organization, Place, Concept, Event, Product, etc. (guess if needed)
      "predicate": "string",        // simple relational phrase/verb, e.g., "FOUNDED", "LOCATED_IN", "PART_OF"
      "object": "string",
      "object_type": "string",
      "confidence": 0.0,            // 0.0 - 1.0
      "evidence": "exact short substring from the text supporting this fact"
    }
  ]
}

Rules:
- Keep names and terms as they appear in the text (canonicalize by trimming whitespace).
- Predicate should be short and relational (no long sentences).
- Avoid duplicate triples.
- Only include facts actually supported by the text.
- Use 5-25 triples per ~2k chars if plausible; it's okay to output fewer if the text is sparse.
- NEVER include extra keys. Return ONLY the JSON object described above.
"""

USER_PROMPT_TMPL = """Extract factual triples from the following text:

TEXT:
\"\"\"{text}\"\"\""""

def call_llm_extract(client: OpenAI, text: str, retries: int = 3, sleep: float = 2.0) -> Dict[str, Any]:
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TMPL.format(text=text)},
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            if not isinstance(data, dict) or "triples" not in data:
                raise ValueError("Invalid JSON structure from LLM.")
            return data
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(sleep * attempt)
    return {"triples": []}

# --------------- Neo4j writing --------------------

CREATE_CONSTRAINTS_CYPHER = """
CREATE CONSTRAINT IF NOT EXISTS unique_entity_name_type
FOR (n:Entity)
REQUIRE (n.name, n.type) IS NODE KEY
"""

def merge_triple_tx(tx, triple: Dict[str, Any]):
    s_name = triple.get("subject","").strip()
    o_name = triple.get("object","").strip()
    s_type_raw = triple.get("subject_type","").strip() or "Thing"
    o_type_raw = triple.get("object_type","").strip() or "Thing"
    rel_raw   = triple.get("predicate","").strip() or "RELATED_TO"
    evidence  = triple.get("evidence","").strip()
    conf      = float(triple.get("confidence", 0.0))

    # Dynamic label for types (optional) + base :Entity
    s_label = sanitize_label(s_type_raw)
    o_label = sanitize_label(o_type_raw)
    rel_typ = sanitize_rel_type(rel_raw)

    # Build dynamic Cypher (rel type cannot be parameterized)
    cypher = f"""
    MERGE (s:Entity:{s_label} {{name: $s_name, type: $s_type}})
    MERGE (o:Entity:{o_label} {{name: $o_name, type: $o_type}})
    MERGE (s)-[r:{rel_typ}]->(o)
    ON CREATE SET r.evidence = $evidence, r.confidence = $confidence
    ON MATCH  SET r.evidence = $evidence, r.confidence = $confidence
    """
    tx.run(
        cypher,
        s_name=s_name,
        s_type=s_type_raw,
        o_name=o_name,
        o_type=o_type_raw,
        evidence=evidence,
        confidence=conf,
    )

def write_triples_to_neo4j(uri: str, user: str, password: str, triples: List[Dict[str, Any]]) -> int:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    count = 0
    try:
        with driver.session() as session:
            session.run(CREATE_CONSTRAINTS_CYPHER)
            # write in small batches
            batch = []
            for t in triples:
                batch.append(t)
                if len(batch) >= 64:
                    session.execute_write(lambda tx: [merge_triple_tx(tx, x) for x in batch])
                    count += len(batch)
                    batch = []
            if batch:
                session.execute_write(lambda tx: [merge_triple_tx(tx, x) for x in batch])
                count += len(batch)
    finally:
        driver.close()
    return count

# --------------- Main pipeline --------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python build_kg_from_txt.py path/to/input.txt")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(2)

    raw = path.read_text(encoding="utf-8", errors="ignore")
    if not raw.strip():
        print("Input file is empty.")
        sys.exit(3)

    # Truncate if extremely long (simple safeguard)
    text = raw[:MAX_CHARS]

    # Init OpenAI-compatible client
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    # Extract in chunks, accumulate triples
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP) if len(text) > CHUNK_SIZE else [text]
    all_triples: List[Dict[str, Any]] = []

    print(f"Extracting triples from {len(chunks)} chunk(s) using model={LLM_MODEL} ...")
    for i, ch in enumerate(chunks, 1):
        try:
            data = call_llm_extract(client, ch)
            triples = data.get("triples", [])
            print(f"  Chunk {i}/{len(chunks)} → {len(triples)} triples")
            all_triples.extend(triples)
        except Exception as e:
            print(f"  Chunk {i} extraction failed: {e}")

    if not all_triples:
        print("No triples extracted. Nothing to write.")
        sys.exit(0)

    # Dedupe & basic cleanup
    all_triples = dedupe_triples(all_triples)
    # Drop obviously broken entries
    all_triples = [
        t for t in all_triples
        if t.get("subject") and t.get("object") and t.get("predicate")
    ]

    if not all_triples:
        print("No valid triples after cleanup. Nothing to write.")
        sys.exit(0)

    print(f"Writing {len(all_triples)} triples to Neo4j at {NEO4J_URI} ...")
    written = write_triples_to_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, all_triples)
    print(f"Done. Wrote {written} triples.")

if __name__ == "__main__":
    main()
