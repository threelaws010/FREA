# neo4j_kg_answer.py
import os
import json
import textwrap
import requests
import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network
from streamlit.components.v1 import html

# === Extra deps
import tempfile
import importlib.util
import types
import uuid
import networkx as nx
from typing import Tuple, List, Dict, Any

# --- NEW: .env support
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", ".env"), override=False)
except Exception:
    # dotenv is optional; the app still works with OS env vars
    pass

# ---------------------------
# Helpers
# ---------------------------
def env_first(*keys: str, default: str = "") -> str:
    """Return the first defined env var among keys, else default."""
    for k in keys:
        v = os.getenv(k)
        if v is not None and v != "":
            return v
    return default

# ---------------------------
# Config (pulled from .env with fallbacks)
# ---------------------------
DEFAULT_LM_BASE_URL = env_first("LMSTUDIO_BASE_URL", "OPENAI_BASE_URL", default="http://192.168.1.119:1234/v1")
DEFAULT_LM_API_KEY  = env_first("LMSTUDIO_API_KEY", "OPENAI_API_KEY", "API_KEY", default="")
DEFAULT_LM_MODEL    = env_first("LMSTUDIO_CHAT_MODEL", "LLM_MODEL", "LLM", "MODEL", default="local-model")

DEFAULT_NEO4J_URI      = env_first("NEO4J_URI", "NEO4J_URL", default="bolt://localhost:7687")
DEFAULT_NEO4J_USER     = env_first("NEO4J_USER", "NEO4J_USERNAME", default="neo4j")
DEFAULT_NEO4J_PASSWORD = env_first("NEO4J_PASSWORD", "NEO4J_PASS", default="password")
DEFAULT_NEO4J_DB       = env_first("NEO4J_DATABASE", "NEO4J_DB", default="neo4j")

# ---------------------------
# LLM helpers (OpenAI-compatible)
# ---------------------------
def openai_chat(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]], temperature: float = 0) -> str:
    """
    Calls an OpenAI-compatible /v1/chat/completions endpoint (LM Studio compatible).
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

EXTRACTION_SYSTEM_PROMPT = """You extract cyber threat intelligence relationships from text.
Return STRICT JSON with this schema, and nothing else:

{
  "entities": [
    { "name": "string", "type": "Actor|Tool|Technique|Vulnerability|Malware|Infrastructure|Asset|Indicator", "props": { "key": "value" } }
  ],
  "relations": [
    { "subject": "entity_name", "predicate": "USES|TARGETS|EXPLOITS|DELIVERS|COMMUNICATES_WITH|INDICATES|MITIGATES|RELATED_TO",
      "object": "entity_name", "props": { "key": "value" } }
  ]
}

Rules:
- Keep names short (canonical).
- Only include relationships actually present or strongly implied.
- If uncertain, omit.
- Use entity names consistently between entities and relations.
- props is optional. If none, use {}.
"""

def extract_cyber_graph_from_text(base_url: str, api_key: str, model: str, answer_text: str):
    user_prompt = f"""Extract cyber entities and relations from the following text:

TEXT:
{answer_text}
"""
    raw = openai_chat(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    # Try to locate JSON in response
    json_str = raw.strip()
    if "{" in raw and "}" in raw:
        json_str = raw[raw.find("{"): raw.rfind("}") + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        fixer_prompt = f"Fix this to be strictly valid JSON following the schema: ```{raw}```"
        fixed = openai_chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": "You output ONLY valid JSON. No commentary."},
                {"role": "user", "content": fixer_prompt},
            ],
            temperature=0,
        )
        json_str = fixed[fixed.find("{"): fixed.rfind("}") + 1]
        data = json.loads(json_str)

    # Normalize
    data.setdefault("entities", [])
    data.setdefault("relations", [])
    for e in data["entities"]:
        e.setdefault("props", {})
    for r in data["relations"]:
        r.setdefault("props", {})

    return data, raw

# ---------------------------
# Neo4j helpers
# ---------------------------
def get_driver(uri: str, user: str, password: str):
    return GraphDatabase.driver(uri, auth=(user, password))

def ensure_constraints(session) -> None:
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")

def _props_to_cypher_updates(props: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if not props:
        return "", {}
    pairs = ", ".join([f"{k}: $prop_{k}" for k in props])
    params = {f"prop_{k}": v for k, v in props.items()}
    return pairs, params

def merge_graph(session, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> None:
    # Entities
    for e in entities:
        name = e["name"].strip()
        etype = e.get("type", "Entity")
        props_pairs, params = _props_to_cypher_updates(e.get("props", {}) or {})
        params.update({"name": name})

        if props_pairs:
            cypher = textwrap.dedent(f"""
            MERGE (n:Entity:`{etype}` {{name: $name}})
            SET n += {{{props_pairs}}}
            """)
        else:
            cypher = f"MERGE (n:Entity:`{etype}` {{name: $name}})"
        session.run(cypher, **params)

    # Relations
    for r in relations:
        s = r["subject"].strip()
        o = r["object"].strip()
        pred = r["predicate"].strip().upper().replace(" ", "_")
        props_pairs, params = _props_to_cypher_updates(r.get("props", {}) or {})
        params.update({"s": s, "o": o})

        if props_pairs:
            cypher = textwrap.dedent(f"""
            MATCH (a:Entity {{name: $s}}), (b:Entity {{name: $o}})
            MERGE (a)-[r:`{pred}`]->(b)
            SET r += {{{props_pairs}}}
            """)
        else:
            cypher = f"""
            MATCH (a:Entity {{name: $s}}), (b:Entity {{name: $o}})
            MERGE (a)-[r:`{pred}`]->(b)
            """
        session.run(cypher, **params)

def fetch_subgraph(session, center_names=None, limit=200):
    if center_names:
        result = session.run("""
        MATCH (n:Entity)
        WHERE n.name IN $names
        OPTIONAL MATCH (n)-[r]-(:Entity) 
        RETURN n, r, endNode(r) as m
        LIMIT $limit
        """, names=center_names, limit=limit)
    else:
        result = session.run("""
        MATCH (n:Entity)-[r]-() RETURN n, r, endNode(r) as m LIMIT $limit
        """, limit=limit)

    nodes = {}
    edges = []
    for rec in result:
        n = rec["n"]; m = rec["m"]; r = rec["r"]
        if n:
            nodes[n.id] = {"id": n.id, "label": n["name"], "group": ":".join(list(n.labels))}
        if m:
            nodes[m.id] = {"id": m.id, "label": m["name"], "group": ":".join(list(m.labels))}
        if r and m and n:
            # try to show the actual relationship type (e.g., USES)
            rel_type = getattr(r, "type", None)
            if callable(rel_type):  # neo4j <5 sometimes exposes as method
                rel_type = r.type()
            rel_label = rel_type if rel_type else "RELATED"
            edges.append({"source": n.id, "target": m.id, "label": str(rel_label)})
    return list(nodes.values()), edges

def render_pyvis(nodes, edges, height="640px") -> Network:
    net = Network(height=height, width="100%", directed=True, notebook=False)
    for n in nodes:
        net.add_node(n["id"], label=n["label"], title=n["group"])
    for e in edges:
        net.add_edge(e["source"], e["target"], title=e["label"], label=e["label"])
    return net

# === uploaded module support ===
def _load_module_from_bytes(py_bytes: bytes) -> types.ModuleType:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    tmp.write(py_bytes)
    tmp.flush()
    tmp.close()
    mod_name = f"kgmod_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, tmp.name)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return module, tmp.name

def _normalize_to_nodes_edges(result):
    if isinstance(result, Network):
        return result  # already a pyvis network

    if isinstance(result, (nx.Graph, nx.DiGraph)):
        nodes = [{"id": i, "label": str(n), "group": ""} for i, n in enumerate(result.nodes())]
        id_map = {n: i for i, n in enumerate(result.nodes())}
        edges = []
        for (u, v, d) in result.edges(data=True):
            edges.append({"source": id_map[u], "target": id_map[v], "label": str(d.get("relation", ""))})
        return nodes, edges

    if isinstance(result, dict) and "nodes" in result and "edges" in result:
        return result["nodes"], result["edges"]

    try:
        triples = list(result)
        if len(triples) == 0:
            return [], []
        node_set = set()
        edges = []
        for t in triples:
            if not (isinstance(t, (tuple, list)) and len(t) in (2, 3)):
                raise ValueError("Items must be (src,dst) or (src,rel,dst).")
            if len(t) == 2:
                u, v = t
                rel = ""
            else:
                u, rel, v = t
            node_set.add(u); node_set.add(v)
            edges.append({"source": u, "target": v, "label": str(rel)})
        nodes = []
        id_map = {}
        for i, n in enumerate(sorted(node_set, key=lambda x: str(x))):
            id_map[n] = i
            nodes.append({"id": i, "label": str(n), "group": ""})
        for e in edges:
            e["source"] = id_map[e["source"]]
            e["target"] = id_map[e["target"]]
        return nodes, edges
    except TypeError:
        pass

    raise ValueError("Unsupported return type from uploaded function.")

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="LM Studio → Neo4j (Cyber KG)", layout="wide")

# === NEW: Global refresh button at the very top
top_cols = st.columns([1, 8, 1])
with top_cols[0]:
    if st.button("🔄 Refresh"):
        st.rerun()

st.sidebar.header("LM Studio (OpenAI API compatible)")
lm_base_url = st.sidebar.text_input("Base URL", value=DEFAULT_LM_BASE_URL, help="e.g., http://localhost:1234/v1")
lm_api_key  = st.sidebar.text_input("API Key (optional)", type="password", value=DEFAULT_LM_API_KEY)
lm_model    = st.sidebar.text_input("Model", value=DEFAULT_LM_MODEL)

st.sidebar.header("Neo4j")
neo4j_uri  = st.sidebar.text_input("URI", value=DEFAULT_NEO4J_URI)
neo4j_user = st.sidebar.text_input("User", value=DEFAULT_NEO4J_USER)
neo4j_pass = st.sidebar.text_input("Password", type="password", value=DEFAULT_NEO4J_PASSWORD)
neo4j_db   = st.sidebar.text_input("Database", value=DEFAULT_NEO4J_DB)

st.title("🔗 LM Studio → Neo4j Cyber Knowledge Graph")

# Chat area
with st.expander("💬 Chat with Local LLM", expanded=True):
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""

    for m in st.session_state.messages:
        if m["role"] in ("user", "assistant"):
            st.chat_message(m["role"]).write(m["content"])

    user_input = st.chat_input("Ask something (cyber context works best)...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("LLM thinking..."):
            try:
                reply = openai_chat(
                    base_url=lm_base_url,
                    api_key=lm_api_key,
                    model=lm_model,
                    messages=st.session_state.messages,
                    temperature=0.2,
                )
            except Exception as e:
                st.error(f"LLM request failed: {e}")
                reply = ""

        if reply:
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.last_answer = reply
            st.chat_message("assistant").write(reply)

# Extraction controls
st.subheader("🧠 Extract cyber relationships from the last answer")
custom_text = st.text_area("Or paste custom text to extract from (overrides last answer):", height=140)
col_a, _ = st.columns([1, 2])

with col_a:
    do_extract = st.button("Extract → Build in Neo4j", type="primary")

extracted = None
raw_extractor_output = None

if do_extract:
    source_text = custom_text.strip() or st.session_state.get("last_answer", "").strip()
    if not source_text:
        st.warning("No text to extract. Either chat first or paste text.")
    else:
        with st.spinner("Extracting entities/relations via LLM..."):
            try:
                extracted, raw_extractor_output = extract_cyber_graph_from_text(
                    base_url=lm_base_url,
                    api_key=lm_api_key,
                    model=lm_model,
                    answer_text=source_text,
                )
            except Exception as e:
                st.error(f"Extraction failed: {e}")

        if extracted:
            st.success(f"Extracted {len(extracted['entities'])} entities and {len(extracted['relations'])} relations.")
            st.json(extracted)

            # Push to Neo4j
            try:
                driver = get_driver(neo4j_uri, neo4j_user, neo4j_pass)
                with driver.session(database=neo4j_db) as sess:
                    ensure_constraints(sess)
                    merge_graph(sess, extracted["entities"], extracted["relations"])
                driver.close()
                st.success("Neo4j graph updated ✅")
            except Exception as e:
                st.error(f"Neo4j error: {e}")

# Visualization
st.subheader("🕸️ Graph preview from Neo4j")
if "center_names_csv" not in st.session_state:
    st.session_state.center_names_csv = ""
if "graph_limit" not in st.session_state:
    st.session_state.graph_limit = 200

st.session_state.center_names_csv = st.text_input(
    "Focus on specific entity names (comma-separated, optional)",
    value=st.session_state.center_names_csv,
)
st.session_state.graph_limit = st.number_input(
    "Max relationships to fetch", min_value=10, max_value=2000,
    value=int(st.session_state.graph_limit), step=10
)

col_g1, col_g2 = st.columns([1, 1])
with col_g1:
    viz_btn = st.button("Refresh Graph")
with col_g2:
    # quick top-area graph refresh (matches the ask to have an easy refresh control)
    quick_viz_btn = st.button("🔄 Quick Refresh Graph")

if viz_btn or quick_viz_btn:
    try:
        driver = get_driver(neo4j_uri, neo4j_user, neo4j_pass)
        with driver.session(database=neo4j_db) as sess:
            names = [x.strip() for x in st.session_state.center_names_csv.split(",") if x.strip()] or None
            nodes, edges = fetch_subgraph(sess, center_names=names, limit=int(st.session_state.graph_limit))
        driver.close()

        net = render_pyvis(nodes, edges, height="680px")
        tmp_html = "graph.html"
        net.save_graph(tmp_html)
        with open(tmp_html, "r", encoding="utf-8") as f:
            html(f.read(), height=720, scrolling=True)
    except Exception as e:
        st.error(f"Failed to render graph: {e}")

# === Uploaded .py builder section (optional) ===
st.subheader("📦 Upload a .py knowledge-graph builder (runs on last answer)")
st.caption("Your file should define a function that accepts text and returns a Graph, triples, nodes/edges dict, or a pyvis Network.")

if "uploaded_module" not in st.session_state:
    st.session_state.uploaded_module = None
if "uploaded_func_name" not in st.session_state:
    st.session_state.uploaded_func_name = "build_knowledge_graph"
if "uploaded_tmp_path" not in st.session_state:
    st.session_state.uploaded_tmp_path = None

ul_col1, ul_col2 = st.columns([2, 1])
with ul_col1:
    uploaded_py = st.file_uploader("Upload .py", type=["py"])
with ul_col2:
    st.text_input("Function name", key="uploaded_func_name", value=st.session_state.uploaded_func_name)

uploaded_graph_slot = st.empty()

if uploaded_py is not None:
    try:
        module, tmp_path = _load_module_from_bytes(uploaded_py.read())
        st.session_state.uploaded_module = module
        st.session_state.uploaded_tmp_path = tmp_path
        st.success("Module loaded ✔")
    except Exception as e:
        st.session_state.uploaded_module = None
        st.error("Failed to load uploaded module.")
        st.exception(e)

ul_text = st.text_area("Optional override text (otherwise uses last answer):", key="uploaded_override_text", height=120)
if st.button("Build / Refresh (Uploaded)"):
    if not st.session_state.uploaded_module:
        st.warning("Please upload a .py module first.")
    else:
        func_name = st.session_state.get("uploaded_func_name") or "build_knowledge_graph"
        try:
            func = getattr(st.session_state.uploaded_module, func_name)
        except AttributeError:
            func = None
            st.error(f"Function '{func_name}' not found in uploaded module.")
        if func:
            text_src = (ul_text or st.session_state.get("last_answer", "")).strip()
            if not text_src:
                st.warning("No text available. Chat first or provide override text.")
            else:
                try:
                    result = func(text_src)
                    norm = _normalize_to_nodes_edges(result)
                    if isinstance(norm, Network):
                        tmp_html2 = "uploaded_graph.html"
                        norm.save_graph(tmp_html2)
                        with open(tmp_html2, "r", encoding="utf-8") as f:
                            uploaded_graph_slot.html(f.read(), height=720, scrolling=True)
                    else:
                        nodes, edges = norm
                        net = render_pyvis(nodes, edges, height="680px")
                        tmp_html2 = "uploaded_graph.html"
                        net.save_graph(tmp_html2)
                        with open(tmp_html2, "r", encoding="utf-8") as f:
                            uploaded_graph_slot.html(f.read(), height=720, scrolling=True)
                except Exception as e:
                    st.error("Error running uploaded function.")
                    st.exception(e)

with st.expander("Advanced (cleanup uploaded temp file)"):
    if st.button("Unload uploaded module"):
        try:
            if st.session_state.get("uploaded_tmp_path") and os.path.exists(st.session_state["uploaded_tmp_path"]):
                os.unlink(st.session_state["uploaded_tmp_path"])
        except Exception:
            pass
        st.session_state.uploaded_module = None
        st.session_state.uploaded_tmp_path = None
        st.success("Uploaded module unloaded.")

st.caption("Tip: LM Studio (or your local OpenAI-compatible server) must be running at the base URL shown in the sidebar.")
