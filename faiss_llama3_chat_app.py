# faiss_llama3_chat_app.py

import os
import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM


from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM  


from langchain_community.vectorstores import FAISS as LangchainFAISS
import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain.chains import RetrievalQA
from py2neo import Graph, Node, Relationship
from datetime import datetime
from dotenv import load_dotenv

from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import torch
print("CUDA available:", torch.cuda.is_available())


# --- Load environment variables from .env file ---
load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "llama3")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR", "docs")
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS","Ka1smbPooh")

# --- Load and Embed Documents (with update check) ---
def build_or_update_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    os.makedirs(DOCUMENT_DIR, exist_ok=True)

    vectorstore = None
    processed_files = set()
    index_exists = os.path.exists(INDEX_PATH)

    if index_exists:
        vectorstore = LangchainFAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        processed_path = os.path.join(INDEX_PATH, "processed")
        processed_files = set(os.listdir(processed_path)) if os.path.exists(processed_path) else set()

    new_docs = []
    for fname in os.listdir(DOCUMENT_DIR):
        if fname.endswith(".txt") and fname not in processed_files:
            loader = TextLoader(os.path.join(DOCUMENT_DIR, fname))
            new_docs.extend(loader.load())
            processed_files.add(fname)

    if new_docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        docs = splitter.split_documents(new_docs)

        if vectorstore:
            vectorstore.add_documents(docs)
        else:
            vectorstore = LangchainFAISS.from_documents(docs, embeddings)

        # 🧠 Move FAISS index to GPU
        if hasattr(vectorstore, "index"):
            try:
                res = faiss.StandardGpuResources()
                vectorstore.index = faiss.index_cpu_to_gpu(res, 0, vectorstore.index)
            except Exception as e:
                print("[WARNING] Failed to move FAISS to GPU:", e)

        vectorstore.save_local(INDEX_PATH)
        os.makedirs(os.path.join(INDEX_PATH, "processed"), exist_ok=True)
        for doc in new_docs:
            open(os.path.join(INDEX_PATH, "processed", os.path.basename(doc.metadata['source'])), "w").close()

    return vectorstore


# --- Initialize Retriever and LLM ---
def create_qa_chain():
    vectorstore = build_or_update_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = OllamaLLM(model=LLM_MODEL)
    if not llm:
        raise ValueError("LLM model not found. Please check your configuration.")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# --- Neo4j Graph Save ---
def extract_triples(llm, question, answer):
    import json
    import re

    prompt = f"""Extract all semantic triples from the following Q&A as a JSON array. 
Each triple should have "subject", "predicate", and "object" fields. Output only the JSON array.

Q: {question}
A: {answer}

JSON:
"""

    response = llm.invoke(prompt)

    # Try to isolate the first valid JSON array using regex
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
        print(f"[DEBUG] Raw response: {response}")
    return []




def save_to_neo4j(chat_history):
    from langchain_ollama import OllamaLLM
    from collections import defaultdict
    llm = OllamaLLM(model=LLM_MODEL)

    print(f"[DEBUG] Connecting to Neo4j at {NEO4J_URL} with user {NEO4J_USER}")
    graph = Graph(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASS))
    try:
        tx = graph.begin()
        conv_node = Node("Conversation", timestamp=str(datetime.now()))
        tx.create(conv_node)

        graph_data = {"nodes": [], "edges": []}
        graph_data["nodes"].append({"id": str(conv_node.identity), "label": "Conversation"})

        seen_entities = set()
        entity_to_answers = defaultdict(set)
        answer_node_ids = {}

        for i, (q, a) in enumerate(chat_history):
            q_text = q["query"] if isinstance(q, dict) and "query" in q else str(q)
            a_text = a["result"] if isinstance(a, dict) and "result" in a else str(a)

            # Create question node
            q_node = Node("Question", text=q_text)
            tx.create(q_node)
            q_node_id = f"Question_{i}"
            graph_data["nodes"].append({"id": q_node_id, "label": "Question", "text": q_text})
            graph_data["edges"].append({"source": str(conv_node.identity), "target": q_node_id, "label": "HAS_QUESTION"})

            # Create answer node
            a_node = Node("Answer", text=a_text)
            tx.create(a_node)
            a_node_id = f"Answer_{i}"
            answer_node_ids[i] = a_node_id
            graph_data["nodes"].append({"id": a_node_id, "label": "Answer", "text": a_text})
            graph_data["edges"].append({"source": q_node_id, "target": a_node_id, "label": "HAS_ANSWER"})

            # Extract triples
            triples = extract_triples(llm, q_text, a_text)
            for triple in triples:
                subj = Node("Entity", name=triple["subject"])
                obj = Node("Entity", name=triple["object"])
                rel = Relationship(subj, triple["predicate"].upper().replace(" ", "_"), obj)
                tx.merge(subj, "Entity", "name")
                tx.merge(obj, "Entity", "name")
                tx.merge(rel)

                for ent in (triple["subject"], triple["object"]):
                    if ent not in seen_entities:
                        graph_data["nodes"].append({"id": ent, "label": "Entity", "text": ent})
                        seen_entities.add(ent)
                    entity_to_answers[ent].add(i)

                graph_data["edges"].append({"source": triple["subject"], "target": triple["object"], "label": triple["predicate"]})

        # Add Answer-to-Answer links based on shared entities
        linked_pairs = set()
        for answer_indexes in entity_to_answers.values():
            answer_list = list(answer_indexes)
            for i in range(len(answer_list)):
                for j in range(i + 1, len(answer_list)):
                    a1 = answer_list[i]
                    a2 = answer_list[j]
                    pair = tuple(sorted((a1, a2)))
                    if pair not in linked_pairs:
                        id1 = answer_node_ids[a1]
                        id2 = answer_node_ids[a2]
                        graph_data["edges"].append({"source": id1, "target": id2, "label": "SHARES_ENTITY"})
                        linked_pairs.add(pair)

        graph.commit(tx)

    except Exception as e:
        import traceback
        tx.rollback()
        print("[ERROR] Neo4j transaction failed:")
        traceback.print_exc()
        raise RuntimeError(f"[ERROR] Failed to save chat to Neo4j: {e}")

    return graph_data




# --- Streamlit UI ---
st.set_page_config(page_title="LLaMA 3 RAG Chat", layout="wide")
st.title("🦙 LLaMA 3 + FAISS Chatbot")
qa_chain = create_qa_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("🗂️ Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.expander(f"Q{i+1}: {q}", expanded=False):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
    from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

if st.button("💾 show graph on conversation"):
    if not st.session_state.chat_history:
        st.warning("No chat history to save.")
    else:
        try:
            graph_data = save_to_neo4j(st.session_state.chat_history)
            st.success("Saved to Neo4j!")

            # --- Create Pyvis graph ---
            net = Network(height="500px", width="50%", bgcolor="#979090", font_color="black")
            for node in graph_data["nodes"]:
                net.add_node(node["id"], label=node["label"], title=node.get("text", node["label"]))
            for edge in graph_data["edges"]:
                net.add_edge(edge["source"], edge["target"], label=edge["label"])

            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                net.save_graph(tmp_file.name)
                tmp_path = tmp_file.name

            st.markdown("### 📈 Knowledge Graph")
            components.html(open(tmp_path, "r", encoding="utf-8").read(), height=550)
        except Exception as e:
            st.error(f"❌ Failed to save and render graph: {e}")



user_question = st.text_input("Ask a question about your documents:", placeholder="What is this about?")

if user_question:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke(user_question)
    st.session_state.chat_history.append((user_question, result))
    st.markdown("---")
    st.markdown("### 💬 Answer")
    st.write(result)

    try:
        graph_data = save_to_neo4j(st.session_state.chat_history)

        # --- Create Pyvis graph ---
        net = Network(height="500px", width="50%", bgcolor="#979090", font_color="black")
        for node in graph_data["nodes"]:
            net.add_node(node["id"], label=node["label"], title=node.get("text", node["label"]))
        for edge in graph_data["edges"]:
            net.add_edge(edge["source"], edge["target"], label=edge["label"])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            tmp_path = tmp_file.name

        st.markdown("### 📈 Knowledge Graph")
        components.html(open(tmp_path, "r", encoding="utf-8").read(), height=550)
    except Exception as e:
        st.error(f"❌ Failed to save and render graph: {e}")

