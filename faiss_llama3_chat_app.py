# faiss_llama3_chat_app.py

import os
import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM


from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM  


from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain.chains import RetrievalQA
from py2neo import Graph, Node, Relationship
from datetime import datetime
from dotenv import load_dotenv

from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

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

    if os.path.exists(INDEX_PATH):
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        processed_files = set(os.listdir(os.path.join(INDEX_PATH, "processed")) if os.path.exists(os.path.join(INDEX_PATH, "processed")) else [])
    else:
        vectorstore = None
        processed_files = set()

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
            vectorstore = FAISS.from_documents(docs, embeddings)
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
    prompt = f"""
You are an AI that extracts structured knowledge. For the following Q&A, extract all semantic triples in the form of:
[{{"subject": "...", "predicate": "...", "object": "..."}}]

Q: {question}
A: {answer}

Output only a JSON array of triples.
"""
    response = llm.invoke(prompt)
    try:
        triples = eval(response) if isinstance(response, str) else response
        if isinstance(triples, list) and all(isinstance(t, dict) for t in triples):
            return triples
    except Exception as e:
        print(f"[ERROR] Failed to parse triples: {e}")
    return []



def save_to_neo4j(chat_history):
    from langchain_ollama import OllamaLLM
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

        for i, (q, a) in enumerate(chat_history):
            q_text = q["query"] if isinstance(q, dict) and "query" in q else str(q)
            a_text = a["result"] if isinstance(a, dict) and "result" in a else str(a)

            q_node = Node("Question", text=q_text)
            a_node = Node("Answer", text=a_text)
            tx.create(q_node)
            tx.create(a_node)
            tx.create(Relationship(conv_node, "HAS_QUESTION", q_node))
            tx.create(Relationship(q_node, "HAS_ANSWER", a_node))

            graph_data["nodes"].append({"id": str(q_node.identity), "label": "Question", "text": q_text})
            graph_data["nodes"].append({"id": str(a_node.identity), "label": "Answer", "text": a_text})
            graph_data["edges"].append({"source": str(conv_node.identity), "target": str(q_node.identity), "label": "HAS_QUESTION"})
            graph_data["edges"].append({"source": str(q_node.identity), "target": str(a_node.identity), "label": "HAS_ANSWER"})

            triples = extract_triples(llm, q_text, a_text)
            for triple in triples:
                subj = Node("Entity", name=triple["subject"])
                obj = Node("Entity", name=triple["object"])
                rel = Relationship(subj, triple["predicate"].upper().replace(" ", "_"), obj)
                tx.merge(subj, "Entity", "name")
                tx.merge(obj, "Entity", "name")
                tx.merge(rel)

                if triple["subject"] not in seen_entities:
                    graph_data["nodes"].append({"id": triple["subject"], "label": "Entity", "text": triple["subject"]})
                    seen_entities.add(triple["subject"])

                if triple["object"] not in seen_entities:
                    graph_data["nodes"].append({"id": triple["object"], "label": "Entity", "text": triple["object"]})
                    seen_entities.add(triple["object"])

                graph_data["edges"].append({"source": triple["subject"], "target": triple["object"], "label": triple["predicate"]})

        tx.commit()
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
