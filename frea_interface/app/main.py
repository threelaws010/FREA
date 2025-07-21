import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Neo4jVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredMarkdownLoader
from pyvis.network import Network
import tempfile
import os
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URL = os.getenv("NEO4J_URL", "bolt://host.containers.internal:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "frea")

# Neo4j config

# Set up embedding + vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Neo4jVector(
    url=NEO4J_URL,
    username=NEO4J_USER ,
    password=NEO4J_PASSWORD,
    index_name="frea",
    embedding=embedding ,
)

# Load local LLaMA 3 model via Ollama
llm = Ollama(model="llama3")

# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Neo4j Driver for graph
graph_driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Extract entities (basic version)
def extract_entities(text):
    import re
    return list(set(re.findall(r"\b[A-Z][a-z]+\b", text)))

def insert_kg_triples(text):
    entities = extract_entities(text)
    with graph_driver.session() as session:
        for i in range(len(entities) - 1):
            session.run(
                """
                MERGE (a:Entity {name: $a})
                MERGE (b:Entity {name: $b})
                MERGE (a)-[:RELATED_TO]->(b)
                """,
                a=entities[i], b=entities[i+1]
            )

def visualize_kg():
    net = Network(height="400px", width="100%", notebook=False)
    with graph_driver.session() as session:
        result = session.run("MATCH (a)-[r]->(b) RETURN a.name, b.name")
        for record in result:
            net.add_node(record["a.name"], label=record["a.name"])
            net.add_node(record["b.name"], label=record["b.name"])
            net.add_edge(record["a.name"], record["b.name"])
    path = tempfile.mktemp(suffix=".html")
    net.save_graph(path)
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    return html

def process_md(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    loader = UnstructuredMarkdownLoader(tmp_path)
    docs = loader.load()
    vectorstore.add_documents(docs)
    for doc in docs:
        insert_kg_triples(doc.page_content)

# ---------------------- Streamlit UI ------------------------

st.set_page_config(page_title="LLaMA 3 Chat + KG", layout="wide")
st.title("📄 Chat with Markdown + Knowledge Graph")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat")
    user_input = st.text_input("Ask a question...")
    if user_input:
        with st.spinner("Thinking..."):
            response = qa.run(user_input)
            st.markdown(f"**Answer:** {response}")

with col2:
    st.subheader("🧠 Knowledge Graph")
    html_graph = visualize_kg()
    st.components.v1.html(html_graph, height=400, scrolling=True)

st.markdown("---")
st.subheader("📤 Upload Markdown")
uploaded_file = st.file_uploader("Upload a `.md` file", type=["md"])
if uploaded_file:
    with st.spinner("Processing file..."):
        process_md(uploaded_file)
    st.success("File processed and added to KG and vector store.")
