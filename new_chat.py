import os
import tempfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from py2neo import Graph, Node, Relationship
from pyvis.network import Network
import streamlit.components.v1 as components

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

st.set_page_config(page_title="🦙 LLaMA 3 + FAISS Chatbot")
st.title("🦙 LLaMA 3 + FAISS Chatbot")

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Set up prompt
custom_prompt_template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:
"""
prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

llm = Ollama(model=LLM_MODEL)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Save to Neo4j
def save_to_neo4j(history):
    try:
        graph = Graph(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASS))
        tx = graph.begin()
        for i, exchange in enumerate(history[-len(history)//2:]):  # Only second half
            user_node = Node("User", name=f"User_{i}", query=exchange["query"])
            bot_node = Node("Bot", name=f"Bot_{i}", result=exchange["result"])
            rel = Relationship(user_node, "ASKED", bot_node)
            tx.create(user_node)
            tx.create(bot_node)
            tx.create(rel)
        graph.commit()

        net = Network(height="275px", width="100%", directed=True)
        for i, exchange in enumerate(history[-len(history)//2:]):
            net.add_node(f"User_{i}", label=exchange["query"], color="#00ccff")
            net.add_node(f"Bot_{i}", label=exchange["result"][:40] + ("..." if len(exchange["result"]) > 40 else ""), color="#ffaa00")
            net.add_edge(f"User_{i}", f"Bot_{i}", label="asks")

        tmp_path = tempfile.mktemp(suffix=".html")
        net.show(tmp_path)
        return tmp_path
    except Exception as e:
        st.error(f"Neo4j error: {e}")
        return None

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for chat history and save
with st.sidebar:
    st.subheader("Chat History")
    for msg in st.session_state.chat_history:
        st.text(f"You: {msg['query']}")
        st.text(f"Bot: {msg['result'][:100]}{'...' if len(msg['result']) > 100 else ''}")

    if st.button("Save Chat to Neo4j"):
        path = save_to_neo4j(st.session_state.chat_history)
        if path:
            st.session_state.graph_path = path

# Show graph at the top if available
if "graph_path" in st.session_state:
    components.html(open(st.session_state.graph_path, "r", encoding="utf-8").read(), height=275)

# Chat interface
user_question = st.text_input("Ask a question about your documents:")
if user_question:
    result = qa_chain.run(user_question)
    st.session_state.chat_history.append({"query": user_question, "result": result})
    st.write("**Answer:**", result)
