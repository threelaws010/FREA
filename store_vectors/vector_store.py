import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_neo4j import Neo4jVector

# --- Neo4j Connection Settings ---
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "frea_password"  # 🔥 IMPORTANT: Change this to your real password

# --- Embeddings and Vectorstore setup ---
embedding_model = OllamaEmbeddings(model="llama3")

vectorstore = Neo4jVector(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    embedding=embedding_model,
    database="neo4j",
)

def store_file(file_path):
    """Store a new .md file into the Neo4j vector database."""
    if not file_path.endswith(".md"):
        print(f"⚠️ Skipping non-Markdown file: {file_path}")
        return

    print(f"📄 Loading file: {file_path}")

    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    print(f"🔵 Inserting {len(docs)} chunks into Neo4j...")
    vectorstore.add_documents(docs)
    print(f"✅ File {file_path} inserted into Neo4j!")

def get_vectorstore():
    """Return the Neo4j Vectorstore object."""
    return vectorstore
