from py2neo import Graph
from dotenv import load_dotenv
import os

load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "llama3")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR", "docs")
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS","Ka1smbPooh")

graph = Graph("bolt://localhost:7687", auth=(NEO4J_USER, NEO4J_PASS))
print(graph.run("RETURN 1").data())
