from langchain_community.llms import Ollama
from langchain_neo4j.vectorstores import Neo4jVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredMarkdownLoader
from neo4j import GraphDatabase
import tempfile
import re

import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URL = os.getenv("NEO4J_URL", "bolt://host.containers.internal:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "frea")

# Load local LLaMA 3 via Ollama
llm = Ollama(model="llama3")

# HuggingFace embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Neo4jVector(
    url=NEO4J_URL,
    username=NEO4J_USER ,
    password=NEO4J_PASSWORD,
    index_name="frea",
    embedding=VECTOR_INDEX_NAME ,
)

# LangChain RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Neo4j driver for KG
graph_driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))


# Function to run chat query
def ask_question(query: str) -> str:
    return qa_chain.run(query)


# Extract simple capitalized entity terms (improve later with NER)
def extract_entities(text: str):
    return list(set(re.findall(r"\b[A-Z][a-zA-Z]+\b", text)))


# Add entity relationships into Neo4j
def insert_kg_triples(text: str):
    entities = extract_entities(text)
    with graph_driver.session() as session:
        for i in range(len(entities) - 1):
            session.run(
                """
                MERGE (a:Entity {name: $a})
                MERGE (b:Entity {name: $b})
                MERGE (a)-[:RELATED_TO]->(b)
                """,
                a=entities[i],
                b=entities[i + 1]
            )


# Parse and index uploaded Markdown file
def process_markdown_file(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    loader = UnstructuredMarkdownLoader(tmp_path)
    docs = loader.load()

    # Add to vector DB
    vectorstore.add_documents(docs)

    # Add to knowledge graph
    for doc in docs:
        insert_kg_triples(doc.page_content)

    return docs


# Generate a list of triples for graph visualization
def get_graph_triples():
    triples = []
    with graph_driver.session() as session:
        results = session.run(
            """
            MATCH (a)-[r]->(b)
            RETURN a.name AS source, b.name AS target, type(r) AS relation
            """
        )
        for record in results:
            triples.append((record["source"], record["relation"], record["target"]))
    return triples
