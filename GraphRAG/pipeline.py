from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformersEmbeddings
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG

driver = GraphDatabase.driver(URI, auth=(USER, PASS))

# Build graph
kg_pipeline = SimpleKGPipeline(
  driver=driver,
  llm=OllamaLLM(model="openlama‑model"),
  embeddings=SentenceTransformersEmbeddings(model_name="all-MiniLM-L6-v2")
)
kg_pipeline.ingest_documents(documents)

retriever = VectorRetriever(driver, index_name="doc_index", embedder=kg_pipeline.embeddings)
llm = OllamaLLM(model="openlama‑model", model_params={"temperature":0})
rag = GraphRAG(retriever=retriever, llm=llm)

response = rag.search(query_text="Your question here", retriever_config={"top_k":5})
print(response.answer)
