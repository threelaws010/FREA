import tempfile
from pyvis.network import Network
import streamlit as st

# Save an uploaded Streamlit file to disk and return path
def save_uploaded_file(uploaded_file, suffix=".md") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


# Visualize a knowledge graph using PyVis and display it in Streamlit
def display_knowledge_graph(triples: list, height="400px"):
    net = Network(height=height, width="100%", notebook=False, directed=True)

    # Add nodes and edges
    added_nodes = set()
    for source, relation, target in triples:
        if source not in added_nodes:
            net.add_node(source, label=source)
            added_nodes.add(source)
        if target not in added_nodes:
            net.add_node(target, label=target)
            added_nodes.add(target)
        net.add_edge(source, target, label=relation)

    # Generate temporary HTML file and read for embedding
    path = tempfile.mktemp(suffix=".html")
    net.save_graph(path)
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=400, scrolling=True)


# Simple utility: Truncate text safely for display
def truncate_text(text, max_chars=500):
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text
