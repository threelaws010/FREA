import os, streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pathlib import Path

# --------- CONFIG ----------
FAISS_DIR = Path(__file__).resolve().parent / "faiss_index" # folder with index.faiss + index.pkl
EMBED_MODEL = "text-embedding-3-large"   # must match what built the index
CHAT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# ---------------------------

# Env for LM Studio OpenAI-compatible server
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:1234/v1")
os.environ.setdefault("OPENAI_API_KEY", "lm-studio")

st.set_page_config(page_title="FAISS + LM Studio RAG", page_icon="🧠")
st.title("🧠 RAG chat (FAISS + LM Studio)")

# Create embeddings object that matches the FAISS index
emb = OpenAIEmbeddings(model=EMBED_MODEL)

# Load existing FAISS index
# NOTE: allow_dangerous_deserialization=True is needed for legacy pickles
vs = FAISS.load_local(FAISS_DIR, emb, allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_kwargs={"k": 4})

# Chat model (LM Studio)
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)

# Simple chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# UI
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

user_q = st.chat_input("Ask anything; answers will use your FAISS context…")
if user_q:
    # Retrieve context
    docs = retriever.get_relevant_documents(user_q)
    context = "\n\n".join(f"[{i+1}] {d.metadata.get('source','?')}\n{d.page_content}"
                          for i, d in enumerate(docs))
    prompt = f"""You are a precise assistant. Use the context to answer.
Context:
{context}

Question: {user_q}
Answer concisely with citations like [1], [2] when used.
"""

    st.session_state.messages.append(("user", user_q))
    with st.chat_message("assistant"):
        resp = llm.invoke(prompt).content
        st.markdown(resp)
        st.session_state.messages.append(("assistant", resp))
