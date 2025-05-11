import streamlit as st
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

from FREA.store_vectors.vector_store import get_vectorstore, store_file

# --- Setup LLM ---
llm = Ollama(model="llama3")

# --- Get Neo4j Vectorstore ---
vectorstore = get_vectorstore()

# --- Setup retriever ---
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# --- Setup QA chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# --- Streamlit UI ---
st.set_page_config(page_title="📚 MD Chatbot with Ollama + Neo4j", page_icon="🤖")

st.title("📚 Chat with your Markdown files!")
st.write("Ask any question based on the files you placed in the `MD/` folder.")

# --- Upload new file interface ---
uploaded_file = st.file_uploader("Upload a Markdown (.md) file", type=["md"])

if uploaded_file is not None:
    # Save the uploaded file to ./MD folder
    save_path = os.path.join("MD", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Saved {uploaded_file.name} to MD folder!")

    # Process and store it
    store_file(save_path)
    st.success(f"Vectors from {uploaded_file.name} added to Neo4j!")

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display conversation history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input box ---
query = st.chat_input("Ask your question...")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Get response from QA chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain({"query": query})
            answer = result['result']
            st.markdown(answer)

            # Show sources
            sources = result['source_documents']
            if sources:
                st.markdown("#### 📄 Sources:")
                for doc in sources:
                    st.markdown(f"- {doc.metadata.get('source', 'unknown')}")

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
