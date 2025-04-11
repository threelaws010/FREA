import streamlit as st
import os
import json

VECTOR_DIR = "./vectorized_files"

st.set_page_config(page_title="Vectorization Dashboard", layout="wide")
st.title("📊 Vectorization Control Panel")

# Ensure directory exists
if not os.path.exists(VECTOR_DIR):
    os.makedirs(VECTOR_DIR)

# List files
vectorized_files = [f for f in os.listdir(VECTOR_DIR) if f.endswith(".json")]

st.metric("✅ Files Vectorized", len(vectorized_files))

if vectorized_files:
    with st.expander("📁 View Vectorized Files"):
        for f in vectorized_files:
            st.write(f)

    if st.button("🗑️ Clear Vector Cache"):
        for f in vectorized_files:
            os.remove(os.path.join(VECTOR_DIR, f))
        st.experimental_rerun()
else:
    st.info("No files vectorized yet.")
