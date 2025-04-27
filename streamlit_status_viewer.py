import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="OCR Processing Status", layout="wide")

status_file = 'E:/test/sample/status.csv'

st.title("📄 OCR Processing Status")

placeholder = st.empty()

while True:
    try:
        df = pd.read_csv(status_file)
        placeholder.dataframe(df)
    except Exception as e:
        st.error(f"Error reading status file: {e}")

    time.sleep(3)  # Refresh every 3 seconds
