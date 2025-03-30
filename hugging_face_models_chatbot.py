import streamlit as st
from transformers import pipeline

st.set_page_config(page_title='Ollama Chatbot', layout='wide')

# Load the local Hugging Face model
@st.cache_resource
def load_model():
    return pipeline('text-generation', model='./models/Llama-3.2-3B-Instruct-GGUF')

model = load_model()

st.title('Chat with Ollama - Local Hugging Face Model')
st.markdown('A local chatbot powered by Ollama and Hugging Face models.')

if 'history' not in st.session_state:
    st.session_state.history = []

user_input = st.text_input('You:')

if st.button('Send') and user_input:
    with st.spinner('Generating response...'):
        response = model(user_input, max_length=100, num_return_sequences=1)
        reply = response[0]['generated_text']
        st.session_state.history.append((user_input, reply))

# Display chat history
for user, bot in st.session_state.history:
    st.write(f'**You:** {user}')
    st.write(f'**Ollama:** {bot}')
