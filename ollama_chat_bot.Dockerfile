FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY hugging_face_models_chatbot.py ./

# Expose the Streamlit port
EXPOSE 8507

# Command to run the Streamlit app
CMD ["streamlit", "run", "hugging_face_models_chatbot.py", "--server.port=8507", "--server.address=0.0.0.0"]