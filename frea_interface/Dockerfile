FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# ✅ Install all Python packages including langchain-neo4j
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Ensure Neo4j is ready before starting the app
RUN chmod +x /app/check_neo4j_ready.sh

CMD ["sh", "-c", "./check_neo4j_ready.sh && streamlit run app/main.py"]
