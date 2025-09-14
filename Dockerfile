# /home/frea/FREA/Dockerfile
FROM python:3.10-slim

# Useful CLI tools for debugging connectivity from inside the container
RUN apt-get update && apt-get install -y curl netcat-openbsd && rm -rf /var/lib/apt/lists/*

# Workdir where your code will live (compose mounts /home/frea/FREA -> /app)
WORKDIR /app

# If you have requirements.txt, install it first for better caching.
# The "|| true" prevents the build from failing if the file is empty/minimal.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy the rest of your code into the image (compose also mounts it over this).
COPY . /app

# ---- Option B: create the shim during build (no escaping headaches) ----
RUN mkdir -p /app/store_vectors && cat > /app/store_vectors/anythingllm_read_embed_model.py <<'PY'
import os

def read_embed_model_from_api(base: str, key: str, slug: str):
    # Return an env override if set; caller can decide how to fallback.
    return os.getenv("ANYLLM_EMBED_MODEL", None)
PY
# -----------------------------------------------------------------------

# Default to an interactive shell (your compose override can replace this with
# `["tail","-f","/dev/null"]` to keep it alive, or with your idle runner).
CMD ["bash"]
