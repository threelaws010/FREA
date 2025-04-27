FROM python:3.10-bullseye

RUN apt-get update && apt-get upgrade -y && apt-get install -y libglib2.0-0

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir streamlit pandas

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_status_viewer.py", "--server.port=8501", "--server.headless=true"]
