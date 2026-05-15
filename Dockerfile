FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY env.py agent.py train.py evaluate.py app.py monitoring.py rollback.py ./
COPY params.yaml ./

# Create runtime directories (volumes can be mounted over these)
RUN mkdir -p models logs plots mlruns

# Streamlit listens on 8501; MLflow UI on 5000
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
  || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
