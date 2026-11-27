FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code (secrets are NOT copied: .env is excluded in .dockerignore).
COPY hotel_agent ./hotel_agent
COPY db ./db
COPY evals/questions.yaml ./evals/questions.yaml
COPY app.py ./

# Run as an unprivileged user; charts are written to /app/outputs/charts.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/outputs/charts /app/evals/results \
    && chown -R app:app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status == 200 else 1)"

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
