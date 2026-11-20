FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLAMA_FARM_DATA_DIR=/data \
    LLAMA_FARM_MODELS_DIR=/data/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl tar util-linux procps libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

RUN chmod +x scripts/*.sh docker/*.sh || true
VOLUME ["/data"]
EXPOSE 8199 8090 8080

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=6 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8199/api/machines', timeout=3)" || exit 1

CMD ["python", "src/app.py"]
