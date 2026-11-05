FROM python:3.12-slim

WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/cache
ENV EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

RUN useradd --create-home --uid 10001 embed \
    && mkdir -p /cache \
    && chown embed:embed /cache

COPY requirements.txt server.py ./

RUN pip install --no-cache-dir -r requirements.txt

USER embed

EXPOSE 8080

CMD ["python", "server.py"]
