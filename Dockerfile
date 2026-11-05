FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 10001 masskb

COPY pyproject.toml README.md LICENSE ./
COPY masskb ./masskb

RUN pip install --no-cache-dir ".[api]"

USER masskb

EXPOSE 8080

CMD ["masskb", "serve", "--host", "0.0.0.0", "--port", "8080"]
