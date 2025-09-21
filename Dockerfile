# ===== Stage 1: Build dependencies (optional for layering) =====
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# If you already have a requirements.txt, keep COPY line; otherwise this block creates one.
# You can delete this echo section if you provide your own requirements.txt.
RUN echo "\
fastapi==0.111.0\n\
uvicorn[standard]==0.30.0\n\
pymongo==4.7.2\n\
python-dotenv==1.0.1\n\
openai==1.35.7\n\
tiktoken==0.7.0\n\
numpy==1.26.4\n\
" > requirements.txt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===== Final Stage =====
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed packages from base layer
COPY --from=base /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application code
COPY src ./src
# If you have a data folder or .env template uncomment below:
# COPY data ./data

# Non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# Environment variable placeholders (override at runtime)
COPY .env .env

# Start FastAPI with reload disabled (enable reload only for dev bind-mounted code)
CMD ["uvicorn", "src.app.searcher_api:app", "--host", "0.0.0.0", "--port", "8000"]