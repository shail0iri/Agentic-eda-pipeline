# Slim base image — smaller footprint, still has everything pip needs to build wheels
FROM python:3.11-slim

WORKDIR /app

# build-essential is needed because some dependencies (e.g. parts of the
# sentence-transformers/torch stack) compile native extensions on install
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first and install BEFORE copying the rest of the code.
# Docker caches layers — this means code changes won't force a full
# re-install of every dependency, only a rebuild when requirements.txt itself changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY . .

# Ensure the folder for uploaded CSVs exists inside the container
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
