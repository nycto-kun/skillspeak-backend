# 1. Use Python 3.10 Slim
FROM python:3.10-slim

# 2. Install Runtime Dependencies
# We REMOVED 'pkg-config' and 'libav...-dev' because they force a broken source build.
# We kept 'ffmpeg' and 'gcc' which are needed for running the app.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    git \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 3. Upgrade pip & Install Dependencies
# CRITICAL: Upgrading pip helps it find the pre-compiled 'binary wheels'
# so it doesn't try (and fail) to build from source.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]