# 1. Use Python 3.10 Slim
FROM python:3.10-slim

# 2. Install SYSTEM dependencies (Crucial Step)
# ffmpeg: For processing audio files
# gcc: For compiling Python C-extensions
# libsndfile1: For reading audio files
# libgomp1: Required by faster-whisper optimizations
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    libsndfile1 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Set work directory
WORKDIR /app

# 4. Copy requirements
COPY requirements.txt .

# 5. Install Python dependencies
# Now this will succeed because gcc and ffmpeg are present
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application code
COPY . .

# 7. Run command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]