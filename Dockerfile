# 1. Use Python 3.10 Slim
FROM python:3.10-slim

# 2. Install SYSTEM dependencies
# Added: pkg-config (to find libraries)
# Added: libav...-dev (development headers for building PyAV)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    pkg-config \
    libsndfile1 \
    libgomp1 \
    git \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Work Directory
WORKDIR /app

# 4. Copy requirements
COPY requirements.txt .

# 5. Install Python dependencies
# Now PyAV can build successfully because it can find the headers
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application code
COPY . .

# 7. Run command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]