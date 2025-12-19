# 1. Use the Full Python Image (Debian-based)
# This includes system libraries that help pip find the correct binary wheels.
FROM python:3.10

# 2. Install Runtime Dependencies
# We only need FFmpeg now (for processing audio). 
# We don't need gcc/pkg-config because we will use binary wheels.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 3. Upgrade pip, setuptools, and wheel
# This is CRITICAL. It ensures the installer can handle the binary files.
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]