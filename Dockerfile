# 1. Use an official lightweight Python runtime
FROM python:3.10-slim

# 2. Install system dependencies
# This is the CRITICAL step: We install 'ffmpeg' here so the server has it.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy requirements and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the app code
COPY . .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]