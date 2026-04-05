FROM python:3.11-slim

WORKDIR /app

# System libs needed by numpy/pandas
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Install only what the server needs
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy backend + model files
COPY backend/ ./backend/

# Copy frontend so FastAPI can serve it
COPY frontend/ ./frontend/

EXPOSE 8000
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
