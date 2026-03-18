FROM python:3.12-slim

# System dependencies for ripser (C++ build tools)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir \
    numpy \
    scikit-learn \
    ripser \
    persim \
    fastapi \
    uvicorn \
    pydantic

# Copy project files
COPY tda_detect/ ./tda_detect/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "tda_detect.serve:app", "--host", "0.0.0.0", "--port", "8000"]