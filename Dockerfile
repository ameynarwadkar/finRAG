FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy all project files
COPY . /app

# Install Python packages using uv
# Install CPU-only torch first to prevent downloading massive CUDA drivers that crash the build
RUN uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --system fastapi uvicorn pydantic qdrant-client sentence-transformers rank-bm25 openai python-dotenv ragas langchain langchain-openai langchain-community langchain-text-splitters datasets pandas python-multipart beautifulsoup4 pypdf scikit-learn

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
