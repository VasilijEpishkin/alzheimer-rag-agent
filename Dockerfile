FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the application
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

# Install the package in development mode
RUN pip install --no-cache-dir -e .[dev]

# Expose the port the app runs on
EXPOSE 8000

# Run the service
CMD ["evidence-rag", "service"]