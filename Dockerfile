FROM python:3.12-slim

WORKDIR /app

# Copy source and tests into the image.
COPY . /app

# Run the Davinci runtime model by default.
CMD ["python", "agentic_runtime.py"]
