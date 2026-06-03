# vector_search tool creature.
# Build context is assembled by scripts/deploy_and_test.py (ships the shared
# tools/_runtime/tool_runtime.py alongside this tool's tool.py + requirements).
FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org \
    -r requirements.txt

COPY . /app

# Persistent collection store + signal upload dir.
RUN mkdir -p /app/input /app/data
ENV DAVINCI_INPUT_DIR=/app/input \
    VECTOR_DATA_DIR=/app/data \
    PYTHONUNBUFFERED=1

CMD ["python", "tool_runtime.py"]
