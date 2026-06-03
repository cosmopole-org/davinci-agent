# git_tool creature.
# Build context is assembled by scripts/deploy_and_test.py (ships the shared
# tools/_runtime/tool_runtime.py alongside this tool's tool.py + requirements).
FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

# git is the core dependency of this tool; ca-certificates lets it talk HTTPS.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org \
    -r requirements.txt

COPY . /app

RUN mkdir -p /app/input /app/workspace
ENV DAVINCI_INPUT_DIR=/app/input PYTHONUNBUFFERED=1

CMD ["python", "tool_runtime.py"]
