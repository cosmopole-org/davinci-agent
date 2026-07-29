# vercel_sandbox tool creature.
# Build context is assembled by scripts/deploy_and_test.py (ships the shared
# tools/_runtime/tool_runtime.py alongside this tool's tool.py + requirements).
# The Vercel credentials are baked in as ENV by the deploy harness — they are
# never read from the signal payload.
FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org \
    -r requirements.txt

COPY . /app

RUN mkdir -p /app/input
ENV DAVINCI_INPUT_DIR=/app/input PYTHONUNBUFFERED=1 TOOL_ID=vercel_sandbox

CMD ["python", "tool_runtime.py"]
