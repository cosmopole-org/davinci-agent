# sandbox_pc tool creature.
# Build context is assembled by scripts/deploy_and_test.py (ships the shared
# tools/_runtime/tool_runtime.py alongside this tool's tool.py).
FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN if [ -s /app/requirements.txt ]; then \
      pip install --no-cache-dir \
        --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org \
        -r requirements.txt ; \
    fi

COPY . /app

RUN mkdir -p /app/input
ENV DAVINCI_INPUT_DIR=/app/input PYTHONUNBUFFERED=1

CMD ["python", "tool_runtime.py"]
