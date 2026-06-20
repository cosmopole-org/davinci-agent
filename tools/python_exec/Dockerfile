# python_exec tool creature.
# Build context is assembled by scripts/deploy_and_test.py, which ships the
# shared tools/_runtime/tool_runtime.py alongside this tool's tool.py +
# requirements.txt. The runtime reads /app/input/task.json and dispatches to
# tool.invoke(), emitting a single TOOL_RESPONSE line.
FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org \
    -r requirements.txt
# Install Chromium + its OS dependencies so executed code can render JS-heavy
# pages via Playwright when a static fetch is insufficient.
RUN python -m playwright install --with-deps chromium

COPY . /app

# The node uploads the signal payload to /app/input before container start.
RUN mkdir -p /app/input
ENV DAVINCI_INPUT_DIR=/app/input PYTHONUNBUFFERED=1

CMD ["python", "tool_runtime.py"]
