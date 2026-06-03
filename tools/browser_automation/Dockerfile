# browser_automation tool creature.
# Uses the official Playwright Python image, which ships Chromium/Firefox/WebKit
# plus all OS dependencies preinstalled — so no heavy `playwright install` runs
# at build time and the creature can drive a real headless browser.
# Build context is assembled by scripts/deploy_and_test.py (ships the shared
# tools/_runtime/tool_runtime.py alongside this tool's tool.py + requirements).
FROM mcr.microsoft.com/playwright/python:v1.49.0-noble

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org \
    -r requirements.txt
# Ensure browser binaries are present for this Playwright version.
RUN python -m playwright install chromium

COPY . /app

RUN mkdir -p /app/input
ENV DAVINCI_INPUT_DIR=/app/input PYTHONUNBUFFERED=1

CMD ["python", "tool_runtime.py"]
