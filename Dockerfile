# Davinci agent creature image.
# Built and run by a Caspar node as a gVisor-sandboxed `docker` creature.
# Uses the ECR mirror of the official python image to avoid Docker Hub
# anonymous-pull rate limits during node-side builds.
FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

# pycryptodome lets the creature sign Caspar action requests (RSA-PSS), so it
# can signal sibling tool creatures through the node's signalling API.
RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org \
    pycryptodome

# Copy source (davinci package, orchestrator, runtime model, tools).
COPY . /app

# The node uploads runtime inputs to /app/input before starting the container;
# the directory must exist for the upload (and container start) to succeed.
RUN mkdir -p /app/input

ENV DAVINCI_INPUT_DIR=/app/input \
    PYTHONUNBUFFERED=1

# Default: run the Caspar docker-creature entrypoint (reads a task/signal from
# /app/input, runs the agent loop, emits DAVINCI_RESULT). Override CMD to run
# `python agentic_runtime.py` for a static capability report.
CMD ["python", "-m", "davinci.caspar_runtime"]
