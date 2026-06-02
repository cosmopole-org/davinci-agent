# Generic Dockerfile for a Davinci tool creature.
# The deploy harness ships this Dockerfile plus tool_runtime.py and the tool's
# own tool.py as the build context; the node builds the image and runs it as a
# gVisor-sandboxed docker creature that responds to signals.
FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app
COPY . /app

# The node uploads the signal payload to /app/input before start.
RUN mkdir -p /app/input

# TOOL_ID is overridden per tool at run time via the signal payload.
ENV DAVINCI_INPUT_DIR=/app/input
CMD ["python", "tool_runtime.py"]
