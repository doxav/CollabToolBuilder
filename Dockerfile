# Description: Dockerfile for building a Docker image with Python 3.10 inside Ollama image.

FROM ollama/ollama

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install wget and build dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    bash \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    libffi-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Download and extract Python 3.10
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz \
    && tar -xzf Python-3.10.0.tgz \
    && cd Python-3.10.0 \
    && ./configure --enable-optimizations \
    && make altinstall

# Reset DEBIAN_FRONTEND to default
ENV DEBIAN_FRONTEND=

COPY requirements.txt /app/requirements.txt

# Assuming Python 3.10 is installed in /usr/local/bin, update $PATH just in case
ENV PATH="/usr/local/bin:${PATH}"

# Create virtual environment and install requirements
RUN python3.10 -m venv /app/.venv && \
    /app/.venv/bin/python -m pip install --upgrade pip && \
    /app/.venv/bin/python -m pip install -r /app/requirements.txt && \
    /app/.venv/bin/python -m pip install aider-chat

# Make the .venv directory writable
RUN chmod -R 777 /app/.venv

# Use bash shell for subsequent commands
SHELL ["/bin/bash", "-c"]
