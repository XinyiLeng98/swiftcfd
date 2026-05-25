FROM python:3.10-slim

WORKDIR /swiftcfd

COPY ./requirements.txt .

# Install system dependencies (Debian-based)
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    python3-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    gfortran \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

ENV PETSC_CONFIGURE_OPTIONS="--download-fblaslapack=1"

RUN python3 -m venv /dev/venv
ENV PATH="/dev/venv/bin:$PATH"

RUN pip install --upgrade pip

# numpy and setuptools must be installed before petsc, which compiles from source
RUN pip install numpy setuptools

RUN pip install -r requirements.txt

RUN rm ./requirements.txt