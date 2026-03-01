FROM python:3.15.0a5-alpine3.23

WORKDIR /swiftcfd

COPY ./requirements.txt .

RUN apk add --no-cache \
    build-base \
    zlib-dev \
    libjpeg-turbo-dev \
    libpng-dev \
    freetype-dev \
    gfortran

ENV PETSC_CONFIGURE_OPTIONS="--download-fblaslapack=1"

RUN python3 -m venv /dev/venv
ENV PATH="/dev/venv/bin:$PATH"

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

RUN rm ./requirements.txt

ENTRYPOINT ["python3"]