FROM python:3.11-slim

ARG LIBKEYFINDER_REF=2.2.8
ARG KEYFINDER_CLI_REF=v1.2.0
ARG SKEY_REF=main

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential ca-certificates cmake ffmpeg git libavcodec-dev \
       libavformat-dev libavutil-dev libfftw3-dev libsamplerate0-dev \
       libsndfile1 libswresample-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# libKeyFinder and its CLI are source-built because Debian does not package the
# library and the public Python binding exposes only a single result. Upstream's
# old Catch tests do not compile on current glibc, so the release library target
# is built without that test subdirectory.
RUN git clone --depth 1 --branch "$LIBKEYFINDER_REF" https://github.com/mixxxdj/libkeyfinder.git /tmp/libKeyFinder \
    && cmake -S /tmp/libKeyFinder -B /tmp/libKeyFinder/build -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build /tmp/libKeyFinder/build --parallel \
    && cmake --install /tmp/libKeyFinder/build \
    && git clone --depth 1 --branch "$KEYFINDER_CLI_REF" https://github.com/evanpurkhiser/keyfinder-cli.git /tmp/keyfinder-cli \
    && cmake -S /tmp/keyfinder-cli -B /tmp/keyfinder-cli/build -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build /tmp/keyfinder-cli/build --parallel \
    && cmake --install /tmp/keyfinder-cli/build \
    && ldconfig \
    && rm -rf /tmp/libKeyFinder /tmp/keyfinder-cli

WORKDIR /app
COPY pyproject.toml README.md requirements-analysis.txt ./
COPY src ./src
COPY scripts ./scripts

RUN pip install --no-cache-dir "numpy>=1.26,<2" "Cython<3" "setuptools<70" \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir --no-build-isolation -r requirements-analysis.txt

# S-KEY requires NumPy 2.x and PyTorch 2.7, so keep it isolated from madmom.
RUN python -m venv /opt/skey-venv \
    && /opt/skey-venv/bin/pip install --no-cache-dir "git+https://github.com/deezer/skey.git@${SKEY_REF}"

ENV SKEY_PYTHON=/opt/skey-venv/bin/python
ENV SKEY_RUNNER=/app/scripts/skey_predict.py

ENTRYPOINT ["autolabel"]

