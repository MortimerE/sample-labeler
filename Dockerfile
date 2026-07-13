FROM python:3.11-slim

ARG LIBKEYFINDER_REF=2.2.8
ARG KEYFINDER_CLI_REF=v1.2.0
ARG SKEY_REF=918b83d273568d5041569bb8068843d19a335726
ARG SKEY_VERSION=918b83d27356
ARG BEAT_THIS_VERSION=1.1.0
ARG TEMPOCNN_GRAPH_URL=https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k16-3.pb
ARG TEMPOCNN_GRAPH_SHA256=21c328332a221695dd6e8572728c617373064df882e8f81da6d88dc3a821e3b3
ARG TEMPOCNN_META_URL=https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k16-3.json
ARG TEMPOCNN_META_SHA256=c0c62a52aa4a05f197208133906775c1e87077a520cdec53598b67ea9d625998
ARG BEAT_THIS_CKPT_URL=https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt
ARG BEAT_THIS_CKPT_SHA256=8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential ca-certificates cmake curl ffmpeg git libavcodec-dev \
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
COPY train ./train
COPY artifacts ./local-artifacts

RUN mkdir -p /app/artifacts /opt/torch-cache \
    && curl -fsSL "$TEMPOCNN_GRAPH_URL" -o /app/artifacts/deeptemp-k16-3.pb \
    && echo "$TEMPOCNN_GRAPH_SHA256  /app/artifacts/deeptemp-k16-3.pb" | sha256sum -c - \
    && curl -fsSL "$TEMPOCNN_META_URL" -o /app/artifacts/deeptemp-k16-3.json \
    && echo "$TEMPOCNN_META_SHA256  /app/artifacts/deeptemp-k16-3.json" | sha256sum -c - \
    && curl -fsSL "$BEAT_THIS_CKPT_URL" -o /app/artifacts/beat_this-final0.ckpt \
    && echo "$BEAT_THIS_CKPT_SHA256  /app/artifacts/beat_this-final0.ckpt" | sha256sum -c - \
    && sha256sum /app/artifacts/* > /app/artifacts/SHA256SUMS

RUN if [ -f /app/local-artifacts/fusion_params.npz ]; then \
      cp /app/local-artifacts/fusion_params.npz /app/artifacts/fusion_params.npz; \
      cp /app/local-artifacts/fusion_params.json /app/artifacts/fusion_params.json; \
      sha256sum /app/artifacts/fusion_params.npz /app/artifacts/fusion_params.json \
        >> /app/artifacts/SHA256SUMS; \
    fi \
    && rm -rf /app/local-artifacts

RUN pip install --no-cache-dir "numpy>=1.26,<2" "Cython<3" "setuptools<70" \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir --no-build-isolation -r requirements-analysis.txt

# ML models are isolated in a dedicated venv (torch stack for S-KEY + Beat This).
RUN python -m venv /opt/ml-venv \
    && /opt/ml-venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "torch==2.7.1" "torchaudio==2.7.1" \
    && /opt/ml-venv/bin/pip install --no-cache-dir "git+https://github.com/deezer/skey.git@${SKEY_REF}" \
    && /opt/ml-venv/bin/pip install --no-cache-dir "beat-this==${BEAT_THIS_VERSION}" \
    && /opt/ml-venv/bin/pip install --no-cache-dir -e .

ENV SKEY_PYTHON=/opt/ml-venv/bin/python
ENV SKEY_RUNNER=/app/scripts/skey_predict.py
ENV SKEY_VERSION=${SKEY_VERSION}
ENV SKEY_MIN_SECONDS=3.75
ENV BEAT_THIS_PYTHON=/opt/ml-venv/bin/python
ENV BEAT_THIS_RUNNER=/app/scripts/beat_this_predict.py
ENV BEAT_THIS_VERSION=${BEAT_THIS_VERSION}
ENV BEAT_THIS_CHECKPOINT=/app/artifacts/beat_this-final0.ckpt
ENV TORCH_HOME=/opt/torch-cache
ENV TEMPOCNN_GRAPH=/app/artifacts/deeptemp-k16-3.pb
ENV TEMPOCNN_METADATA=/app/artifacts/deeptemp-k16-3.json
ENV MODEL_SHA256SUMS=/app/artifacts/SHA256SUMS

ENTRYPOINT ["autolabel"]
