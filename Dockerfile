FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# Native/model packages must be pinned to versions whose APIs match backends.py.
# Kept separate so image builds fail visibly when a platform wheel disappears.
ARG ANALYSIS_REQUIREMENTS=requirements-analysis.txt
COPY requirements-analysis.txt ./
RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir -r "$ANALYSIS_REQUIREMENTS"

ARG LIBKEYFINDER_WHEEL
ARG SKEY_WHEEL
COPY artifacts ./artifacts
RUN test -n "$LIBKEYFINDER_WHEEL" \
    && test -n "$SKEY_WHEEL" \
    && pip install --no-cache-dir "$LIBKEYFINDER_WHEEL" "$SKEY_WHEEL"

ENTRYPOINT ["autolabel"]
