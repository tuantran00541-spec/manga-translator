FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8100 \
    GATEWAY_DB=/data/gateway.sqlite

WORKDIR /srv

RUN pip install --only-binary=:all: fastapi==0.115.0 "uvicorn[standard]==0.30.6" requests==2.32.3 pydantic==2.9.2 \
    && useradd --create-home --uid 10001 gateway \
    && mkdir /data && chown gateway:gateway /data

COPY gateway/ ./gateway/

USER gateway
VOLUME /data
EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=4)"

CMD ["python", "-m", "gateway"]
