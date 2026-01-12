FROM ghcr.io/astral-sh/uv:0.6.17-debian AS build
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LOCKED=1

COPY --from=fcollonval/unmute-asr:latest /opt/wheels/sphn*.whl /opt/wheels/

COPY . .
RUN UV_LOCKED=0 uv lock -P sphn \ 
    && uv venv && uv pip install /opt/wheels/sphn*.whl \
    && uv run --no-dev --extra-index-url /opt/wheels echo hello

ENV HOSTNAME="0.0.0.0"

HEALTHCHECK --start-period=15s \
    CMD curl --fail http://localhost:80/metrics || exit 1

FROM build AS prod

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y --no-install-recommends libopus0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Running through uvicorn directly to be able to deactive the Websocket per message deflate which is slowing
# down the replies by a few ms.
CMD ["uv", "run", "--no-dev", "uvicorn", "unmute.main_websocket:app", "--host", "0.0.0.0", "--port", "80", "--ws-per-message-deflate=false"]


FROM build AS hot-reloading
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y --no-install-recommends libopus0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

CMD ["uv", "run", "--no-dev", "uvicorn", "unmute.main_websocket:app", "--reload", "--host", "0.0.0.0", "--port", "80", "--ws-per-message-deflate=false"]
