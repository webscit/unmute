import asyncio
import json
import logging
from functools import cache

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from unmute import metrics as mt
from unmute.exceptions import (
    MissingServiceAtCapacity,
    MissingServiceTimeout,
    WebSocketClosedError,
    make_ora_error,
)
from unmute.kyutai_constants import MAX_VOICE_FILE_SIZE_MB
from unmute.services.health_service import check_session_admission, get_health
from unmute.services.websocket_session_manager import WebSocketSessionManager
from unmute.timer import Stopwatch
from unmute.tts.voice_cloning import clone_voice
from unmute.tts.voice_donation import (
    VoiceDonationSubmission,
    generate_verification,
    submit_voice_donation,
)
from unmute.tts.voices import VoiceList
from unmute.unmute_handler import UnmuteHandler
from unmute.websocket_auth import (
    REALTIME_SUBPROTOCOL,
    perform_handshake_validation,
    reject_connection_with_error,
)

app = FastAPI()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# We prefer to scale this by running more instances of the server than having a single
# server handle more. This is to avoid the GIL.
MAX_CLIENTS = 4
SEMAPHORE = asyncio.Semaphore(MAX_CLIENTS)

Instrumentator().instrument(app).expose(app)
PROFILE_ACTIVE = False
_last_profile = None
_current_profile = None

# Allow CORS for local development
CORS_ALLOW_ORIGINS = ["http://localhost", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "You've reached the Unmute backend server."}


if PROFILE_ACTIVE:

    @app.get("/profile")
    def profile():
        if _last_profile is None:
            return HTMLResponse("<body>No last profiler saved</body>")
        else:
            return HTMLResponse(_last_profile.output_html())  # type: ignore


@app.get("/v1/health")
async def health_endpoint():
    health = await get_health()
    mt.HEALTH_OK.observe(health.ok)
    return health


@app.get("/healthz")
async def liveness_probe():
    """
    Liveness probe for Kubernetes.
    Returns 200 if the application is alive and running.
    """
    # Basic liveness check - just ensure the app is responding
    return {"status": "alive"}


@app.get("/readyz")
async def readiness_probe():
    """
    Readiness probe for Kubernetes.
    Returns 200 if the system is ready to accept new sessions.
    Returns 503 if the system is overloaded or dependencies are unavailable.
    """
    can_admit, reason = await check_session_admission()

    if not can_admit:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"ready": False, "reason": reason},
        )

    return {"ready": True, "reason": ""}


@app.get("/v1/voices")
@cache
def voices():
    voice_list = VoiceList()
    # Note that `voice.good` is bool | None, here we really take only True values.
    good_voices = [
        voice.model_dump(exclude={"comment"})
        for voice in voice_list.voices
        if voice.good
    ]
    return good_voices


class LimitUploadSizeForPath(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_upload_size: int, path: str) -> None:
        super().__init__(app)
        self.max_upload_size = max_upload_size
        self.path = path

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method == "POST" and request.url.path == self.path:
            if "content-length" not in request.headers:
                return Response(status_code=status.HTTP_411_LENGTH_REQUIRED)

            content_length = int(request.headers["content-length"])
            if content_length > self.max_upload_size:
                return Response(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        return await call_next(request)


app.add_middleware(
    LimitUploadSizeForPath,
    max_upload_size=MAX_VOICE_FILE_SIZE_MB * 1024 * 1024,
    path="/v1/voices",
)


@app.post("/v1/voices")
async def post_voices(file: UploadFile):
    """Upload a voice list file.

    Make sure the maximum file size is configured in uvicorn.
    """
    name = clone_voice(file.file.read())
    return {"name": name}


@app.get("/v1/voice-donation")
async def get_voice_donation():
    """Initiate a voice donation by asking for a verification text."""
    verification = generate_verification()
    return verification


app.add_middleware(
    LimitUploadSizeForPath,
    max_upload_size=MAX_VOICE_FILE_SIZE_MB * 1024 * 1024,
    path="/v1/voice-donation",
)


@app.post("/v1/voice-donation")
async def post_voice_donation(
    file: UploadFile = File(...),  # noqa: B008
    metadata: str = Form(...),
):
    """Finish a voice donation."""
    file_bytes = file.file.read()

    try:
        metadata_parsed = VoiceDonationSubmission(**json.loads(metadata))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid submission: {e.errors()}",
        ) from e

    try:
        submit_voice_donation(metadata_parsed, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {}


@app.websocket("/v1/realtime")
async def websocket_route(websocket: WebSocket):
    global _last_profile, _current_profile

    # Perform handshake validation before accepting
    valid, error = await perform_handshake_validation(websocket)
    if not valid and error is not None:
        logger.info(f"WebSocket handshake failed: {error.error.message}")
        await reject_connection_with_error(websocket, error)
        return

    # Check if system can admit new session based on health probes
    can_admit, reason = await check_session_admission()
    if not can_admit:
        logger.warning(f"Session admission denied: {reason}")
        error = make_ora_error(
            code="server_overloaded",
            message=f"Server is overloaded: {reason}. Please try again later.",
        )
        await reject_connection_with_error(websocket, error)
        return

    mt.SESSIONS.inc()
    mt.ACTIVE_SESSIONS.inc()
    session_watch = Stopwatch()
    if PROFILE_ACTIVE and _current_profile is None:
        from pyinstrument import Profiler

        logger.info("Profiler started.")
        _current_profile = Profiler(interval=0.0001, async_mode="disabled")
        import inspect

        frame = inspect.currentframe()
        while frame is not None and frame.f_back:
            frame = frame.f_back
        _current_profile.start(caller_frame=frame)

    async with SEMAPHORE:
        try:
            # The `subprotocol` argument is important because the client specifies what
            # protocol(s) it supports and OpenAI uses "realtime" as the value. If we
            # don't set this, the client will think this is not the right endpoint and
            # will not connect.
            # await websocket.accept(subprotocol=REALTIME_SUBPROTOCOL)
            await websocket.accept()

            handler = UnmuteHandler()

            async with handler:
                await handler.start_up()
                session_manager = WebSocketSessionManager(websocket, handler)
                await session_manager.run()

        except Exception as exc:
            await _report_websocket_exception(websocket, exc)
        finally:
            if _current_profile is not None:
                _current_profile.stop()
                logger.info("Profiler saved.")
                _last_profile = _current_profile
                _current_profile = None

            mt.ACTIVE_SESSIONS.dec()
            mt.SESSION_DURATION.observe(session_watch.time())


async def _report_websocket_exception(websocket: WebSocket, exc: Exception):
    if isinstance(exc, ExceptionGroup):
        exceptions = exc.exceptions
    else:
        exceptions = [exc]

    error_message = None

    for exc in exceptions:
        if isinstance(exc, (MissingServiceAtCapacity)):
            mt.FATAL_SERVICE_MISSES.inc()
            error_message = (
                f"Too many people are connected to service '{exc.service}'. "
                "Please try again later."
            )
        elif isinstance(exc, MissingServiceTimeout):
            mt.FATAL_SERVICE_MISSES.inc()
            error_message = (
                f"Service '{exc.service}' timed out. Please try again later."
            )
        elif isinstance(exc, WebSocketClosedError):
            logger.debug("Websocket was closed.")
        else:
            logger.exception("Unexpected error: %r", exc)
            mt.HARD_ERRORS.inc()
            error_message = "Internal server error :( Complain to Kyutai"

    if error_message is not None:
        mt.FORCE_DISCONNECTS.inc()

        try:
            await websocket.send_text(
                make_ora_error(type="fatal", message=error_message).model_dump_json()
            )
        except WebSocketDisconnect:
            logger.warning("Failed to send error message due to disconnect.")

        try:
            await websocket.close(
                code=status.WS_1011_INTERNAL_ERROR,
                reason=error_message,
            )
        except RuntimeError:
            logger.warning("Socket already closed.")


def _cors_headers_for_error(request: Request):
    origin = request.headers.get("origin")
    allow_origin = origin if origin in CORS_ALLOW_ORIGINS else None
    headers = {"Access-Control-Allow-Credentials": "true"}
    if allow_origin:
        headers["Access-Control-Allow-Origin"] = allow_origin

    return headers


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # We need this so that CORS header are added even when the route raises an
    # exception. Otherwise you get a confusing CORS error even if the issue is totally
    # unrelated.
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=_cors_headers_for_error(request),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # We need this so that CORS header are added even when the route raises an
    # exception. Otherwise you get a confusing CORS error even if the issue is totally
    # unrelated.
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers=_cors_headers_for_error(request),
    )


if __name__ == "__main__":
    import sys

    print(f"Run this via:\nfastapi dev {sys.argv[0]}")
    exit(1)
