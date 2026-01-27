import asyncio
import base64
import json
import logging
from functools import cache, partial
from typing import Annotated

import numpy as np
import requests
import sphn
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
from fastapi.websockets import WebSocketState
from fastrtc import AdditionalOutputs, CloseStream, audio_to_float32
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, computed_field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

import unmute.openai_realtime_api_events as ora
from unmute import metrics as mt
from unmute.exceptions import (
    MissingServiceAtCapacity,
    MissingServiceTimeout,
    WebSocketClosedError,
    make_ora_error,
)
from unmute.kyutai_constants import (
    KYUTAI_LLM_API_KEY,
    LLM_SERVER,
    MAX_VOICE_FILE_SIZE_MB,
    SAMPLE_RATE,
    STT_SERVER,
    TTS_SERVER,
    VOICE_CLONING_SERVER,
)
from unmute.service_discovery import async_ttl_cached
from unmute.timer import Stopwatch
from unmute.tracing import (
    WEBSOCKET_SEND_DURATION,
    end_trace,
    start_trace,
    trace_queue_operation,
    trace_span,
)
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
    NegotiatedSession,
    create_extension_error,
    perform_handshake_validation,
    reject_connection_with_error,
    validate_extensions,
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

ClientEventAdapter = TypeAdapter(
    Annotated[ora.ClientEvent, Field(discriminator="type")]
)

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


def _ws_to_http(ws_url: str) -> str:
    """Convert a WebSocket URL to an HTTP URL."""
    return ws_url.replace("ws://", "http://").replace("wss://", "https://")


def _check_server_status(server_url: str, headers: dict | None = None) -> bool:
    """Check if the server is up by sending a GET request."""
    try:
        response = requests.get(
            server_url,
            timeout=2,
            headers=headers or {},
        )
        logger.info(f"Response from {server_url}: {response}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.info(f"Couldn't connect to {server_url}: {e}")
        return False


async def debug_running_tasks():
    while True:
        logger.debug(f"Running tasks: {len(asyncio.all_tasks())}")
        for task in asyncio.all_tasks():
            logger.debug(f"  Task: {task.get_name()} - {task.get_coro()}")
        await asyncio.sleep(5)


class HealthStatus(BaseModel):
    tts_up: bool
    stt_up: bool
    llm_up: bool
    voice_cloning_up: bool

    @computed_field
    @property
    def ok(self) -> bool:
        # Note that voice cloning is not required for the server to be healthy.
        return self.tts_up and self.stt_up and self.llm_up


@partial(async_ttl_cached, ttl_sec=0.5)
async def _get_health(
    _none: None,
):  # dummy param _none because caching function expects a single param as cache key.
    async with asyncio.TaskGroup() as tg:
        tts_up = tg.create_task(
            asyncio.to_thread(
                _check_server_status, _ws_to_http(TTS_SERVER) + "/api/build_info"
            )
        )
        stt_up = tg.create_task(
            asyncio.to_thread(
                _check_server_status, _ws_to_http(STT_SERVER) + "/api/build_info"
            )
        )
        llm_up = tg.create_task(
            asyncio.to_thread(
                _check_server_status,
                _ws_to_http(LLM_SERVER) + "/v1/models",
                # The default vLLM server doesn't use auth, but this is needed if you
                # use OpenAI or another LLM server.
                headers={"Authorization": f"Bearer {KYUTAI_LLM_API_KEY}"},
            )
        )
        voice_cloning_up = tg.create_task(
            asyncio.to_thread(
                _check_server_status,
                _ws_to_http(VOICE_CLONING_SERVER) + "/api/build_info",
            )
        )
        tts_up_res = await tts_up
        stt_up_res = await stt_up
        llm_up_res = await llm_up
        voice_cloning_up_res = await voice_cloning_up

    return HealthStatus(
        tts_up=tts_up_res,
        stt_up=stt_up_res,
        llm_up=llm_up_res,
        voice_cloning_up=voice_cloning_up_res,
    )


@app.get("/v1/health")
async def get_health():
    health = await _get_health(None)
    mt.HEALTH_OK.observe(health.ok)
    return health


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
            await websocket.accept(subprotocol=REALTIME_SUBPROTOCOL)

            # Initialize session configuration for extension negotiation
            negotiated_session = NegotiatedSession()

            # Start distributed tracing for this session
            session_id = ora.random_id("sess")
            start_trace(session_id)

            handler = UnmuteHandler()
            # Store negotiated session on handler for access during message handling
            handler.negotiated_session = negotiated_session  # type: ignore[attr-defined]

            async with handler:
                await handler.start_up()
                await _run_route(websocket, handler, negotiated_session)

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

            # End tracing and log summary
            trace_ctx = end_trace()
            if trace_ctx:
                from unmute.tracing import format_trace_summary

                logger.info("Session trace summary:\n%s", format_trace_summary(trace_ctx))


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


async def _run_route(
    websocket: WebSocket,
    handler: UnmuteHandler,
    negotiated_session: NegotiatedSession,
):
    health = await get_health()
    if not health.ok:
        logger.info("Health check failed, closing WebSocket connection.")
        await websocket.close(
            code=status.WS_1011_INTERNAL_ERROR,
            reason=f"Server is not healthy: {health}",
        )
        return

    emit_queue: asyncio.Queue[ora.ServerEvent] = asyncio.Queue()
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(
                receive_loop(websocket, handler, emit_queue, negotiated_session),
                name="receive_loop()",
            )
            tg.create_task(
                emit_loop(websocket, handler, emit_queue), name="emit_loop()"
            )
            tg.create_task(handler.quest_manager.wait(), name="quest_manager.wait()")
            tg.create_task(debug_running_tasks(), name="debug_running_tasks()")
    finally:
        await handler.cleanup()
        logger.info("websocket_route() finished")


async def receive_loop(
    websocket: WebSocket,
    handler: UnmuteHandler,
    emit_queue: asyncio.Queue[ora.ServerEvent],
    negotiated_session: NegotiatedSession,
):
    """Receive messages from the WebSocket.

    Can decide to send messages via `emit_queue`.
    """
    # Use the handler's audio buffer for Opus decoding and frame management
    audio_buffer = handler.audio_buffer
    while True:
        try:
            message_raw = await websocket.receive_text()
        except WebSocketDisconnect as e:
            logger.info(
                "receive_loop() stopped because WebSocket disconnected: "
                f"{e.code=} {e.reason=}"
            )
            raise WebSocketClosedError() from e
        except RuntimeError as e:
            # This is expected when the client disconnects
            if "WebSocket is not connected" not in str(e):
                raise  # re-raise unexpected errors

            logger.info("receive_loop() stopped because WebSocket disconnected.")
            raise WebSocketClosedError() from e

        try:
            message: ora.ClientEvent = ClientEventAdapter.validate_json(message_raw)
        except json.JSONDecodeError as e:
            await emit_queue.put(
                ora.Error(
                    error=ora.ErrorDetails(
                        type="invalid_request_error",
                        message=f"Invalid JSON: {e}",
                    )
                )
            )
            continue
        except ValidationError as e:
            await emit_queue.put(
                ora.Error(
                    error=ora.ErrorDetails(
                        type="invalid_request_error",
                        message="Invalid message",
                        details=json.loads(e.json()),
                    )
                )
            )
            continue

        message_to_record = message

        if isinstance(message, ora.InputAudioBufferAppend):
            opus_bytes = base64.b64decode(message.audio)
            # Use audio buffer for Opus decoding with first-packet sync and
            # frame metadata tracking (timestamps, sequences, latency)

            # Check for overflow before appending
            buffer_was_full = audio_buffer.total_samples >= audio_buffer._max_buffer_samples

            pcm = await audio_buffer.append_opus_async(opus_bytes)

            message_to_record = ora.UnmuteInputAudioBufferAppendAnonymized(
                number_of_samples=pcm.size if pcm is not None else 0,
            )

            # If buffer overflow detected, emit error event
            if buffer_was_full and pcm is None:
                error = make_ora_error(
                    type="buffer_overflow",
                    message=f"Audio buffer overflow: {audio_buffer.total_samples} samples exceeds maximum {audio_buffer._max_buffer_samples}. Frame discarded.",
                )
                await emit_queue.put(error)
                mt.BUFFER_OVERFLOW_ERRORS.inc()
                logger.warning("Buffer overflow detected, frame discarded")

            if pcm is not None and pcm.size:
                await handler.receive((SAMPLE_RATE, pcm[np.newaxis, :]))
        elif isinstance(message, ora.SessionUpdate):
            # Handle extension negotiation from session.update payload
            session_dict = (
                message.session
                if isinstance(message.session, dict)
                else message.session.model_dump()
            )

            # Check for requested extensions in the session config
            requested_extensions = session_dict.get("unmute_extensions")
            if requested_extensions:
                accepted, rejected = validate_extensions(requested_extensions)
                if rejected:
                    # Send error for unsupported extensions
                    error_event = create_extension_error(rejected)
                    await emit_queue.put(error_event)
                    logger.warning(
                        f"Client requested unsupported extensions: {rejected}"
                    )
                # Update negotiated session with accepted extensions
                negotiated_session.extensions = accepted
                logger.info(f"Negotiated extensions: {accepted}")

            # Update negotiated session with other config
            negotiated_session.update_from_session(message.session)

            await handler.update_session(message.session)

            # Convert session to Session object if it's a dict for the response
            session_for_response = (
                ora.Session(**message.session)
                if isinstance(message.session, dict)
                else message.session
            )
            await emit_queue.put(ora.SessionUpdated(session=session_for_response))

            # Record state snapshot after session update
            if handler.recorder is not None:
                await handler.recorder.add_state_snapshot(handler.session_state)

        elif isinstance(message, ora.UnmuteAdditionalOutputs):
            # Don't record this: it's a debugging message and can be verbose. Anything
            # important to store should be in the other event types.
            message_to_record = None

        # Response control events
        elif isinstance(message, ora.ResponseCreate):
            await handler.handle_response_create(message)

        elif isinstance(message, ora.ResponseCancel):
            ack = await handler.handle_response_cancel()
            if ack is not None:
                await emit_queue.put(ack)

        # Conversation item management events
        elif isinstance(message, ora.ConversationItemCreate):
            item, ack = await handler.handle_item_create(message)
            await emit_queue.put(ack)
            # Record state snapshot after item creation
            if handler.recorder is not None:
                await handler.recorder.add_state_snapshot(handler.session_state)

        elif isinstance(message, ora.ConversationItemDelete):
            ack = await handler.handle_item_delete(message.item_id)
            await emit_queue.put(ack)
            # Record state snapshot after item deletion
            if handler.recorder is not None:
                await handler.recorder.add_state_snapshot(handler.session_state)

        elif isinstance(message, ora.ConversationItemRetrieve):
            ack = handler.handle_item_retrieve(message.item_id)
            if ack is not None:
                await emit_queue.put(ack)
            else:
                await emit_queue.put(
                    ora.Error(
                        error=ora.ErrorDetails(
                            type="invalid_request_error",
                            code="item_not_found",
                            message=f"Item '{message.item_id}' not found",
                        )
                    )
                )

        elif isinstance(message, ora.ConversationItemTruncate):
            ack = await handler.handle_item_truncate(message)
            if ack is not None:
                await emit_queue.put(ack)
            else:
                await emit_queue.put(
                    ora.Error(
                        error=ora.ErrorDetails(
                            type="invalid_request_error",
                            code="item_not_found",
                            message=f"Item '{message.item_id}' not found for truncation",
                        )
                    )
                )

        # Input audio buffer control events
        elif isinstance(message, ora.InputAudioBufferCommit):
            item_id, prev_id = await handler.commit_audio_buffer()
            await emit_queue.put(
                ora.InputAudioBufferCommitted(
                    item_id=item_id, previous_item_id=prev_id
                )
            )

        elif isinstance(message, ora.InputAudioBufferClear):
            await handler.clear_audio_buffer()
            await emit_queue.put(ora.InputAudioBufferCleared())

        else:
            logger.info("Ignoring message:", str(message)[:100])

        if message_to_record is not None and handler.recorder is not None:
            await handler.recorder.add_event("client", message_to_record)


class EmitDebugLogger:
    def __init__(self):
        self.last_emitted_n = 0
        self.last_emitted_type = ""

    def on_emit(self, to_emit: ora.ServerEvent):
        if self.last_emitted_type == to_emit.type:
            self.last_emitted_n += 1
        else:
            self.last_emitted_n = 1
            self.last_emitted_type = to_emit.type

        if self.last_emitted_n == 1:
            logger.debug(f"Emitting: {to_emit.type}")
        else:
            logger.debug(f"Emitting ({self.last_emitted_n}): {self.last_emitted_type}")


async def emit_loop(
    websocket: WebSocket,
    handler: UnmuteHandler,
    emit_queue: asyncio.Queue[ora.ServerEvent],
):
    """Send messages to the WebSocket."""
    emit_debug_logger = EmitDebugLogger()

    opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)

    while True:
        if (
            websocket.application_state == WebSocketState.DISCONNECTED
            or websocket.client_state == WebSocketState.DISCONNECTED
        ):
            logger.info("emit_loop() stopped because WebSocket disconnected")
            raise WebSocketClosedError()

        # Monitor emit_queue size for backpressure
        emit_queue_size = emit_queue.qsize()
        mt.EMIT_QUEUE_SIZE.set(emit_queue_size)

        try:
            # Try to get from buffered queue first (non-blocking)
            async with trace_queue_operation("emit_queue", "get_nowait"):
                to_emit = emit_queue.get_nowait()
        except asyncio.QueueEmpty:
            # Fall back to handler emit (blocking)
            async with trace_queue_operation("handler", "emit"):
                emitted_by_handler = await handler.emit()

            if emitted_by_handler is None:
                continue
            elif isinstance(emitted_by_handler, AdditionalOutputs):
                assert len(emitted_by_handler.args) == 1
                to_emit = ora.UnmuteAdditionalOutputs(
                    args=emitted_by_handler.args[0],
                )
            elif isinstance(emitted_by_handler, CloseStream):
                # Close here explicitly so that the receive loop stops too
                await websocket.close()
                break
            elif isinstance(emitted_by_handler, ora.ServerEvent):
                to_emit = emitted_by_handler
            else:
                _sr, audio = emitted_by_handler
                audio = audio_to_float32(audio)
                opus_bytes = await asyncio.to_thread(opus_writer.append_pcm, audio)
                # Due to buffering/chunking, Opus doesn't necessarily output something on every PCM added
                if opus_bytes:
                    to_emit = ora.ResponseAudioDelta(
                        delta=base64.b64encode(opus_bytes).decode("utf-8"),
                    )
                else:
                    continue

        emit_debug_logger.on_emit(to_emit)

        if handler.recorder is not None:
            await handler.recorder.add_event("server", to_emit)

        try:
            # Trace websocket send operations to detect network/serialization bottlenecks
            async with trace_span(
                "websocket_send",
                histogram=WEBSOCKET_SEND_DURATION,
                attributes={"event_type": to_emit.type},
            ):
                await websocket.send_text(to_emit.model_dump_json())
        except (WebSocketDisconnect, RuntimeError) as e:
            if isinstance(e, RuntimeError):
                if "Unexpected ASGI message 'websocket.send'" in str(e):
                    # This is expected when the client disconnects
                    message = f"emit_loop() stopped because WebSocket disconnected: {e}"
                else:
                    raise
            else:
                message = (
                    "emit_loop() stopped because WebSocket disconnected: "
                    f"{e.code=} {e.reason=}"
                )

            logger.info(message)
            raise WebSocketClosedError() from e


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
