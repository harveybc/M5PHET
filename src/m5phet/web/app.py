import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hashlib
import hmac
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .engine import Engine, parse_file
from .store import Store


class CreateChat(BaseModel):
    title: str = Field(default="Nuevo chat", min_length=1, max_length=120)


class EditChat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, min_length=1, max_length=120)
    config: dict | None = None


class ProposeTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(max_length=4000)
    file_ids: list[str] = Field(default_factory=list, max_length=8)


class RunTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(default="", max_length=4000)
    task: dict
    file_ids: list[str] = Field(default_factory=list, max_length=8)
    client_id: str = Field(min_length=1, max_length=80)
    language: str = Field(default="es", pattern="^(es|en)$")


class SendMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=64000)
    client_id: str = Field(min_length=1, max_length=100)
    file_ids: list[str] = Field(default_factory=list, max_length=5)


def create_app(root=None, *, engine=None, access_token=None, allowed_hosts=None):
    root = root or Path.home() / ".local/state/m5phet/chat"
    store = Store(root)
    engine = engine or Engine()
    allowed = set(allowed_hosts or ["127.0.0.1", "localhost", "::1"])
    cookie = hashlib.sha256((access_token or "local").encode()).hexdigest()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="m5phet-chat")

    @asynccontextmanager
    async def lifespan(app):
        store.recover()
        yield
        pool.shutdown(wait=True)

    ATTACHMENT_LIMIT = 8 * 1024 * 1024

    app = FastAPI(title="M5PHET Chat", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def boundary(request, call_next):
        host = urlsplit("//" + request.headers.get("host", "")).hostname
        if host not in allowed:
            return JSONResponse({"detail": "Host not allowed"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse({"detail": "Cross-origin request refused"}, status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-site request refused"}, status_code=403)
        if access_token and request.url.path.startswith("/api/") and request.url.path != "/api/login":
            if not hmac.compare_digest(request.cookies.get("m5phet_owner", ""), cookie):
                return JSONResponse({"detail": "Owner access token required"}, status_code=401)
        # A declared length over the attachment limit is refused here, before anything parses the body. The multipart
        # parser gives up on its own first and answers "there was an error parsing the body", which tells someone holding
        # a 10 MB CSV nothing about what to do; this names the limit and the size they sent.
        declared = request.headers.get("content-length")
        if (request.url.path.endswith("/files") and declared and declared.isdigit()
                and int(declared) > ATTACHMENT_LIMIT + 4096):
            return JSONResponse({"detail": f"Attachment limit: {ATTACHMENT_LIMIT // (1024 * 1024)} MiB; this one is "
                                           f"{int(declared) / (1024 * 1024):.1f} MiB"}, status_code=413)
        # Limit streamed bodies before multipart/JSON parsing, including chunked requests.
        size = 0
        receive = request._receive
        async def bounded_receive():
            nonlocal size
            msg = await receive()
            size += len(msg.get("body", b""))
            if size > 9 * 1024 * 1024:
                from starlette.exceptions import HTTPException
                raise HTTPException(413, "Request too large")
            return msg
        request._receive = bounded_receive
        result = await call_next(request)
        result.headers["X-Content-Type-Options"] = "nosniff"
        result.headers["Referrer-Policy"] = "no-referrer"
        result.headers["Cache-Control"] = "no-store"
        result.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return result

    for exc, status in ((KeyError, 404), (ValueError, 422), (RuntimeError, 409)):
        async def handle(request, error, status=status):
            return JSONResponse({"detail": str(error)}, status_code=status)
        app.add_exception_handler(exc, handle)

    @app.post("/api/login")
    async def login(request: Request):
        value = (await request.json()).get("token", "")
        if access_token and (not isinstance(value, str) or not hmac.compare_digest(value, access_token)):
            return JSONResponse({"detail": "Invalid token"}, status_code=401)
        response = JSONResponse({"ok": True})
        response.set_cookie("m5phet_owner", cookie, httponly=True, samesite="strict", secure=request.url.scheme == "https")
        return response

    @app.get("/api/catalog")
    def catalog():
        return engine.catalog()

    @app.get("/api/chats")
    def chats():
        return store.list()

    @app.post("/api/chats", status_code=201)
    def create(body: CreateChat):
        return store.create(body.title)

    @app.get("/api/chats/{cid}")
    def get(cid: str):
        return store.get(cid)

    @app.patch("/api/chats/{cid}")
    def edit(cid: str, body: EditChat):
        return store.update(cid, body.title, body.config)

    @app.delete("/api/chats/{cid}", status_code=204)
    def delete(cid: str):
        store.delete(cid)
        return Response(status_code=204)

    @app.get("/api/chats/{cid}/export")
    def export(cid: str):
        return JSONResponse(store.get(cid), headers={"Content-Disposition": 'attachment; filename="m5phet-chat.json"'})

    @app.post("/api/chats/{cid}/files", status_code=201)
    async def upload(cid: str, request: Request, file: UploadFile = File()):
        # The multipart parser gives up on an oversize body before the check below can run, and its message -- "there was
        # an error parsing the body" -- tells someone with a 10 MB CSV nothing about what to do. The declared length is
        # refused first, with the limit named, and without reading the body at all.
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > ATTACHMENT_LIMIT + 4096:
            return JSONResponse({"detail": f"Attachment limit: {ATTACHMENT_LIMIT // (1024 * 1024)} MiB; "
                                           f"this one declares {int(declared) // (1024 * 1024)} MiB"}, status_code=413)
        try:
            data = await file.read(ATTACHMENT_LIMIT + 1)
        finally:
            await file.close()
        if len(data) > ATTACHMENT_LIMIT:
            return JSONResponse({"detail": f"Attachment limit: {ATTACHMENT_LIMIT // (1024 * 1024)} MiB"}, status_code=413)
        name = Path((file.filename or "upload").replace("\\", "/")).name
        name = "".join(ch for ch in name if ch.isprintable())[:160]
        parse_file(name, data)
        return store.upload(cid, name, data)

    @app.get("/api/chats/{cid}/files/{fid}")
    def download(cid: str, fid: str):
        entry = store.file(cid, fid)
        return Response(entry["data"], media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{fid}.txt"'})

    def execute(mid, job):
        prompt, config, attachments, snapshot = job
        started = time.monotonic()
        try:
            detail = engine.execute(prompt, config, attachments)
            result = detail["result"]
            status = result["status"]
            content = result.get("why") or ("Resultado" if status == "OK" else status)
            store.finish(mid, status, content, snapshot | detail | {"elapsed_seconds": time.monotonic() - started})
        except Exception as error:
            store.finish(mid, "REFUSED", f"{type(error).__name__}: {error}", snapshot | {
                "elapsed_seconds": time.monotonic() - started, "execution_authorized": False})

    @app.get("/api/tasks/catalog")
    def task_catalog():
        return {"areas": engine.task_catalog(), "interpreter": engine.interpreter.identity(),
                "outputs": engine.output_headers(), "execution_authorized": False}

    @app.post("/api/chats/{cid}/tasks/propose")
    def propose(cid: str, body: ProposeTask):
        # a preview, not a message: nothing is recorded until the person runs the envelope they saw
        attachments = [store.file(cid, fid) for fid in body.file_ids]
        return engine.propose_task(body.prompt, attachments)

    @app.post("/api/chats/{cid}/preview")
    def preview(cid: str, body: ProposeTask):
        # the sentence path's window: the typed request as it would run, and which words or model resolved each field.
        # Nothing runs and nothing is recorded; the person sends the same sentence to run it.
        chat = store.get(cid)
        attachments = [store.file(cid, fid) for fid in body.file_ids]
        return engine.execute(body.prompt, chat["config"], attachments, dry_run=True)

    def execute_task(mid, job, task, language):
        prompt, _config, attachments, snapshot = job
        started = time.monotonic()
        try:
            detail = engine.execute_task(prompt, task, attachments, language=language)
            response = detail["response"]
            status = "OK" if response["refused"] == 0 else ("PARTIAL" if response["answered"] else "REFUSED")
            store.finish(mid, status, detail["narration"]["text"],
                         snapshot | detail | {"elapsed_seconds": time.monotonic() - started})
        except Exception as error:
            store.finish(mid, "REFUSED", f"{type(error).__name__}: {error}", snapshot | {
                "task": task, "elapsed_seconds": time.monotonic() - started, "execution_authorized": False})

    @app.post("/api/chats/{cid}/tasks/run", status_code=202)
    def run_envelope(cid: str, body: RunTask):
        if not isinstance(body.task, dict):
            raise ValueError("The envelope must be a JSON object")
        # the envelope is part of the request identity, so editing it and sending again is a new request, not a replay
        mid, job = store.begin(cid, body.client_id, body.prompt or "", body.file_ids, extra=body.task)
        if job is not None:
            pool.submit(execute_task, mid, job, body.task, body.language)
        return {"message_id": mid}

    @app.post("/api/chats/{cid}/messages", status_code=202)
    def send(cid: str, body: SendMessage):
        if not body.prompt.strip():
            raise ValueError("Question is empty")
        mid, job = store.begin(cid, body.client_id, body.prompt, body.file_ids)
        if job is not None:
            pool.submit(execute, mid, job)
        return {"message_id": mid}

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static, check_dir=False), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    return app


def main():
    parser = argparse.ArgumentParser(description="M5PHET local single-owner chat")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--allowed-host", action="append", default=[])
    args = parser.parse_args()
    token = os.getenv("M5PHET_CHAT_TOKEN")
    if args.host not in ("127.0.0.1", "localhost", "::1") and (not token or len(token) < 24 or not args.allowed_host):
        parser.error("Non-loopback requires M5PHET_CHAT_TOKEN (24+ characters) and --allowed-host")
    import uvicorn
    uvicorn.run(create_app(args.state_dir, access_token=token,
                          allowed_hosts=["localhost", "127.0.0.1", "::1"] + args.allowed_host),
                host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
