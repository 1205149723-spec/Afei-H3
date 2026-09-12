from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import re
import subprocess
from pathlib import Path

from jupyter_server.utils import url_path_join
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from tornado.ioloop import IOLoop
from tornado.web import HTTPError, RequestHandler


_JUPYTER_CONFIG = Path("/init/jupyter/jupyter_config.py")
_H3_ORIGIN = "http://127.0.0.1:6006"
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_PRIVATE_REQUEST_HEADERS = {"authorization", "cookie", "host", "x-xsrftoken"}
_START_LOCK: asyncio.Lock | None = None


def _jupyter_token() -> str:
    try:
        text = _JUPYTER_CONFIG.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(r"c\.ServerApp\.token\s*=\s*['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else ""


def _access_key() -> str:
    token = _jupyter_token()
    if not token:
        return ""
    return hashlib.sha256(("afei-h3-public-v1:" + token).encode("utf-8")).hexdigest()[:32]


def setup_afei_h3() -> dict:
    key = _access_key()
    path_info = f"afei-h3-open/{key}/index.html" if key else "afei-h3-unavailable/"
    return {
        "command": [],
        "launcher_entry": {
            "enabled": bool(key),
            "title": "阿飞 H3 工作台",
            "category": "Other",
            "path_info": path_info,
        },
        "new_browser_tab": False,
    }


def _jupyter_server_extension_points() -> list[dict[str, str]]:
    return [{"module": "afei_h3_proxy"}]


def _start_lock() -> asyncio.Lock:
    global _START_LOCK
    if _START_LOCK is None:
        _START_LOCK = asyncio.Lock()
    return _START_LOCK


async def _h3_ready() -> bool:
    request = HTTPRequest(
        _H3_ORIGIN + "/api/health",
        method="GET",
        connect_timeout=1.5,
        request_timeout=2.0,
    )
    try:
        response = await AsyncHTTPClient().fetch(request, raise_error=False)
    except Exception:
        return False
    return response.code == 200


async def _ensure_h3_started() -> None:
    if await _h3_ready():
        return
    async with _start_lock():
        if await _h3_ready():
            return
        log_path = Path("/root/h3-jupyter-launcher.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("ab", buffering=0)
        env = os.environ.copy()
        env["H3_HOST"] = "127.0.0.1"
        env["H3_PORT"] = "6006"
        subprocess.Popen(
            ["bash", "/root/start.sh"],
            cwd="/root",
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(180):
            if await _h3_ready():
                return
            await asyncio.sleep(1)
        raise HTTPError(503, "H3 startup timed out")


class AfeiH3ProxyHandler(RequestHandler):
    def check_xsrf_cookie(self) -> None:
        return

    def _check_key(self, key: str) -> None:
        expected = _access_key()
        if not expected or not hmac.compare_digest(key, expected):
            raise HTTPError(404)

    def _backend_url(self, path: str) -> str:
        target = _H3_ORIGIN + "/" + path.lstrip("/")
        if self.request.query:
            target += "?" + self.request.query
        return target

    def _backend_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        for name, value in self.request.headers.get_all():
            lower = name.lower()
            if lower in _HOP_BY_HOP or lower in _PRIVATE_REQUEST_HEADERS or lower == "content-length":
                continue
            headers[name] = value
        return headers

    def _copy_response_headers(self, headers) -> None:
        for name, value in headers.get_all():
            lower = name.lower()
            if lower in _HOP_BY_HOP or lower in {"content-length", "set-cookie"}:
                continue
            self.set_header(name, value)
        self.set_header("Referrer-Policy", "no-referrer")

    async def _proxy_sse(self, target: str, headers: dict[str, str]) -> None:
        self.set_status(200)
        self.set_header("Content-Type", "text/event-stream")
        self.set_header("Cache-Control", "no-cache")
        self.set_header("X-Accel-Buffering", "no")
        self.set_header("Referrer-Policy", "no-referrer")
        await self.flush()

        def on_chunk(chunk: bytes) -> None:
            if not chunk or self.request.connection.stream.closed():
                return
            try:
                self.write(chunk)
                IOLoop.current().add_callback(self.flush)
            except Exception:
                return

        request = HTTPRequest(
            target,
            method="GET",
            headers=headers,
            connect_timeout=10,
            request_timeout=0,
            streaming_callback=on_chunk,
            decompress_response=False,
        )
        try:
            await AsyncHTTPClient().fetch(request, raise_error=False)
        except Exception:
            return

    async def _proxy(self, key: str, path: str) -> None:
        self._check_key(key)
        await _ensure_h3_started()
        target = self._backend_url(path)
        headers = self._backend_headers()

        if path.rstrip("/") == "api/events" and self.request.method == "GET":
            await self._proxy_sse(target, headers)
            return

        method = self.request.method.upper()
        body = None if method in {"GET", "HEAD"} else self.request.body
        request = HTTPRequest(
            target,
            method=method,
            headers=headers,
            body=body,
            follow_redirects=False,
            connect_timeout=10,
            request_timeout=3600,
            decompress_response=False,
            allow_nonstandard_methods=True,
        )
        try:
            response = await AsyncHTTPClient().fetch(request, raise_error=False)
        except Exception as exc:
            raise HTTPError(502, f"H3 proxy failed: {exc}") from exc

        self.set_status(response.code, response.reason)
        self._copy_response_headers(response.headers)
        if method != "HEAD" and response.body:
            self.write(response.body)

    async def get(self, key: str, path: str = "") -> None:
        await self._proxy(key, path)

    async def post(self, key: str, path: str = "") -> None:
        await self._proxy(key, path)

    async def put(self, key: str, path: str = "") -> None:
        await self._proxy(key, path)

    async def patch(self, key: str, path: str = "") -> None:
        await self._proxy(key, path)

    async def delete(self, key: str, path: str = "") -> None:
        await self._proxy(key, path)

    async def head(self, key: str, path: str = "") -> None:
        await self._proxy(key, path)

    async def options(self, key: str, path: str = "") -> None:
        await self._proxy(key, path)


def _load_jupyter_server_extension(server_app) -> None:
    web_app = server_app.web_app
    base_url = web_app.settings.get("base_url", "/")
    route = url_path_join(base_url, r"afei-h3-open/([0-9a-f]{32})/(.*)")
    web_app.add_handlers(".*$", [(route, AfeiH3ProxyHandler)])
    server_app.log.info("Afei H3 protected proxy registered at %s", route)
