from __future__ import annotations

import base64
import binascii
import ipaddress
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from services.http.client_factory import get_http_client
from services.settings import AppSettings

logger = logging.getLogger(__name__)

_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


class AssetFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchedAsset:
    body: bytes
    content_type: str
    final_url: str


def _safe_url_for_log(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _validate_public_http_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise AssetFetchError("资源地址必须是有效的 HTTP 或 HTTPS URL。")

    try:
        addresses = socket.getaddrinfo(
            parts.hostname, parts.port, type=socket.SOCK_STREAM
        )
    except OSError as exc:
        raise AssetFetchError("资源地址无法解析。") from exc

    for address in addresses:
        ip_value = ipaddress.ip_address(address[4][0])
        if not ip_value.is_global:
            raise AssetFetchError("资源地址指向了不允许访问的网络。")


def detect_image_content_type(body: bytes) -> str:
    """根据文件头识别受支持的图片 MIME 类型。"""
    if body.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if body.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if body.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if body.startswith(b"RIFF") and body[8:12] == b"WEBP":
        return "image/webp"
    if body.startswith(b"BM"):
        return "image/bmp"
    if body.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    return ""


def resolve_image_data_url(value: str, asset_fetcher: AssetFetcher) -> tuple[str, int]:
    """把远程图片或 Data URL 统一转换为可信的图片 Data URL。

    参数：
        value: 待处理的远程图片 URL 或 Base64 Data URL。
        asset_fetcher: 用于安全下载远程图片的资源下载器。

    返回值：
        规范化后的图片 Data URL，以及原始图片字节数。
    """
    normalized = str(value or "").strip()
    if normalized.startswith("data:"):
        prefix, separator, encoded = normalized.partition(",")
        if not separator or ";base64" not in prefix:
            raise AssetFetchError("图片 Data URL 格式不正确。")
        try:
            body = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise AssetFetchError("图片 Data URL 无法解码。") from exc
        content_type = detect_image_content_type(body)
        if not content_type:
            raise AssetFetchError("资源内容不是受支持的图片格式。")
    else:
        fetched = asset_fetcher.fetch(normalized)
        body = fetched.body
        content_type = fetched.content_type

    encoded_body = base64.b64encode(body).decode("ascii")
    return f"data:{content_type};base64,{encoded_body}", len(body)


class AssetFetcher:
    """安全下载外部图片资源。"""

    def __init__(
        self, client: httpx.Client, max_bytes: int, max_redirects: int
    ) -> None:
        self._client = client
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects

    def fetch(self, url: str) -> FetchedAsset:
        """下载资源并限制重定向和响应体大小。

        参数：
            url: 待下载的外部 HTTP 或 HTTPS 地址。

        返回值：
            包含响应字节、内容类型和最终地址的下载结果。
        """
        current_url = str(url or "").strip()
        if not current_url:
            raise AssetFetchError("资源地址不能为空。")

        for redirect_count in range(self._max_redirects + 1):
            _validate_public_http_url(current_url)
            safe_url = _safe_url_for_log(current_url)
            logger.debug(
                "http.asset.download.start: %s",
                {"url": safe_url, "redirectCount": redirect_count},
            )
            try:
                with self._client.stream("GET", current_url) as response:
                    if response.status_code in _REDIRECT_STATUS_CODES:
                        location = response.headers.get("location", "").strip()
                        if not location:
                            raise AssetFetchError("资源重定向响应缺少 Location。")
                        if redirect_count >= self._max_redirects:
                            raise AssetFetchError("资源重定向次数超过限制。")
                        current_url = urljoin(current_url, location)
                        continue

                    response.raise_for_status()
                    content_length = response.headers.get("content-length", "").strip()
                    if content_length and int(content_length) > self._max_bytes:
                        raise AssetFetchError("资源大小超过限制。")

                    chunks: list[bytes] = []
                    total_bytes = 0
                    for chunk in response.iter_bytes():
                        total_bytes += len(chunk)
                        if total_bytes > self._max_bytes:
                            raise AssetFetchError("资源大小超过限制。")
                        chunks.append(chunk)

                    body = b"".join(chunks)
                    detected_content_type = detect_image_content_type(body)
                    if not detected_content_type:
                        raise AssetFetchError("资源内容不是受支持的图片格式。")
                    logger.debug(
                        "http.asset.download.success: %s",
                        {"url": safe_url, "size": len(body)},
                    )
                    return FetchedAsset(
                        body=body,
                        content_type=detected_content_type,
                        final_url=current_url,
                    )
            except AssetFetchError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning(
                    "http.asset.download.failed: %s",
                    {"url": safe_url, "errorType": type(exc).__name__},
                )
                raise AssetFetchError("资源下载失败。") from exc

        raise AssetFetchError("资源重定向次数超过限制。")


def build_asset_fetcher(settings: AppSettings) -> AssetFetcher:
    """根据应用配置创建资源下载器。

    参数：
        settings: 包含资源下载客户端与大小限制的应用配置。

    返回值：
        复用当前 worker 连接池的资源下载器。
    """
    return AssetFetcher(
        client=get_http_client("asset", settings.http.asset),
        max_bytes=settings.http.asset_max_bytes,
        max_redirects=settings.http.asset.max_redirects,
    )
