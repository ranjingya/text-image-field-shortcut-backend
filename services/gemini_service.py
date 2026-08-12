from __future__ import annotations

import base64
import binascii
import json
import logging
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import parse

import httpx

from services.domain.errors import (
    ErrorCategory,
    ProviderError,
    provider_error_from_httpx,
    provider_error_from_status,
    sanitize_provider_error_message,
)
from services.domain.requests import (
    GenerateImageRequest,
    ReferenceImageInfo,
    UnderstandImageRequest,
)
from services.http import (
    AssetFetcher,
    build_asset_fetcher,
    build_request_timeout,
    get_http_client,
)
from services.settings import AppSettings, HttpClientSettings

logger = logging.getLogger(__name__)

def _parse_gemini_error_payload(
    response_body: bytes,
    status_code: int,
) -> tuple[str, str]:
    """提取 EasyRouter 错误类型和安全消息。

    参数：
        response_body: EasyRouter 返回的原始错误响应正文。
        status_code: EasyRouter 返回的 HTTP 状态码。

    返回值：
        服务商错误类型和已脱敏、限长的错误消息。
    """
    fallback_message = f"Gemini HTTP {status_code}"
    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError, UnicodeDecodeError, TypeError:
        message = sanitize_provider_error_message(
            response_body.decode("utf-8", errors="replace")
        )
        return "", message or fallback_message

    if not isinstance(payload, dict):
        return "", fallback_message
    error = payload.get("error", payload)
    if isinstance(error, dict):
        metadata = error.get("metadata")
        metadata_error_type = (
            metadata.get("error_type") if isinstance(metadata, dict) else ""
        )
        error_type = str(
            metadata_error_type
            or error.get("type")
            or error.get("error_type")
            or error.get("status")
            or payload.get("error_type")
            or ""
        )
        message = sanitize_provider_error_message(
            error.get("message")
            or error.get("detail")
            or payload.get("message")
            or fallback_message
        )
        return error_type, message
    message = sanitize_provider_error_message(error or payload.get("message"))
    return "", message or fallback_message


def _guess_mime_type(file_name: str, fallback: str = "application/octet-stream") -> str:
    guessed, _ = mimetypes.guess_type(file_name)
    return guessed or fallback


def _safe_file_name(file_name: str, fallback: str) -> str:
    clean_name = Path(str(file_name or "")).name.strip()
    return clean_name or fallback


def _build_inline_data_part(base64_data: str, mime_type: str) -> dict[str, Any]:
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": base64_data,
        }
    }


def _build_file_data_part(reference: ReferenceImageInfo) -> dict[str, Any]:
    """构建 EasyRouter Gemini 接受的外部参考图部分。"""
    return {
        "fileData": {
            "mimeType": reference.mime_type,
            "fileUri": reference.url,
        }
    }


def _decode_data_url(value: str) -> tuple[str, bytes] | None:
    prefix, separator, encoded = str(value or "").partition(",")
    if not separator or not prefix.startswith("data:") or ";base64" not in prefix:
        return None

    mime_type = (
        prefix.removeprefix("data:").split(";", 1)[0].strip()
        or "application/octet-stream"
    )
    return mime_type, base64.b64decode(encoded)


@dataclass
class PreparedReferenceInput:
    source_type: str
    mime_type: str
    file_name: str
    payload_size: int
    base64_data: str
    has_source_reference: bool = False

    @classmethod
    def from_bytes(
        cls,
        *,
        source_type: str,
        mime_type: str,
        file_name: str,
        payload: bytes,
        has_source_reference: bool,
    ) -> PreparedReferenceInput:
        """从图片字节构建不保留原始字节的参考图输入。

        参数：
            source_type: 参考图来源类型。
            mime_type: 已识别的图片 MIME 类型。
            file_name: 用于调试摘要的安全文件名。
            payload: 待编码的原始图片字节。
            has_source_reference: 是否来自外部资源地址。

        返回值：
            仅保存 Base64 与字节数的参考图输入。
        """
        return cls(
            source_type=source_type,
            mime_type=mime_type,
            file_name=file_name,
            payload_size=len(payload),
            base64_data=base64.b64encode(payload).decode("ascii"),
            has_source_reference=has_source_reference,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceType": self.source_type,
            "mimeType": self.mime_type,
            "fileName": self.file_name,
            "payloadSize": self.payload_size,
            "hasSourceReference": self.has_source_reference,
        }


@dataclass
class GeminiInvocationPlan:
    api_url: str
    api_path: str
    model: str
    prompt: str
    prepared_inputs: list[PreparedReferenceInput]
    request_body: dict[str, Any]

    def _build_request_body_preview(self) -> dict[str, Any]:
        parts = self.request_body.get("contents", [{}])[0].get("parts", [])
        parts_preview = []
        for item in parts:
            if isinstance(item.get("text"), str):
                parts_preview.append({"textLength": len(item["text"])})
                continue

            inline_data = item.get("inline_data")
            if isinstance(inline_data, dict):
                parts_preview.append(
                    {
                        "inline_data": {
                            "mime_type": inline_data.get(
                                "mime_type", "application/octet-stream"
                            ),
                            "data": "<base64>",
                        }
                    }
                )
                continue

            file_data = item.get("fileData")
            if isinstance(file_data, dict):
                parts_preview.append(
                    {
                        "fileData": {
                            "mimeType": file_data.get(
                                "mimeType", "application/octet-stream"
                            ),
                            "fileUri": "<signed-url>",
                        }
                    }
                )
                continue

            parts_preview.append(item)

        return {
            "contents": [
                {
                    "role": "user",
                    "parts": parts_preview,
                }
            ],
            "generationConfig": self.request_body.get("generationConfig", {}),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "apiUrl": self.api_url,
            "apiPath": self.api_path,
            "model": self.model,
            "promptLength": len(self.prompt),
            "preparedInputCount": len(self.prepared_inputs),
            "preparedInputs": [item.to_dict() for item in self.prepared_inputs],
            "requestBody": self._build_request_body_preview(),
        }


@dataclass
class GeminiRawResponse:
    status_code: int
    content_type: str
    content_disposition: str
    body: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "statusCode": self.status_code,
            "contentType": self.content_type,
            "contentDisposition": self.content_disposition,
            "bodyLength": len(self.body),
        }


def _normalize_api_url(api_url: str) -> str:
    normalized = str(api_url or "").strip().rstrip("/")
    base_url = normalized.removesuffix("/v1/chat/completions")
    return base_url


def _build_endpoint(api_url: str, api_path: str) -> str:
    base_url = _normalize_api_url(api_url)
    return f"{base_url}{api_path if not base_url.endswith('/v1') else api_path.removeprefix('/v1')}"


def _read_url_as_inline_input(
    file_url: str, asset_fetcher: AssetFetcher
) -> PreparedReferenceInput:
    try:
        decoded_data_url = _decode_data_url(file_url)
    except (binascii.Error, ValueError) as exc:
        raise ProviderError(
            provider="easyrouter",
            category=ErrorCategory.INVALID_ASSET,
            message="参考图片 Data URL 无法解码。",
            retryable=False,
            counts_toward_circuit=False,
            cause=exc,
        ) from exc
    if decoded_data_url:
        mime_type, payload = decoded_data_url
        return PreparedReferenceInput.from_bytes(
            source_type="data_url",
            mime_type=mime_type,
            file_name=f"reference{mimetypes.guess_extension(mime_type) or '.bin'}",
            payload=payload,
            has_source_reference=True,
        )

    request_url = str(file_url or "").strip()
    if not request_url:
        raise RuntimeError("Encountered an empty reference image URL.")

    url_parts = parse.urlparse(request_url)
    fallback_name = Path(url_parts.path).name or "reference"

    try:
        fetched_asset = asset_fetcher.fetch(request_url)
        payload = fetched_asset.body
        response_mime_type = fetched_asset.content_type
    except Exception as exc:
        raise ProviderError(
            provider="easyrouter",
            category=ErrorCategory.INVALID_ASSET,
            message="参考图片下载失败或格式不受支持。",
            retryable=False,
            counts_toward_circuit=False,
            cause=exc,
        ) from exc

    mime_type = response_mime_type or _guess_mime_type(fallback_name)
    file_name = _safe_file_name(
        fallback_name, f"reference{mimetypes.guess_extension(mime_type) or '.bin'}"
    )
    return PreparedReferenceInput.from_bytes(
        source_type="url",
        mime_type=mime_type,
        file_name=file_name,
        payload=payload,
        has_source_reference=True,
    )


def _prepare_url_reference_inputs(
    file_urls: list[str], asset_fetcher: AssetFetcher
) -> list[PreparedReferenceInput]:
    return [
        _read_url_as_inline_input(file_url, asset_fetcher) for file_url in file_urls
    ]


def _build_gemini_request_body(
    prompt: str,
    references: list[ReferenceImageInfo],
    aspect_ratio: str | None,
    image_size: str,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    parts.extend(_build_file_data_part(reference) for reference in references)
    image_config = {
        "imageSize": image_size,
    }
    if aspect_ratio:
        image_config["aspectRatio"] = aspect_ratio

    return {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": image_config,
        },
    }


def build_gemini_invocation_plan(
    request_data: GenerateImageRequest,
    settings: AppSettings,
    base_url: str,
) -> GeminiInvocationPlan:
    """构建 Gemini 图片生成调用计划。

    参数：
        request_data: 已解析且已映射服务商模型的图片生成请求。
        settings: HTTP 客户端与资源下载配置。
        base_url: 服务商配置文件中定义的接口地址。

    返回值：
        包含接口地址、模型、参考图和请求体的调用计划。
    """
    if request_data.file_urls or request_data.files:
        raise ProviderError(
            provider="easyrouter",
            category=ErrorCategory.INVALID_REQUEST,
            message="生成参考图尚未完成临时 OSS 暂存。",
            retryable=False,
            counts_toward_circuit=False,
        )
    resolved_model = request_data.model
    api_path = f"/v1beta/models/{resolved_model}:generateContent"
    request_body = _build_gemini_request_body(
        request_data.prompt,
        request_data.reference_images,
        request_data.aspect_ratio,
        request_data.image_size,
    )
    return GeminiInvocationPlan(
        api_url=_build_endpoint(base_url, api_path),
        api_path=api_path,
        model=resolved_model,
        prompt=request_data.prompt,
        prepared_inputs=[],
        request_body=request_body,
    )


def _build_gemini_text_request_body(
    prompt: str,
    prepared_inputs: list[PreparedReferenceInput],
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    parts.extend(
        _build_inline_data_part(item.base64_data, item.mime_type)
        for item in prepared_inputs
    )
    if prompt:
        parts.append({"text": prompt})
    return {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
    }


def build_gemini_understand_plan(
    request_data: UnderstandImageRequest,
    settings: AppSettings,
    base_url: str,
) -> GeminiInvocationPlan:
    """构建 Gemini 图片理解调用计划。

    参数：
        request_data: 已解析且已映射服务商模型的图片理解请求。
        settings: HTTP 客户端与资源下载配置。
        base_url: 服务商配置文件中定义的接口地址。

    返回值：
        包含接口地址、模型、参考图和请求体的调用计划。
    """
    prepared_inputs = _prepare_url_reference_inputs(
        request_data.file_urls, build_asset_fetcher(settings)
    )
    resolved_model = request_data.model
    api_path = f"/v1beta/models/{resolved_model}:generateContent"
    request_body = _build_gemini_text_request_body(
        request_data.prompt,
        prepared_inputs,
    )
    return GeminiInvocationPlan(
        api_url=_build_endpoint(base_url, api_path),
        api_path=api_path,
        model=resolved_model,
        prompt=request_data.prompt,
        prepared_inputs=prepared_inputs,
        request_body=request_body,
    )


def invoke_gemini(
    invocation_plan: GeminiInvocationPlan,
    api_key: str,
    client_settings: HttpClientSettings | None = None,
    client: httpx.Client | None = None,
    timeout_seconds: float | None = None,
) -> GeminiRawResponse:
    """调用 Gemini 兼容接口。

    参数：
        invocation_plan: 已完成模型解析和请求体构建的调用计划。
        api_key: 服务商 API Key。
        client_settings: 共享客户端使用的超时与连接池配置。
        client: 测试或特殊场景注入的 HTTPX 客户端。
        timeout_seconds: 路由层分配给本次调用的最大秒数。

    返回值：
        包含响应状态、响应头和原始字节的 Gemini 响应。
    """
    if not api_key:
        raise ProviderError(
            provider="easyrouter",
            category=ErrorCategory.AUTHENTICATION,
            message="EasyRouter API Key 未配置。",
            retryable=True,
            counts_toward_circuit=True,
        )

    request_body_size = len(str(invocation_plan.request_body).encode("utf-8"))
    resolved_client_settings = client_settings or HttpClientSettings()
    http_client = client or get_http_client(
        "easyrouter",
        resolved_client_settings,
    )
    start_time = time.perf_counter()

    logger.debug(
        "gemini.backend.request.start: %s",
        {
            "apiUrl": invocation_plan.api_url,
            "hasApiKey": bool(api_key),
            "requestBodySize": request_body_size,
        },
    )

    try:
        response = http_client.post(
            invocation_plan.api_url,
            json=invocation_plan.request_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Accept": "*/*",
            },
            timeout=build_request_timeout(resolved_client_settings, timeout_seconds),
        )
        response_body = response.content
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.debug(
            "gemini.backend.generation_time: %s",
            {
                "model": invocation_plan.model,
                "status": response.status_code,
                "elapsedMs": elapsed_ms,
            },
        )

        logger.debug(
            "gemini.backend.response.received: %s",
            {
                "status": response.status_code,
                "contentType": response.headers.get("content-type", ""),
                "contentDisposition": response.headers.get("content-disposition", ""),
                "bodyLength": len(response_body),
                "elapsedMs": elapsed_ms,
            },
        )

        if response.status_code >= 400:
            error_type, error_message = _parse_gemini_error_payload(
                response_body,
                response.status_code,
            )
            logger.debug(
                "gemini.backend.request.http_error: %s",
                {
                    "status": response.status_code,
                    "elapsedMs": elapsed_ms,
                    "bodyLength": len(response_body),
                    "providerErrorType": error_type,
                    "message": error_message,
                },
            )
            raise provider_error_from_status(
                "easyrouter",
                response.status_code,
                error_message,
                headers=response.headers,
                error_type=error_type,
                request_id=response.headers.get("x-request-id", ""),
                response_bytes=len(response_body),
            )

        return GeminiRawResponse(
            status_code=response.status_code,
            content_type=response.headers.get("content-type", ""),
            content_disposition=response.headers.get("content-disposition", ""),
            body=response_body,
        )
    except ProviderError:
        raise
    except httpx.HTTPError as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.debug(
            "gemini.backend.request.http_exception: %s",
            {
                "apiUrl": invocation_plan.api_url,
                "model": invocation_plan.model,
                "elapsedMs": elapsed_ms,
                "errorType": type(exc).__name__,
                "error": str(exc),
            },
            exc_info=True,
        )
        raise provider_error_from_httpx("easyrouter", exc) from exc
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(
            "gemini.backend.request.unexpected_exception: %s",
            {
                "apiUrl": invocation_plan.api_url,
                "model": invocation_plan.model,
                "elapsedMs": elapsed_ms,
                "errorType": type(exc).__name__,
                "error": str(exc),
            },
            exc_info=True,
        )
        raise RuntimeError(
            f"Gemini request to {invocation_plan.api_url} failed: {exc}"
        ) from exc
