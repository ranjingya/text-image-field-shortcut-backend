from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from services.domain.errors import (
    ErrorCategory,
    ProviderError,
    provider_error_from_httpx,
    provider_error_from_status,
)
from services.domain.provider import ImageProviderResult, TextProviderResult
from services.domain.requests import (
    GenerateImageRequest,
    UnderstandImageRequest,
)
from services.gemini_service import GeminiRawResponse
from services.http import (
    AssetFetcher,
    AssetFetchError,
    build_asset_fetcher,
    build_request_timeout,
    get_http_client,
    resolve_image_data_url,
)
from services.response_extractor import extract_text_from_gemini_response
from services.response_normalizer import normalize_gemini_response
from services.settings import AppSettings

logger = logging.getLogger(__name__)


def _build_input_references(
    request: GenerateImageRequest,
    asset_fetcher: AssetFetcher | None,
) -> list[dict[str, Any]]:
    """把临时 OSS 参考图转换为 OpenRouter 接受的 Base64 Data URL。

    参数：
        request: 已完成临时 OSS 暂存的图片生成请求。
        asset_fetcher: 用于读取私有签名 URL 的安全资源下载器。

    返回值：
        仅包含 Base64 Data URL 的 OpenRouter `input_references` 列表。
    """
    if request.file_urls or request.files:
        raise ProviderError(
            provider="openrouter",
            category=ErrorCategory.INVALID_REQUEST,
            message="生成参考图尚未完成临时 OSS 暂存。",
            retryable=False,
            counts_toward_circuit=False,
        )
    if not request.reference_images:
        return []
    if asset_fetcher is None:
        raise RuntimeError("OpenRouter 参考图下载器尚未初始化。")

    references: list[dict[str, Any]] = []
    total_bytes = 0
    try:
        for reference in request.reference_images:
            data_url, content_length = resolve_image_data_url(
                reference.url,
                asset_fetcher,
            )
            total_bytes += content_length
            references.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )
    except AssetFetchError as exc:
        raise ProviderError(
            provider="openrouter",
            category=ErrorCategory.INVALID_ASSET,
            message="OpenRouter 参考图读取或 Base64 转换失败。",
            retryable=False,
            counts_toward_circuit=False,
            cause=exc,
        ) from exc

    logger.debug(
        "provider.openrouter.reference.base64.completed: %s",
        {
            "requestId": request.request_id,
            "referenceCount": len(references),
            "totalBytes": total_bytes,
        },
    )
    return references


def _parse_error_payload(response: httpx.Response) -> tuple[str, str]:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return "", response.text[:1000]
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    if isinstance(error, dict):
        metadata = error.get("metadata", {})
        metadata_error_type = (
            metadata.get("error_type") if isinstance(metadata, dict) else ""
        )
        return str(
            metadata_error_type
            or error.get("type")
            or error.get("error_type")
            or payload.get("error_type")
            or ""
        ), str(error.get("message") or "OpenRouter request failed.")[:1000]
    return "", str(error or "OpenRouter request failed.")[:1000]


class OpenRouterProvider:
    """通过 OpenRouter REST API 执行图片生成和图片理解。"""

    name = "openrouter"

    def __init__(
        self,
        settings: AppSettings,
        base_url: str,
        api_key: str,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client or get_http_client(self.name, settings.http.provider)

    def generate_image(
        self,
        request: GenerateImageRequest,
        public_model: str,
        provider_model: str,
        timeout_seconds: float | None = None,
    ) -> ImageProviderResult:
        """调用 OpenRouter Images API 生成图片。

        参数：
            request: 业务层标准图片生成请求。
            public_model: 客户端使用的公共模型 ID。
            provider_model: OpenRouter 实际接收的模型 ID。
            timeout_seconds: 路由层分配给本次调用的最大秒数。

        返回值：
            包含标准图片结果、模型和耗时的服务商结果。
        """
        body: dict[str, Any] = {
            "model": provider_model,
            "prompt": request.prompt,
            "n": 1,
            "resolution": request.image_size,
        }
        if request.aspect_ratio:
            body["aspect_ratio"] = request.aspect_ratio
        input_references = _build_input_references(
            request,
            build_asset_fetcher(self._settings) if request.reference_images else None,
        )
        if input_references:
            body["input_references"] = input_references

        response, elapsed_ms = self._post(
            "/images", body, request.request_id, timeout_seconds
        )
        raw_response = GeminiRawResponse(
            status_code=response.status_code,
            content_type=response.headers.get("content-type", "application/json"),
            content_disposition="",
            body=response.content,
        )
        try:
            result = normalize_gemini_response(raw_response)
        except Exception as exc:
            raise ProviderError(
                provider=self.name,
                category=ErrorCategory.INVALID_RESPONSE,
                message="OpenRouter 返回了无法解析的图片响应。",
                retryable=True,
                cause=exc,
            ) from exc
        return ImageProviderResult(
            provider=self.name,
            public_model=public_model,
            provider_model=provider_model,
            result=result,
            request_id=response.headers.get("x-request-id", ""),
            elapsed_ms=elapsed_ms,
        )

    def understand_image(
        self,
        request: UnderstandImageRequest,
        public_model: str,
        provider_model: str,
        timeout_seconds: float | None = None,
    ) -> TextProviderResult:
        """调用 OpenRouter Chat Completions API 理解图片。

        参数：
            request: 业务层标准图片理解请求。
            public_model: 客户端使用的公共模型 ID。
            provider_model: OpenRouter 实际接收的模型 ID。
            timeout_seconds: 路由层分配给本次调用的最大秒数。

        返回值：
            包含文本、模型和耗时的服务商结果。
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        asset_fetcher = build_asset_fetcher(self._settings)
        try:
            for url in request.file_urls:
                data_url, _ = resolve_image_data_url(url, asset_fetcher)
                content.append({"type": "image_url", "image_url": {"url": data_url}})
        except AssetFetchError as exc:
            raise ProviderError(
                provider=self.name,
                category=ErrorCategory.INVALID_ASSET,
                message="OpenRouter 图片理解参考图读取或 Base64 转换失败。",
                retryable=False,
                counts_toward_circuit=False,
                cause=exc,
            ) from exc
        body = {
            "model": provider_model,
            "messages": [{"role": "user", "content": content}],
        }
        response, elapsed_ms = self._post(
            "/chat/completions", body, request.request_id, timeout_seconds
        )
        raw_response = GeminiRawResponse(
            status_code=response.status_code,
            content_type=response.headers.get("content-type", "application/json"),
            content_disposition="",
            body=response.content,
        )
        try:
            text = extract_text_from_gemini_response(raw_response)
        except Exception as exc:
            raise ProviderError(
                provider=self.name,
                category=ErrorCategory.INVALID_RESPONSE,
                message="OpenRouter 返回了无法解析的图片理解响应。",
                retryable=True,
                cause=exc,
            ) from exc
        return TextProviderResult(
            provider=self.name,
            public_model=public_model,
            provider_model=provider_model,
            text=text,
            request_id=response.headers.get("x-request-id", ""),
            elapsed_ms=elapsed_ms,
        )

    def _post(
        self,
        path: str,
        body: dict[str, Any],
        request_id: str,
        timeout_seconds: float | None,
    ) -> tuple[httpx.Response, float]:
        if not self._api_key:
            raise ProviderError(
                provider=self.name,
                category=ErrorCategory.AUTHENTICATION,
                message="OpenRouter API Key 未配置。",
                retryable=False,
                counts_toward_circuit=False,
            )

        start_time = time.perf_counter()
        logger.debug(
            "provider.openrouter.request.start: %s",
            {"path": path, "requestId": request_id, "model": body.get("model", "")},
        )
        try:
            response = self._client.post(
                f"{self._base_url}{path}",
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=build_request_timeout(
                    self._settings.http.provider, timeout_seconds
                ),
            )
        except httpx.HTTPError as exc:
            logger.debug(
                "provider.openrouter.request.failed: %s",
                {"requestId": request_id, "errorType": type(exc).__name__},
            )
            raise provider_error_from_httpx(self.name, exc) from exc

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.debug(
            "provider.openrouter.request.finish: %s",
            {
                "path": path,
                "requestId": request_id,
                "statusCode": response.status_code,
                "elapsedMs": elapsed_ms,
            },
        )
        if response.status_code >= 400:
            error_type, message = _parse_error_payload(response)
            raise provider_error_from_status(
                self.name,
                response.status_code,
                message,
                headers=response.headers,
                error_type=error_type,
                request_id=response.headers.get("x-request-id", ""),
                response_bytes=len(response.content),
            )
        return response, elapsed_ms
