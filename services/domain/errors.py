from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx


class ErrorCategory(StrEnum):
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    LOCAL_CAPACITY = "local_capacity"
    INVALID_RESPONSE = "invalid_response"
    EMPTY_RESPONSE = "empty_response"
    AUTHENTICATION = "authentication"
    BILLING = "billing"
    PERMISSION = "permission"
    INVALID_REQUEST = "invalid_request"
    INVALID_ASSET = "invalid_asset"
    CONTENT_POLICY = "content_policy"
    CAPABILITY = "capability"
    UNKNOWN = "unknown"


_TRANSIENT_REFERENCE_FETCH_MARKERS = (
    "URL_UNREACHABLE",
    "UNREACHABLE_CONNECT_TIMEOUT",
)


@dataclass
class ProviderError(RuntimeError):
    provider: str
    category: ErrorCategory
    message: str
    status_code: int | None = None
    retryable: bool = False
    retry_after_seconds: float | None = None
    request_id: str = ""
    provider_error_type: str = ""
    response_bytes: int | None = None
    counts_toward_circuit: bool = True
    cause: Exception | None = None

    def __str__(self) -> str:
        return self.message

    def detached(self) -> ProviderError:
        """创建不携带底层异常、调用栈和大型 HTTP 请求体的错误副本。"""
        return ProviderError(
            provider=self.provider,
            category=self.category,
            message=self.message,
            status_code=self.status_code,
            retryable=self.retryable,
            retry_after_seconds=self.retry_after_seconds,
            request_id=self.request_id,
            provider_error_type=self.provider_error_type,
            response_bytes=self.response_bytes,
            counts_toward_circuit=self.counts_toward_circuit,
        )


def sanitize_provider_error_message(value: Any, max_length: int = 1000) -> str:
    """压缩服务商错误消息并清理其中的远程 URL 和 Data URL。

    参数：
        value: 服务商返回的原始错误消息。
        max_length: 脱敏后允许保留的最大字符数。

    返回值：
        已完成空白压缩、敏感地址替换和限长的错误摘要。
    """
    normalized = " ".join(str(value or "").split())
    without_data_urls = re.sub(
        r"data:[^;\s,]+;base64,[A-Za-z0-9+/=_-]+",
        "<data-url>",
        normalized,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"https?://[^\s\"'<>]+",
        "<url>",
        without_data_urls,
        flags=re.IGNORECASE,
    )
    return redacted[:max_length]


def _read_retry_after(headers: Any) -> float | None:
    value = str(headers.get("retry-after", "") or "").strip()
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def _is_transient_reference_fetch_error(
    provider: str,
    status_code: int,
    error_type: str,
    message: str,
) -> bool:
    """判断服务商是否临时无法连接参考图地址。

    参数：
        provider: 返回错误的服务商名称。
        status_code: 服务商返回的 HTTP 状态码。
        error_type: 服务商返回的细分错误类型。
        message: 已脱敏的服务商错误摘要。

    返回值：
        EasyRouter 明确返回参考图连接不可达时为真，否则为假。
    """
    if (
        provider.strip().lower() != "easyrouter"
        or status_code != 400
        or error_type.strip().lower() != "upstream_error"
    ):
        return False
    normalized_message = message.upper()
    return any(
        marker in normalized_message
        for marker in _TRANSIENT_REFERENCE_FETCH_MARKERS
    )


def provider_error_from_status(
    provider: str,
    status_code: int,
    message: str,
    *,
    headers: Any | None = None,
    error_type: str = "",
    request_id: str = "",
    response_bytes: int | None = None,
) -> ProviderError:
    """将服务商 HTTP 错误转换成统一错误。

    参数：
        provider: 发生错误的服务商名称。
        status_code: HTTP 响应状态码。
        message: 已脱敏且限制长度的错误摘要。
        headers: 用于读取 Retry-After 的响应头。
        error_type: 服务商返回的细分错误类型。
        request_id: 服务商响应中的请求标识。
        response_bytes: 服务商错误响应正文的字节数。

    返回值：
        可供路由层判断重试、兜底和熔断的统一错误。
    """
    normalized_error_type = error_type.strip().lower()
    image_errors = {
        "invalid_image",
        "image_too_large",
        "image_too_small",
        "unsupported_image_format",
        "image_not_found",
        "image_download_failed",
    }
    if _is_transient_reference_fetch_error(
        provider,
        status_code,
        normalized_error_type,
        message,
    ):
        category, retryable = ErrorCategory.UPSTREAM_UNAVAILABLE, True
    elif normalized_error_type in image_errors:
        category, retryable = ErrorCategory.INVALID_ASSET, False
    elif normalized_error_type in {"content_policy_violation", "refusal"}:
        category, retryable = ErrorCategory.CONTENT_POLICY, False
    elif status_code == 401:
        category, retryable = ErrorCategory.AUTHENTICATION, True
    elif status_code == 402:
        category, retryable = ErrorCategory.BILLING, True
    elif status_code == 403:
        category, retryable = ErrorCategory.PERMISSION, True
    elif status_code == 429:
        category, retryable = ErrorCategory.RATE_LIMIT, True
    elif status_code in {408, 500, 502, 503, 504, 524, 529}:
        category, retryable = ErrorCategory.UPSTREAM_UNAVAILABLE, True
    elif status_code in {400, 404, 413, 422}:
        category, retryable = ErrorCategory.INVALID_REQUEST, False
    else:
        category, retryable = ErrorCategory.UNKNOWN, False

    return ProviderError(
        provider=provider,
        category=category,
        message=sanitize_provider_error_message(message),
        status_code=status_code,
        retryable=retryable,
        retry_after_seconds=_read_retry_after(headers or {}),
        request_id=request_id,
        provider_error_type=normalized_error_type,
        response_bytes=response_bytes,
    )


def provider_error_from_httpx(provider: str, error: httpx.HTTPError) -> ProviderError:
    """将 HTTPX 异常转换成统一服务商错误。

    参数：
        provider: 发生异常的服务商名称。
        error: HTTPX 抛出的请求或响应异常。

    返回值：
        包含故障分类和熔断计数规则的统一错误。
    """
    if isinstance(error, httpx.PoolTimeout):
        return ProviderError(
            provider=provider,
            category=ErrorCategory.LOCAL_CAPACITY,
            message="本地 HTTP 连接池暂时不可用。",
            retryable=True,
            counts_toward_circuit=False,
            cause=error,
        )
    if isinstance(error, (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout)):
        category = ErrorCategory.TIMEOUT
    elif isinstance(error, httpx.ConnectError):
        category = ErrorCategory.CONNECTION
    elif isinstance(error, httpx.RemoteProtocolError):
        category = ErrorCategory.UPSTREAM_UNAVAILABLE
    elif isinstance(error, httpx.DecodingError):
        category = ErrorCategory.INVALID_RESPONSE
    elif isinstance(error, httpx.InvalidURL):
        return ProviderError(
            provider=provider,
            category=ErrorCategory.INVALID_REQUEST,
            message="服务商请求地址无效。",
            retryable=False,
            counts_toward_circuit=False,
            cause=error,
        )
    else:
        category = ErrorCategory.CONNECTION

    return ProviderError(
        provider=provider,
        category=category,
        message="服务商 HTTP 请求失败。",
        retryable=True,
        cause=error,
    )
