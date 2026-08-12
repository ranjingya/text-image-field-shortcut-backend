from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit


def normalize_endpoint(endpoint: str) -> str:
    normalized = str(endpoint or "").strip()
    return normalized.removeprefix("https://").removeprefix("http://").rstrip("/")


def endpoint_to_region(endpoint: str) -> str:
    return normalize_endpoint(endpoint).replace(".aliyuncs.com", "").removeprefix("oss-")


def normalize_custom_domain(value: str) -> str:
    """规范化 OSS 自定义域名，仅接受不带路径的 HTTPS 地址。"""
    normalized = str(value or "").strip().rstrip("/")
    if not normalized:
        return ""
    parsed = urlsplit(normalized)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("OSS_TEMP_CUSTOM_DOMAIN 端口格式不正确。") from exc
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or port is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "OSS_TEMP_CUSTOM_DOMAIN 必须是无路径、无端口的 HTTPS 域名。"
        )
    return f"https://{parsed.hostname.lower()}"


@dataclass
class OssSettings:
    endpoint: str
    region: str
    bucket_name: str
    bucket_prefix: str
    temporary_reference_prefix: str = "temp-references"
    temporary_url_ttl_seconds: int = 3600
    temporary_custom_domain: str = ""


@dataclass(frozen=True)
class HttpTimeoutSettings:
    connect: float = 10.0
    read: float = 300.0
    write: float = 60.0
    pool: float = 5.0


@dataclass(frozen=True)
class HttpClientSettings:
    timeout: HttpTimeoutSettings = field(default_factory=HttpTimeoutSettings)
    max_connections: int = 16
    max_keepalive_connections: int = 8
    max_redirects: int = 0
    trust_env: bool = False


@dataclass(frozen=True)
class HttpSettings:
    provider: HttpClientSettings = field(default_factory=HttpClientSettings)
    asset: HttpClientSettings = field(
        default_factory=lambda: HttpClientSettings(
            timeout=HttpTimeoutSettings(connect=10.0, read=120.0, write=60.0, pool=5.0),
            max_redirects=3,
        )
    )
    auth: HttpClientSettings = field(
        default_factory=lambda: HttpClientSettings(
            timeout=HttpTimeoutSettings(connect=2.0, read=5.0, write=5.0, pool=1.0),
        )
    )
    notification: HttpClientSettings = field(
        default_factory=lambda: HttpClientSettings(
            timeout=HttpTimeoutSettings(connect=2.0, read=3.0, write=3.0, pool=1.0),
        )
    )
    asset_max_bytes: int = 52_428_800


@dataclass(frozen=True)
class RoutingSettings:
    provider_timeout_seconds: float = 300.0
    primary_max_attempts: int = 1
    fallback_max_attempts: int = 1
    primary_empty_response_retry_count: int = 1


@dataclass(frozen=True)
class ImageGenerationSettings:
    max_count: int = 5
    max_concurrency: int = 5
    queue_timeout_seconds: float = 420.0
    trim_memory_after_request: bool = False
    trim_rss_threshold_bytes: int = 512 * 1024 * 1024
    trim_cooldown_seconds: float = 60.0


@dataclass(frozen=True)
class CircuitBreakerSettings:
    failure_threshold: int = 3
    open_seconds: float = 60.0
    max_open_seconds: float = 900.0
    state_ttl_seconds: int = 86_400


@dataclass(frozen=True)
class AlertSettings:
    enabled: bool = False
    webhook_url: str = ""
    keyword: str = ""
    service_name: str = "text-image-field-shortcut-backend"
    environment: str = "production"
    fallback_window_seconds: int = 300
    fallback_threshold: int = 3
    fallback_cooldown_seconds: int = 900
    critical_cooldown_seconds: int = 60
    recovery_success_threshold: int = 3
    incident_ttl_seconds: int = 86_400


@dataclass
class AppSettings:
    oss: OssSettings
    http: HttpSettings = field(default_factory=HttpSettings)
    provider_config_path: str = "config/providers.json"
    fallback_enabled: bool = False
    routing: RoutingSettings = field(default_factory=RoutingSettings)
    image_generation: ImageGenerationSettings = field(
        default_factory=ImageGenerationSettings
    )
    circuit: CircuitBreakerSettings = field(default_factory=CircuitBreakerSettings)
    alert: AlertSettings = field(default_factory=AlertSettings)

def get_app_settings() -> AppSettings:
    """加载应用配置。

    返回值：
        从环境变量解析得到的应用配置。
    """
    oss_endpoint = normalize_endpoint(os.getenv("OSS_ENDPOINT", ""))

    max_connections = _read_positive_int("HTTP_MAX_CONNECTIONS", 16)
    max_keepalive_connections = _read_positive_int("HTTP_MAX_KEEPALIVE_CONNECTIONS", 8)
    provider_timeout = HttpTimeoutSettings(
        connect=_read_positive_float("PROVIDER_CONNECT_TIMEOUT_SECONDS", 10.0),
        read=_read_positive_float("PROVIDER_READ_TIMEOUT_SECONDS", 300.0),
        write=_read_positive_float("PROVIDER_WRITE_TIMEOUT_SECONDS", 60.0),
        pool=_read_positive_float("PROVIDER_POOL_TIMEOUT_SECONDS", 5.0),
    )
    trust_env = _read_bool("HTTP_TRUST_ENV", False)

    settings = AppSettings(
        oss=OssSettings(
            endpoint=oss_endpoint,
            region=endpoint_to_region(oss_endpoint),
            bucket_name=os.getenv("OSS_BUCKET_NAME", "").strip(),
            bucket_prefix=_normalize_bucket_prefix(
                os.getenv("OSS_BUCKET_FOLDER_PREFIX", "")
            ),
            temporary_reference_prefix=_normalize_bucket_prefix(
                os.getenv("OSS_TEMP_FOLDER_PREFIX", "temp-references")
            ),
            temporary_url_ttl_seconds=_read_positive_int(
                "OSS_TEMP_URL_TTL_SECONDS", 3600
            ),
            temporary_custom_domain=normalize_custom_domain(
                os.getenv("OSS_TEMP_CUSTOM_DOMAIN", "")
            ),
        ),
        http=HttpSettings(
            provider=HttpClientSettings(
                timeout=provider_timeout,
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
                trust_env=trust_env,
            ),
            asset=HttpClientSettings(
                timeout=HttpTimeoutSettings(
                    connect=_read_positive_float("REFERENCE_CONNECT_TIMEOUT_SECONDS", 10.0),
                    read=_read_positive_float("REFERENCE_READ_TIMEOUT_SECONDS", 120.0),
                    write=provider_timeout.write,
                    pool=provider_timeout.pool,
                ),
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
                max_redirects=_read_non_negative_int("REFERENCE_MAX_REDIRECTS", 3),
                trust_env=trust_env,
            ),
            auth=HttpClientSettings(
                timeout=HttpTimeoutSettings(
                    connect=_read_positive_float("AUTH_VERIFY_CONNECT_TIMEOUT_SECONDS", 2.0),
                    read=_read_positive_float("AUTH_VERIFY_READ_TIMEOUT_SECONDS", 5.0),
                    write=5.0,
                    pool=_read_positive_float("AUTH_VERIFY_POOL_TIMEOUT_SECONDS", 1.0),
                ),
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
                trust_env=trust_env,
            ),
            notification=HttpClientSettings(
                timeout=HttpTimeoutSettings(
                    connect=_read_positive_float("FEISHU_CONNECT_TIMEOUT_SECONDS", 2.0),
                    read=_read_positive_float("FEISHU_READ_TIMEOUT_SECONDS", 3.0),
                    write=3.0,
                    pool=_read_positive_float("FEISHU_POOL_TIMEOUT_SECONDS", 1.0),
                ),
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
                trust_env=trust_env,
            ),
            asset_max_bytes=_read_positive_int("REFERENCE_MAX_BYTES", 52_428_800),
        ),
        fallback_enabled=_read_bool("FALLBACK_ENABLED", False),
        routing=RoutingSettings(
            provider_timeout_seconds=_read_positive_float(
                "PROVIDER_REQUEST_TIMEOUT_SECONDS", 300.0
            ),
            primary_max_attempts=_read_positive_int("PRIMARY_MAX_ATTEMPTS", 1),
            fallback_max_attempts=_read_positive_int("FALLBACK_MAX_ATTEMPTS", 1),
            primary_empty_response_retry_count=_read_non_negative_int(
                "PRIMARY_EMPTY_RESPONSE_RETRY_COUNT", 1
            ),
        ),
        image_generation=ImageGenerationSettings(
            max_count=_read_positive_int("IMAGE_GENERATION_MAX_COUNT", 5),
            max_concurrency=_read_positive_int(
                "IMAGE_GENERATION_MAX_CONCURRENCY", 5
            ),
            queue_timeout_seconds=_read_positive_float(
                "IMAGE_GENERATION_QUEUE_TIMEOUT_SECONDS", 420.0
            ),
            trim_memory_after_request=_read_bool(
                "MEMORY_TRIM_AFTER_IMAGE_REQUEST", False
            ),
            trim_rss_threshold_bytes=(
                _read_positive_int("MEMORY_TRIM_RSS_THRESHOLD_MB", 512)
                * 1024
                * 1024
            ),
            trim_cooldown_seconds=_read_positive_float(
                "MEMORY_TRIM_COOLDOWN_SECONDS", 60.0
            ),
        ),
        circuit=CircuitBreakerSettings(
            failure_threshold=_read_positive_int("CIRCUIT_FAILURE_THRESHOLD", 3),
            open_seconds=_read_positive_float("CIRCUIT_OPEN_SECONDS", 60.0),
            max_open_seconds=_read_positive_float(
                "CIRCUIT_MAX_OPEN_SECONDS", 900.0
            ),
            state_ttl_seconds=_read_positive_int(
                "CIRCUIT_STATE_TTL_SECONDS", 86_400
            ),
        ),
        alert=AlertSettings(
            enabled=_read_bool("FEISHU_ALERT_ENABLED", False),
            webhook_url=os.getenv("FEISHU_ALERT_WEBHOOK_URL", "").strip(),
            keyword=os.getenv("FEISHU_ALERT_KEYWORD", "").strip(),
            service_name=os.getenv(
                "SERVICE_NAME", "text-image-field-shortcut-backend"
            ).strip(),
            environment=os.getenv("APP_ENV", "production").strip(),
            fallback_window_seconds=_read_positive_int(
                "FALLBACK_ALERT_WINDOW_SECONDS", 300
            ),
            fallback_threshold=_read_positive_int(
                "FALLBACK_ALERT_THRESHOLD", 3
            ),
            fallback_cooldown_seconds=_read_positive_int(
                "FALLBACK_ALERT_COOLDOWN_SECONDS", 900
            ),
            critical_cooldown_seconds=_read_positive_int(
                "CRITICAL_ALERT_COOLDOWN_SECONDS", 60
            ),
            recovery_success_threshold=_read_positive_int(
                "PRIMARY_RECOVERY_SUCCESS_THRESHOLD", 3
            ),
            incident_ttl_seconds=_read_positive_int(
                "ALERT_INCIDENT_TTL_SECONDS", 86_400
            ),
        ),
    )
    validate_app_settings(settings)
    return settings


def validate_app_settings(settings: AppSettings) -> None:
    """校验需要组合判断的应用配置。

    参数：
        settings: 已完成环境变量解析的应用配置。

    返回值：
        无；配置不完整时抛出明确异常。
    """
    if settings.alert.enabled and (
        not settings.alert.webhook_url or not settings.alert.keyword
    ):
        raise ValueError(
            "FEISHU_ALERT_ENABLED=true 时必须配置 Webhook URL 和关键词。"
        )
    temporary_prefix = settings.oss.temporary_reference_prefix
    permanent_prefix = settings.oss.bucket_prefix
    if not temporary_prefix:
        raise ValueError("OSS_TEMP_FOLDER_PREFIX 不能为空。")
    if permanent_prefix and (
        temporary_prefix == permanent_prefix
        or temporary_prefix.startswith(f"{permanent_prefix}/")
        or permanent_prefix.startswith(f"{temporary_prefix}/")
    ):
        raise ValueError(
            "OSS_TEMP_FOLDER_PREFIX 不能与 OSS_BUCKET_FOLDER_PREFIX 重叠。"
        )


def _normalize_bucket_prefix(value: str) -> str:
    """规范化 OSS 对象前缀，便于安全比较生命周期作用范围。"""
    return str(value or "").strip().strip("/")


def _read_positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _read_positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _read_non_negative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be zero or greater.")
    return value


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
