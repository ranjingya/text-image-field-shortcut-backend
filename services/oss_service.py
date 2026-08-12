from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import mimetypes
from pathlib import Path
import re
from uuid import UUID, uuid4

import alibabacloud_oss_v2 as oss

from services.response_normalizer import NormalizedGeneratedAsset
from services.http import build_asset_fetcher
from services.settings import AppSettings

logger = logging.getLogger(__name__)


@dataclass
class OssUploadResult:
    bucket_name: str
    bucket_prefix: str
    endpoint: str
    region: str
    object_key: str
    object_url: str
    etag: str
    request_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "bucketName": self.bucket_name,
            "bucketPrefix": self.bucket_prefix,
            "endpoint": self.endpoint,
            "region": self.region,
            "objectKey": self.object_key,
            "objectUrl": self.object_url,
            "etag": self.etag,
            "requestId": self.request_id,
        }


@dataclass(frozen=True)
class TemporaryReferenceObject:
    """记录一张临时参考图在 OSS 中的对象信息。"""

    object_key: str
    signed_url: str
    mime_type: str
    content_length: int


@dataclass(frozen=True)
class TemporaryReferenceCleanupResult:
    """记录一批临时参考图的主动删除结果。"""

    attempted_count: int
    deleted_count: int
    failed_count: int


class TemporaryReferenceStore:
    """管理参考图的私有临时上传、签名访问和主动删除。"""

    def __init__(self, settings: AppSettings, client: oss.Client | None = None) -> None:
        """创建临时参考图存储器。

        参数：
            settings: 包含 OSS Bucket、临时前缀和签名时长的应用配置。
            client: 可选的 OSS 客户端，测试时用于注入替身。

        返回值：
            无。
        """
        self._settings = settings
        self._client = client or create_oss_client(settings)
        self._presign_client = (
            create_oss_cname_client(settings)
            if settings.oss.temporary_custom_domain
            else self._client
        )

    def upload(
        self,
        body: bytes,
        mime_type: str,
        batch_id: str,
    ) -> TemporaryReferenceObject:
        """上传一张私有参考图并生成短期 GET 签名 URL。

        参数：
            body: 已完成下载和校验的参考图字节。
            mime_type: 参考图的 MIME 类型。
            batch_id: 服务端生成的随机批次标识，用于隔离对象目录。

        返回值：
            包含对象键、签名 URL、类型和大小的临时对象信息。
        """
        normalized_batch_id = _normalize_temporary_batch_id(batch_id)
        extension = _guess_safe_extension(mime_type)
        file_name = f"{normalized_batch_id}/{uuid4().hex}{extension}"
        object_key = build_object_key(
            self._settings.oss.temporary_reference_prefix,
            file_name,
        )
        self._client.put_object(
            oss.PutObjectRequest(
                bucket=self._settings.oss.bucket_name,
                key=object_key,
                body=body,
                content_type=mime_type,
                acl="private",
            )
        )
        try:
            presigned = self._presign_client.presign(
                oss.GetObjectRequest(
                    bucket=self._settings.oss.bucket_name,
                    key=object_key,
                ),
                expires=timedelta(
                    seconds=self._settings.oss.temporary_url_ttl_seconds
                ),
            )
        except Exception:
            try:
                self._client.delete_object(
                    oss.DeleteObjectRequest(
                        bucket=self._settings.oss.bucket_name,
                        key=object_key,
                    )
                )
            except Exception as cleanup_error:
                logger.warning(
                    "image.reference.oss.presign.cleanup.failed: %s",
                    {"errorType": type(cleanup_error).__name__},
                )
            raise
        return TemporaryReferenceObject(
            object_key=object_key,
            signed_url=presigned.url,
            mime_type=mime_type,
            content_length=len(body),
        )

    def delete_many(
        self,
        objects: list[TemporaryReferenceObject],
    ) -> TemporaryReferenceCleanupResult:
        """尽力删除一批临时参考图，不因单个对象失败中断清理。

        参数：
            objects: 当前批次已经成功上传的临时参考图列表。

        返回值：
            包含尝试数、成功数和失败数的清理汇总。
        """
        deleted_count = 0
        failed_count = 0
        for item in objects:
            try:
                self._client.delete_object(
                    oss.DeleteObjectRequest(
                        bucket=self._settings.oss.bucket_name,
                        key=item.object_key,
                    )
                )
                deleted_count += 1
            except Exception as exc:
                failed_count += 1
                logger.warning(
                    "image.reference.oss.cleanup.item.failed: %s",
                    {
                        "errorType": type(exc).__name__,
                    },
                )
        return TemporaryReferenceCleanupResult(
            attempted_count=len(objects),
            deleted_count=deleted_count,
            failed_count=failed_count,
        )


def build_datetime_file_name(extension: str = ".png") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f")
    return f"{timestamp}{extension if extension.startswith('.') else f'.{extension}'}"


def build_object_key(bucket_prefix: str, file_name: str) -> str:
    clean_prefix = str(bucket_prefix or "").strip().strip("/")
    return f"{clean_prefix}/{file_name}" if clean_prefix else file_name


def build_object_url(bucket_name: str, endpoint: str, object_key: str) -> str:
    return f"https://{bucket_name}.{endpoint}/{object_key}"


def _normalize_temporary_batch_id(batch_id: str) -> str:
    """只接受服务端生成的 UUID，避免把业务标识写入 OSS 对象键。"""
    return UUID(str(batch_id)).hex


def _guess_safe_extension(mime_type: str) -> str:
    """根据 MIME 类型生成不含路径字符的文件扩展名。"""
    extension = mimetypes.guess_extension(str(mime_type or "").lower()) or ".bin"
    normalized = re.sub(r"[^a-z0-9.]", "", extension.lower())
    return normalized if normalized.startswith(".") and len(normalized) <= 10 else ".bin"


def create_oss_client(settings: AppSettings) -> oss.Client:
    credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
    cfg = oss.config.load_default()
    cfg.credentials_provider = credentials_provider
    cfg.region = settings.oss.region
    cfg.endpoint = settings.oss.endpoint
    return oss.Client(cfg)


def create_oss_cname_client(settings: AppSettings) -> oss.Client:
    """创建仅用于自定义域名签名的 OSS 客户端。

    参数：
        settings: 包含 OSS 地域、凭证来源和临时自定义域名的应用配置。

    返回值：
        使用 CNAME 寻址方式的 OSS 客户端。
    """
    custom_domain = settings.oss.temporary_custom_domain
    if not custom_domain:
        raise ValueError("OSS_TEMP_CUSTOM_DOMAIN 尚未配置。")
    credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
    cfg = oss.config.load_default()
    cfg.credentials_provider = credentials_provider
    cfg.region = settings.oss.region
    cfg.endpoint = custom_domain
    cfg.use_cname = True
    return oss.Client(cfg)


def _looks_like_timestamp_name(file_name: str) -> bool:
    stem = Path(str(file_name or "")).stem
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}[_-]\d{2}[-_]\d{2}[-_]\d{2}", stem) or re.match(r"^\d{8}[_-]?\d{6}", stem))


def _resolve_file_name(asset: NormalizedGeneratedAsset) -> str:
    original_name = Path(asset.file_name or "").name
    suffix = Path(original_name).suffix or ".bin"
    if original_name and _looks_like_timestamp_name(original_name):
        return original_name
    return build_datetime_file_name(suffix)


def _resolve_asset_bytes(settings: AppSettings, asset: NormalizedGeneratedAsset) -> bytes:
    if asset.source_kind == "bytes":
        body = asset.payload if isinstance(asset.payload, bytes) else bytes(asset.payload)
        return body
    if asset.source_kind == "text":
        body = str(asset.payload).encode("utf-8")
        return body
    if asset.source_kind == "url":
        return build_asset_fetcher(settings).fetch(str(asset.payload)).body
    raise RuntimeError(f"Unsupported asset source kind: {asset.source_kind}")


def upload_asset_to_oss(settings: AppSettings, asset: NormalizedGeneratedAsset) -> OssUploadResult:
    file_name = _resolve_file_name(asset)
    object_key = build_object_key(settings.oss.bucket_prefix, file_name)
    body = _resolve_asset_bytes(settings, asset)

    logger.debug(
        "gemini.backend.oss.upload.start: %s",
        {
            "bucketName": settings.oss.bucket_name,
            "endpoint": settings.oss.endpoint,
            "objectKey": object_key,
            "assetType": asset.asset_type,
            "sourceKind": asset.source_kind,
            "mimeType": asset.mime_type,
            "bodyLength": len(body),
        },
    )

    client = create_oss_client(settings)

    result = client.put_object(
        oss.PutObjectRequest(
            bucket=settings.oss.bucket_name,
            key=object_key,
            body=body,
            content_type=asset.mime_type,
        )
    )

    upload_result = OssUploadResult(
        bucket_name=settings.oss.bucket_name,
        bucket_prefix=settings.oss.bucket_prefix,
        endpoint=settings.oss.endpoint,
        region=settings.oss.region,
        object_key=object_key,
        object_url=build_object_url(settings.oss.bucket_name, settings.oss.endpoint, object_key),
        etag=getattr(result, "etag", ""),
        request_id=getattr(result, "request_id", ""),
    )

    logger.debug(
        "gemini.backend.oss.upload.success: %s",
        {
            "bucketName": upload_result.bucket_name,
            "objectKey": upload_result.object_key,
            "objectUrl": upload_result.object_url,
            "etag": upload_result.etag,
            "requestId": upload_result.request_id,
            "bodyLength": len(body),
        },
    )

    return upload_result
