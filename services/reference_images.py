from __future__ import annotations

import base64
import binascii
from contextlib import contextmanager
import logging
from dataclasses import replace
import time
from typing import Iterator
from uuid import uuid4

from services.domain.errors import ErrorCategory, ProviderError
from services.domain.requests import (
    GenerateImageRequest,
    ReferenceImageInfo,
    UploadedFileInfo,
)
from services.http import (
    AssetFetchError,
    build_asset_fetcher,
    detect_image_content_type,
)
from services.oss_service import (
    TemporaryReferenceCleanupResult,
    TemporaryReferenceObject,
    TemporaryReferenceStore,
)
from services.settings import AppSettings

logger = logging.getLogger(__name__)


def _validate_reference_body(
    body: bytes,
    settings: AppSettings,
) -> str:
    """校验参考图大小和文件头，并返回可信 MIME 类型。"""
    if len(body) > settings.http.asset_max_bytes:
        raise AssetFetchError("资源大小超过限制。")
    mime_type = detect_image_content_type(body)
    if not mime_type:
        raise AssetFetchError("资源内容不是受支持的图片格式。")
    return mime_type


def _read_uploaded_reference(
    uploaded_file: UploadedFileInfo,
    settings: AppSettings,
) -> tuple[bytes, str]:
    """读取上传文件并以文件头确认图片类型。"""
    storage = uploaded_file.storage
    storage.stream.seek(0)
    body = storage.read()
    storage.stream.seek(0)
    return body, _validate_reference_body(body, settings)


def _release_uploaded_reference(uploaded_file: UploadedFileInfo) -> None:
    """关闭上传文件流并解除大块本地数据引用。"""
    storage = uploaded_file.storage
    try:
        close = getattr(storage, "close", None)
        if callable(close):
            close()
        else:
            stream_close = getattr(
                getattr(storage, "stream", None),
                "close",
                None,
            )
            if callable(stream_close):
                stream_close()
    except Exception as exc:
        logger.warning(
            "image.reference.local.release.failed: %s",
            {"errorType": type(exc).__name__},
        )
    uploaded_file.storage = None


def _read_url_reference(
    file_url: str,
    settings: AppSettings,
    asset_fetcher,
) -> tuple[bytes, str]:
    """解码 Data URL 或下载外部参考图并返回可信图片类型。"""
    try:
        decoded_data_url = _decode_data_url(file_url)
    except (binascii.Error, ValueError) as exc:
        raise AssetFetchError("参考图片 Data URL 无法解码。") from exc
    if decoded_data_url:
        _, body = decoded_data_url
        return body, _validate_reference_body(body, settings)
    fetched_asset = asset_fetcher.fetch(file_url)
    return fetched_asset.body, fetched_asset.content_type


@contextmanager
def stage_reference_images(
    request_data: GenerateImageRequest,
    settings: AppSettings,
) -> Iterator[GenerateImageRequest]:
    """暂存一批生成参考图，并在全部模型调用结束后主动清理。

    参数：
        request_data: 包含外部 URL、Data URL 或上传文件的图片生成请求。
        settings: 包含下载限制和临时 OSS 配置的应用设置。

    返回值：
        上下文中返回只包含短期签名 URL 的图片生成请求。
    """
    if not request_data.file_urls and not request_data.files:
        yield request_data
        return

    store = TemporaryReferenceStore(settings)
    batch_id = uuid4().hex
    uploaded_objects: list[TemporaryReferenceObject] = []
    references: list[ReferenceImageInfo] = []
    upload_start = time.perf_counter()
    asset_fetcher = build_asset_fetcher(settings) if request_data.file_urls else None

    try:
        try:
            for uploaded_file in request_data.files:
                body = b""
                try:
                    body, mime_type = _read_uploaded_reference(
                        uploaded_file,
                        settings,
                    )
                    uploaded = store.upload(body, mime_type, batch_id)
                    uploaded_objects.append(uploaded)
                    references.append(
                        ReferenceImageInfo(
                            url=uploaded.signed_url,
                            mime_type=uploaded.mime_type,
                        )
                    )
                finally:
                    body = b""
                    _release_uploaded_reference(uploaded_file)
            for file_url in request_data.file_urls:
                if asset_fetcher is None:
                    raise RuntimeError("参考图片下载器尚未初始化。")
                body, mime_type = _read_url_reference(
                    file_url,
                    settings,
                    asset_fetcher,
                )
                uploaded = store.upload(body, mime_type, batch_id)
                uploaded_objects.append(uploaded)
                references.append(
                    ReferenceImageInfo(
                        url=uploaded.signed_url,
                        mime_type=uploaded.mime_type,
                    )
                )
                body = b""
        except AssetFetchError as exc:
            logger.warning(
                "image.reference.oss.upload.failed: %s",
                {
                    "requestId": request_data.request_id,
                    "errorType": type(exc).__name__,
                },
            )
            raise ProviderError(
                provider="reference",
                category=ErrorCategory.INVALID_ASSET,
                message="参考图片下载失败或格式不受支持。",
                retryable=False,
                counts_toward_circuit=False,
                cause=exc,
            ) from exc
        except Exception as exc:
            logger.warning(
                "image.reference.oss.upload.failed: %s",
                {
                    "requestId": request_data.request_id,
                    "errorType": type(exc).__name__,
                },
            )
            raise

        logger.info(
            "image.reference.oss.upload.completed: %s",
            {
                "requestId": request_data.request_id,
                "referenceCount": len(uploaded_objects),
                "totalBytes": sum(
                    item.content_length for item in uploaded_objects
                ),
                "elapsedMs": round(
                    (time.perf_counter() - upload_start) * 1000,
                    2,
                ),
            },
        )
        yield replace(
            request_data,
            file_urls=[],
            files=[],
            reference_images=references,
        )
    finally:
        for uploaded_file in request_data.files:
            _release_uploaded_reference(uploaded_file)
        if uploaded_objects:
            cleanup_start = time.perf_counter()
            try:
                cleanup = store.delete_many(uploaded_objects)
            except Exception as exc:
                logger.warning(
                    "image.reference.oss.cleanup.failed: %s",
                    {
                        "requestId": request_data.request_id,
                        "errorType": type(exc).__name__,
                    },
                )
                cleanup = TemporaryReferenceCleanupResult(
                    attempted_count=len(uploaded_objects),
                    deleted_count=0,
                    failed_count=len(uploaded_objects),
                )
            logger.info(
                "image.reference.oss.cleanup.completed: %s",
                {
                    "requestId": request_data.request_id,
                    "attemptedCount": cleanup.attempted_count,
                    "deletedCount": cleanup.deleted_count,
                    "failedCount": cleanup.failed_count,
                    "elapsedMs": round(
                        (time.perf_counter() - cleanup_start) * 1000,
                        2,
                    ),
                },
            )


def _decode_data_url(value: str) -> tuple[str, bytes] | None:
    prefix, separator, encoded = str(value or "").partition(",")
    if not separator or not prefix.startswith("data:") or ";base64" not in prefix:
        return None
    mime_type = (
        prefix.removeprefix("data:").split(";", 1)[0].strip()
        or "application/octet-stream"
    )
    return mime_type, base64.b64decode(encoded, validate=True)
