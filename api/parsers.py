from __future__ import annotations

import logging
from typing import Any

from services.domain.requests import (
    GenerateImageRequest,
    RequestValidationError,
    UnderstandImageRequest,
    UploadedFileInfo,
)

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_SIZES = ("1K", "2K", "4K")
SUPPORTED_ASPECT_RATIOS = (
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
)
DEFAULT_IMAGE_SIZE = "1K"
DEFAULT_IMAGE_COUNT = 1
MAX_REFERENCE_IMAGE_COUNT = 14


def _stringify(value: Any, default: str = "") -> str:
    return str(value or default).strip()


def _normalize_url_values(*values: Any) -> list[str]:
    urls: list[str] = []
    for value in values:
        if isinstance(value, list):
            urls.extend(_normalize_url_values(*value))
            continue
        text = _stringify(value)
        if text:
            urls.append(text)
    return urls


def _resolve_request_value(
    is_json_request: bool,
    payload: dict[str, Any],
    form_data: Any,
    key: str,
    default: str = "",
) -> str:
    value = payload.get(key) if is_json_request else form_data.get(key)
    return _stringify(value, default=default)


def _normalize_image_size(value: Any) -> str:
    normalized = _stringify(value, DEFAULT_IMAGE_SIZE).upper()
    if normalized not in SUPPORTED_IMAGE_SIZES:
        raise RequestValidationError(
            f"Unsupported imageSize: {normalized}. "
            f"Supported values: {', '.join(SUPPORTED_IMAGE_SIZES)}."
        )
    return normalized


def _normalize_aspect_ratio(value: Any) -> str:
    normalized = _stringify(value).replace(" ", "")
    if normalized not in SUPPORTED_ASPECT_RATIOS:
        raise RequestValidationError(
            f"Unsupported aspectRatio: {normalized}. "
            f"Supported values: {', '.join(SUPPORTED_ASPECT_RATIOS)}."
        )
    return normalized


def _normalize_image_count(value: Any) -> int:
    """解析客户端请求的图片数量。

    参数：
        value: JSON 或表单中读取到的原始图片数量。

    返回值：
        大于零的整数图片数量；未传入时返回默认值。
    """
    if value is None or value == "":
        return DEFAULT_IMAGE_COUNT
    if isinstance(value, bool):
        raise RequestValidationError("imageCount must be a positive integer.")
    if isinstance(value, int):
        image_count = value
    elif isinstance(value, str) and value.strip().isdigit():
        image_count = int(value.strip())
    else:
        raise RequestValidationError("imageCount must be a positive integer.")
    if image_count <= 0:
        raise RequestValidationError("imageCount must be a positive integer.")
    return image_count


def _validate_reference_count(
    file_urls: list[str], files: list[UploadedFileInfo]
) -> None:
    reference_count = len(file_urls) + len(files)
    if reference_count > MAX_REFERENCE_IMAGE_COUNT:
        raise RequestValidationError(
            f"Too many reference images: {reference_count}. "
            f"The current limit is {MAX_REFERENCE_IMAGE_COUNT}."
        )


def _collect_uploaded_files(flask_request: Any) -> list[UploadedFileInfo]:
    files: list[UploadedFileInfo] = []
    for field_name in flask_request.files:
        for file_storage in flask_request.files.getlist(field_name):
            files.append(
                UploadedFileInfo(
                    field_name=field_name,
                    file_name=_stringify(getattr(file_storage, "filename", "")),
                    content_type=_stringify(
                        getattr(file_storage, "content_type", "")
                    ),
                    content_length=int(
                        getattr(file_storage, "content_length", 0) or 0
                    ),
                    storage=file_storage,
                )
            )
    return files


def parse_generate_image_request(flask_request: Any) -> GenerateImageRequest:
    """解析图片生成接口请求。

    参数：
        flask_request: 当前 Flask 请求对象。

    返回值：
        统一的图片生成请求对象。
    """
    is_json_request = bool(flask_request.is_json)
    payload = flask_request.get_json(silent=True) or {} if is_json_request else {}
    form_data = flask_request.form
    file_urls = _normalize_url_values(
        payload.get("fileUrl"),
        payload.get("fileUrls"),
        form_data.get("fileUrl"),
        form_data.getlist("fileUrls"),
    )
    files = _collect_uploaded_files(flask_request)
    _validate_reference_count(file_urls, files)

    if files and file_urls:
        input_type = "mixed"
    elif files:
        input_type = "file_stream"
    elif file_urls:
        input_type = "file_url"
    else:
        input_type = "empty"

    raw_aspect_ratio = (
        payload.get("aspectRatio")
        if is_json_request
        else form_data.get("aspectRatio")
    )
    raw_image_count = (
        payload.get("imageCount")
        if is_json_request
        else form_data.get("imageCount")
    )
    parsed_request = GenerateImageRequest(
        request_id=_resolve_request_value(
            is_json_request, payload, form_data, "requestId"
        ),
        prompt=_resolve_request_value(
            is_json_request, payload, form_data, "prompt"
        ),
        model=_resolve_request_value(is_json_request, payload, form_data, "model"),
        aspect_ratio=(
            _normalize_aspect_ratio(raw_aspect_ratio)
            if _stringify(raw_aspect_ratio)
            else None
        ),
        image_size=_normalize_image_size(
            payload.get("imageSize")
            if is_json_request
            else form_data.get("imageSize")
        ),
        input_type=input_type,
        file_urls=file_urls,
        files=files,
        image_count=_normalize_image_count(raw_image_count),
    )
    logger.debug(
        "api.request.generate.parsed: %s",
        parsed_request.to_dict(),
    )
    return parsed_request


def parse_understand_image_request(flask_request: Any) -> UnderstandImageRequest:
    """解析图片理解接口请求。

    参数：
        flask_request: 当前 Flask 请求对象。

    返回值：
        统一的图片理解请求对象。
    """
    payload = flask_request.get_json(silent=True) or {}
    file_urls = _normalize_url_values(
        payload.get("fileUrl"), payload.get("fileUrls")
    )
    if len(file_urls) > MAX_REFERENCE_IMAGE_COUNT:
        raise RequestValidationError(
            f"Too many reference images: {len(file_urls)}. "
            f"The current limit is {MAX_REFERENCE_IMAGE_COUNT}."
        )

    parsed_request = UnderstandImageRequest(
        request_id=_stringify(payload.get("requestId")),
        prompt=_stringify(payload.get("prompt")),
        model=_stringify(payload.get("model")),
        file_urls=file_urls,
    )
    logger.debug(
        "api.request.understanding.parsed: %s",
        parsed_request.to_dict(),
    )
    return parsed_request
