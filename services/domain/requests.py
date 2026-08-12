from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class RequestValidationError(ValueError):
    """客户端请求内容不符合接口约束。"""


@dataclass
class UploadedFileInfo:
    field_name: str
    file_name: str
    content_type: str
    content_length: int
    storage: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "fieldName": self.field_name,
            "fileName": self.file_name,
            "contentType": self.content_type,
            "contentLength": self.content_length,
        }


@dataclass(frozen=True)
class ReferenceImageInfo:
    """描述已经暂存且可由服务商访问的参考图。"""

    url: str
    mime_type: str


@dataclass
class GenerateImageRequest:
    request_id: str
    prompt: str
    model: str
    aspect_ratio: str | None
    image_size: str
    input_type: str
    file_urls: list[str]
    files: list[UploadedFileInfo]
    image_count: int = 1
    reference_images: list[ReferenceImageInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requestId": self.request_id,
            "promptLength": len(self.prompt),
            "model": self.model,
            "aspectRatio": self.aspect_ratio,
            "imageSize": self.image_size,
            "imageCount": self.image_count,
            "inputType": self.input_type,
            "fileUrlCount": len(self.file_urls),
            "fileCount": len(self.files),
            "referenceImageCount": len(self.reference_images),
            "receivedFiles": [item.to_dict() for item in self.files],
        }


@dataclass
class UnderstandImageRequest:
    request_id: str
    prompt: str
    model: str
    file_urls: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requestId": self.request_id,
            "promptLength": len(self.prompt),
            "model": self.model,
            "fileUrlCount": len(self.file_urls),
        }
