from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from services.domain.requests import GenerateImageRequest, UploadedFileInfo
from services.http import FetchedAsset
from services.oss_service import TemporaryReferenceObject
from services.reference_images import stage_reference_images
from services.settings import AppSettings, OssSettings


def _build_settings() -> AppSettings:
    return AppSettings(
        oss=OssSettings(
            endpoint="oss-cn-hangzhou.aliyuncs.com",
            region="cn-hangzhou",
            bucket_name="bucket",
            bucket_prefix="images",
            temporary_reference_prefix="temp-references",
            temporary_url_ttl_seconds=3600,
        )
    )


def _build_request() -> GenerateImageRequest:
    return GenerateImageRequest(
        request_id="request-reference-staging",
        prompt="生成图片",
        model="gemini-3.1-flash-image",
        aspect_ratio="1:1",
        image_size="1K",
        input_type="file_url",
        file_urls=["https://assets.example/reference.png?token=secret"],
        files=[],
        image_count=2,
    )


class ReferenceImageStagingTestCase(unittest.TestCase):
    @patch("services.reference_images.TemporaryReferenceStore")
    def test_request_without_reference_skips_temporary_oss(
        self,
        temporary_reference_store,
    ) -> None:
        request_data = _build_request()
        request_data.file_urls = []

        with stage_reference_images(
            request_data,
            _build_settings(),
        ) as staged_request:
            self.assertIs(staged_request, request_data)

        temporary_reference_store.assert_not_called()

    @patch("services.reference_images.TemporaryReferenceStore")
    @patch("services.reference_images.build_asset_fetcher")
    def test_data_url_is_decoded_and_uploaded_without_base64_forwarding(
        self,
        build_asset_fetcher,
        temporary_reference_store,
    ) -> None:
        image_bytes = b"\x89PNG\r\n\x1a\nreference"
        request_data = _build_request()
        request_data.file_urls = [
            "data:image/png;base64,"
            + base64.b64encode(image_bytes).decode("ascii")
        ]
        temporary_object = TemporaryReferenceObject(
            object_key="temp-references/batch/reference.png",
            signed_url="https://bucket.example/reference.png?signature=secret",
            mime_type="image/png",
            content_length=len(image_bytes),
        )
        temporary_reference_store.return_value.upload.return_value = (
            temporary_object
        )
        temporary_reference_store.return_value.delete_many.return_value = (
            SimpleNamespace(
                attempted_count=1,
                deleted_count=1,
                failed_count=0,
            )
        )

        with stage_reference_images(
            request_data,
            _build_settings(),
        ) as staged_request:
            self.assertEqual(
                staged_request.reference_images[0].url,
                temporary_object.signed_url,
            )

        temporary_reference_store.return_value.upload.assert_called_once()
        uploaded_body, uploaded_mime_type, _ = (
            temporary_reference_store.return_value.upload.call_args.args
        )
        self.assertEqual(uploaded_body, image_bytes)
        self.assertEqual(uploaded_mime_type, "image/png")
        build_asset_fetcher.return_value.fetch.assert_not_called()

    @patch("services.reference_images.TemporaryReferenceStore")
    def test_uploaded_file_stream_is_released_before_provider_call(
        self,
        temporary_reference_store,
    ) -> None:
        image_bytes = b"\x89PNG\r\n\x1a\nreference"
        stream = BytesIO(image_bytes)
        storage = SimpleNamespace(
            stream=stream,
            read=stream.read,
            close=MagicMock(),
        )
        uploaded_file = UploadedFileInfo(
            field_name="files",
            file_name="reference.png",
            content_type="image/png",
            content_length=len(image_bytes),
            storage=storage,
        )
        request_data = _build_request()
        request_data.file_urls = []
        request_data.files = [uploaded_file]
        temporary_object = TemporaryReferenceObject(
            object_key="temp-references/batch/reference.png",
            signed_url="https://bucket.example/reference.png?signature=secret",
            mime_type="image/png",
            content_length=len(image_bytes),
        )
        temporary_reference_store.return_value.upload.return_value = (
            temporary_object
        )
        temporary_reference_store.return_value.delete_many.return_value = (
            SimpleNamespace(
                attempted_count=1,
                deleted_count=1,
                failed_count=0,
            )
        )

        with stage_reference_images(
            request_data,
            _build_settings(),
        ) as staged_request:
            storage.close.assert_called_once_with()
            self.assertIsNone(uploaded_file.storage)
            self.assertEqual(len(staged_request.reference_images), 1)

        storage.close.assert_called_once_with()

    @patch("services.reference_images.TemporaryReferenceStore")
    @patch("services.reference_images.build_asset_fetcher")
    def test_staged_reference_is_cleaned_after_context_exit(
        self,
        build_asset_fetcher,
        temporary_reference_store,
    ) -> None:
        fetched = FetchedAsset(
            body=b"reference-image",
            content_type="image/png",
        )
        temporary_object = TemporaryReferenceObject(
            object_key="temp-references/batch/reference.png",
            signed_url=(
                "https://bucket.example/temp-references/reference.png"
                "?signature=secret"
            ),
            mime_type="image/png",
            content_length=len(fetched.body),
        )
        build_asset_fetcher.return_value.fetch.return_value = fetched
        temporary_reference_store.return_value.upload.return_value = (
            temporary_object
        )
        temporary_reference_store.return_value.delete_many.return_value = (
            SimpleNamespace(
                attempted_count=1,
                deleted_count=1,
                failed_count=0,
            )
        )

        with self.assertLogs("services.reference_images", level="INFO") as logs:
            with stage_reference_images(
                _build_request(),
                _build_settings(),
            ) as staged_request:
                self.assertEqual(staged_request.file_urls, [])
                self.assertEqual(staged_request.files, [])
                self.assertEqual(
                    staged_request.reference_images[0].url,
                    temporary_object.signed_url,
                )
                temporary_reference_store.return_value.delete_many.assert_not_called()

        temporary_reference_store.return_value.delete_many.assert_called_once_with(
            [temporary_object]
        )
        rendered_logs = "\n".join(logs.output)
        self.assertIn("image.reference.oss.upload.completed", rendered_logs)
        self.assertIn("image.reference.oss.cleanup.completed", rendered_logs)
        self.assertNotIn("signature=secret", rendered_logs)
        self.assertNotIn("token=secret", rendered_logs)

    @patch("services.reference_images.TemporaryReferenceStore")
    @patch("services.reference_images.build_asset_fetcher")
    def test_provider_failure_still_cleans_staged_reference(
        self,
        build_asset_fetcher,
        temporary_reference_store,
    ) -> None:
        temporary_object = TemporaryReferenceObject(
            object_key="temp-references/batch/reference.png",
            signed_url="https://bucket.example/reference.png?signature=secret",
            mime_type="image/png",
            content_length=9,
        )
        build_asset_fetcher.return_value.fetch.return_value = FetchedAsset(
            body=b"reference",
            content_type="image/png",
        )
        temporary_reference_store.return_value.upload.return_value = (
            temporary_object
        )
        temporary_reference_store.return_value.delete_many.return_value = (
            SimpleNamespace(
                attempted_count=1,
                deleted_count=1,
                failed_count=0,
            )
        )

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            with stage_reference_images(
                _build_request(),
                _build_settings(),
            ):
                raise RuntimeError("provider failed")

        temporary_reference_store.return_value.delete_many.assert_called_once_with(
            [temporary_object]
        )

    @patch("services.reference_images.TemporaryReferenceStore")
    @patch("services.reference_images.build_asset_fetcher")
    def test_cleanup_failure_does_not_replace_provider_failure(
        self,
        build_asset_fetcher,
        temporary_reference_store,
    ) -> None:
        temporary_object = TemporaryReferenceObject(
            object_key="temp-references/batch/reference.png",
            signed_url="https://bucket.example/reference.png?signature=secret",
            mime_type="image/png",
            content_length=9,
        )
        build_asset_fetcher.return_value.fetch.return_value = FetchedAsset(
            body=b"reference",
            content_type="image/png",
        )
        temporary_reference_store.return_value.upload.return_value = (
            temporary_object
        )
        temporary_reference_store.return_value.delete_many.side_effect = RuntimeError(
            "cleanup failed"
        )

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            with stage_reference_images(
                _build_request(),
                _build_settings(),
            ):
                raise RuntimeError("provider failed")

    @patch("services.reference_images.TemporaryReferenceStore")
    @patch("services.reference_images.build_asset_fetcher")
    def test_partial_upload_failure_cleans_completed_objects(
        self,
        build_asset_fetcher,
        temporary_reference_store,
    ) -> None:
        request_data = _build_request()
        request_data.file_urls.append("https://assets.example/second.png")
        first_object = TemporaryReferenceObject(
            object_key="temp-references/batch/first.png",
            signed_url="https://bucket.example/first.png?signature=secret",
            mime_type="image/png",
            content_length=9,
        )
        build_asset_fetcher.return_value.fetch.side_effect = [
            FetchedAsset(
                body=b"reference",
                content_type="image/png",
            ),
            FetchedAsset(
                body=b"second",
                content_type="image/png",
            ),
        ]
        temporary_reference_store.return_value.upload.side_effect = [
            first_object,
            RuntimeError("upload failed"),
        ]
        temporary_reference_store.return_value.delete_many.return_value = (
            SimpleNamespace(
                attempted_count=1,
                deleted_count=1,
                failed_count=0,
            )
        )

        with self.assertRaisesRegex(RuntimeError, "upload failed"):
            with stage_reference_images(request_data, _build_settings()):
                self.fail("上传失败时不应进入模型调用阶段")

        temporary_reference_store.return_value.delete_many.assert_called_once_with(
            [first_object]
        )


if __name__ == "__main__":
    unittest.main()
