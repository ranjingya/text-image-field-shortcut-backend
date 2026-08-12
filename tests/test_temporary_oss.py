from __future__ import annotations

from datetime import timedelta
import unittest
from unittest.mock import patch
from uuid import uuid4

from services.oss_service import (
    TemporaryReferenceStore,
    create_oss_cname_client,
)
from services.settings import AppSettings, OssSettings


class _FakePresignResult:
    url = "https://bucket.example/temp-references/signed?signature=secret"


class _FakeOssClient:
    def __init__(
        self,
        failed_delete_keys: set[str] | None = None,
        fail_presign: bool = False,
    ) -> None:
        self.put_requests = []
        self.presign_requests = []
        self.delete_requests = []
        self.failed_delete_keys = failed_delete_keys or set()
        self.fail_presign = fail_presign

    def put_object(self, request):
        self.put_requests.append(request)
        return object()

    def presign(self, request, **kwargs):
        self.presign_requests.append((request, kwargs))
        if self.fail_presign:
            raise RuntimeError("presign failed")
        return _FakePresignResult()

    def delete_object(self, request):
        self.delete_requests.append(request)
        if request.key in self.failed_delete_keys:
            raise RuntimeError("delete failed")
        return object()


def _build_settings(custom_domain: str = "") -> AppSettings:
    return AppSettings(
        oss=OssSettings(
            endpoint="oss-cn-hangzhou.aliyuncs.com",
            region="cn-hangzhou",
            bucket_name="bucket",
            bucket_prefix="images",
            temporary_reference_prefix="temp-references",
            temporary_url_ttl_seconds=3600,
            temporary_custom_domain=custom_domain,
        )
    )


class TemporaryReferenceStoreTestCase(unittest.TestCase):
    def test_upload_creates_private_object_and_signed_get_url(self) -> None:
        client = _FakeOssClient()
        store = TemporaryReferenceStore(_build_settings(), client)
        batch_id = uuid4().hex

        uploaded = store.upload(b"reference", "image/png", batch_id)

        self.assertEqual(len(client.put_requests), 1)
        put_request = client.put_requests[0]
        self.assertEqual(put_request.bucket, "bucket")
        self.assertTrue(
            put_request.key.startswith(f"temp-references/{batch_id}/")
        )
        self.assertTrue(put_request.key.endswith(".png"))
        self.assertEqual(put_request.body, b"reference")
        self.assertEqual(put_request.content_type, "image/png")
        self.assertEqual(put_request.acl, "private")

        get_request, kwargs = client.presign_requests[0]
        self.assertEqual(get_request.bucket, "bucket")
        self.assertEqual(get_request.key, put_request.key)
        self.assertEqual(kwargs["expires"], timedelta(seconds=3600))
        self.assertEqual(uploaded.object_key, put_request.key)
        self.assertEqual(uploaded.signed_url, _FakePresignResult.url)
        self.assertEqual(uploaded.content_length, len(b"reference"))

    def test_custom_domain_client_uses_cname_addressing(self) -> None:
        settings = _build_settings("https://temp-images.example.com")
        cfg = type("FakeConfig", (), {})()
        expected_client = object()

        with (
            patch(
                "services.oss_service.oss.credentials."
                "EnvironmentVariableCredentialsProvider",
                return_value=object(),
            ),
            patch(
                "services.oss_service.oss.config.load_default",
                return_value=cfg,
            ),
            patch(
                "services.oss_service.oss.Client",
                return_value=expected_client,
            ) as client_factory,
        ):
            client = create_oss_cname_client(settings)

        self.assertIs(client, expected_client)
        self.assertEqual(cfg.region, "cn-hangzhou")
        self.assertEqual(cfg.endpoint, "https://temp-images.example.com")
        self.assertTrue(cfg.use_cname)
        client_factory.assert_called_once_with(cfg)

    def test_custom_domain_store_uses_separate_presign_client(self) -> None:
        settings = _build_settings("https://temp-images.example.com")
        upload_client = _FakeOssClient()
        presign_client = _FakeOssClient()

        with patch(
            "services.oss_service.create_oss_cname_client",
            return_value=presign_client,
        ):
            store = TemporaryReferenceStore(settings, upload_client)
            uploaded = store.upload(
                b"reference",
                "image/png",
                uuid4().hex,
            )

        self.assertEqual(len(upload_client.put_requests), 1)
        self.assertEqual(upload_client.presign_requests, [])
        self.assertEqual(len(presign_client.presign_requests), 1)
        self.assertEqual(uploaded.signed_url, _FakePresignResult.url)

    def test_upload_rejects_non_uuid_batch_id(self) -> None:
        client = _FakeOssClient()
        store = TemporaryReferenceStore(_build_settings(), client)

        with self.assertRaises(ValueError):
            store.upload(b"reference", "image/png", "request-id-from-client")

        self.assertEqual(client.put_requests, [])

    def test_presign_failure_deletes_uploaded_object(self) -> None:
        client = _FakeOssClient(fail_presign=True)
        store = TemporaryReferenceStore(_build_settings(), client)

        with self.assertRaisesRegex(RuntimeError, "presign failed"):
            store.upload(b"reference", "image/png", uuid4().hex)

        self.assertEqual(len(client.put_requests), 1)
        self.assertEqual(len(client.delete_requests), 1)
        self.assertEqual(
            client.delete_requests[0].key,
            client.put_requests[0].key,
        )

    def test_delete_many_continues_after_single_object_failure(self) -> None:
        settings = _build_settings()
        initial_client = _FakeOssClient()
        initial_store = TemporaryReferenceStore(settings, initial_client)
        batch_id = uuid4().hex
        first = initial_store.upload(b"first", "image/png", batch_id)
        second = initial_store.upload(b"second", "image/jpeg", batch_id)
        client = _FakeOssClient(failed_delete_keys={first.object_key})
        store = TemporaryReferenceStore(settings, client)

        with self.assertLogs("services.oss_service", level="WARNING") as logs:
            result = store.delete_many([first, second])

        self.assertEqual(result.attempted_count, 2)
        self.assertEqual(result.deleted_count, 1)
        self.assertEqual(result.failed_count, 1)
        self.assertEqual(len(client.delete_requests), 2)
        self.assertNotIn("signature=secret", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
