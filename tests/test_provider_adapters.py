from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from services.domain.errors import (
    ErrorCategory,
    ProviderError,
    provider_error_from_httpx,
)
from services.domain.requests import (
    GenerateImageRequest,
    ReferenceImageInfo,
    UnderstandImageRequest,
)
from services.gemini_service import build_gemini_invocation_plan
from services.http import AssetFetchError, FetchedAsset
from services.model_registry import load_model_registry
from services.providers.openrouter import OpenRouterProvider, _build_input_references
from services.settings import AppSettings, OssSettings


def _build_settings() -> AppSettings:
    return AppSettings(
        oss=OssSettings(endpoint="", region="", bucket_name="", bucket_prefix=""),
    )


class ModelRegistryTestCase(unittest.TestCase):
    def test_preview_alias_routes_to_stable_provider_models(self) -> None:
        registry = load_model_registry("config/providers.json")

        public_model = registry.resolve("gemini-3.1-flash-image-preview")

        self.assertEqual(public_model, "gemini-3.1-flash-image")
        self.assertEqual(
            registry.provider_model(public_model, "easyrouter"),
            "gemini-3.1-flash-image",
        )
        self.assertEqual(
            registry.provider_model(public_model, "openrouter"),
            "google/gemini-3.1-flash-image",
        )

    def test_empty_model_uses_configured_default(self) -> None:
        registry = load_model_registry("config/providers.json")

        self.assertEqual(registry.resolve(""), "gemini-3.1-flash-image")
        self.assertEqual(
            registry.configuration.providers["easyrouter"].base_url,
            "https://easyrouter.io",
        )

    def test_gemini_2_5_flash_image_routes_to_provider_models(self) -> None:
        registry = load_model_registry("config/providers.json")

        public_model = registry.resolve("gemini-2.5-flash-image-preview")

        self.assertEqual(public_model, "gemini-2.5-flash-image")
        self.assertEqual(
            registry.provider_model(public_model, "easyrouter"),
            "gemini-2.5-flash-image",
        )
        self.assertEqual(
            registry.provider_model(public_model, "openrouter"),
            "google/gemini-2.5-flash-image",
        )
        self.assertTrue(registry.supports(public_model, "image_generation"))
        self.assertTrue(registry.supports(public_model, "image_understanding"))
        self.assertTrue(registry.supports(public_model, "reference_image"))

    def test_invalid_duplicate_alias_is_rejected(self) -> None:
        config_path = Path("tests/.provider-config-invalid.json")
        config_path.write_text(
            json.dumps(
                {
                    "primary_provider": "primary",
                    "fallback_providers": [],
                    "default_model": "model-a",
                    "providers": {
                        "primary": {
                            "adapter": "easyrouter",
                            "base_url": "https://provider.example",
                            "api_key_env": "KEY",
                        }
                    },
                    "models": {
                        "model-a": {
                            "aliases": ["legacy"],
                            "capabilities": [],
                            "providers": {"primary": "a"},
                        },
                        "model-b": {
                            "aliases": ["legacy"],
                            "capabilities": [],
                            "providers": {"primary": "b"},
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        self.addCleanup(config_path.unlink, missing_ok=True)

        with self.assertRaisesRegex(ValueError, "模型别名重复"):
            load_model_registry(str(config_path))


class OpenRouterProviderTestCase(unittest.TestCase):
    def test_easyrouter_uses_signed_reference_url_without_logging_it(self) -> None:
        signed_url = "https://bucket.example/reference.png?signature=secret"
        request_data = GenerateImageRequest(
            request_id="request-signed-reference",
            prompt="生成图片",
            model="gemini-3.1-flash-image",
            aspect_ratio="1:1",
            image_size="1K",
            input_type="file_url",
            file_urls=[],
            files=[],
            raw_payload={},
            reference_images=[
                ReferenceImageInfo(url=signed_url, mime_type="image/png")
            ],
        )

        plan = build_gemini_invocation_plan(
            request_data,
            _build_settings(),
            "https://easyrouter.example",
        )

        self.assertEqual(
            plan.request_body["contents"][0]["parts"][1],
            {
                "fileData": {
                    "mimeType": "image/png",
                    "fileUri": signed_url,
                }
            },
        )
        self.assertNotIn(signed_url, json.dumps(plan.to_dict()))
        self.assertIn("<signed-url>", json.dumps(plan.to_dict()))

    def test_openrouter_converts_signed_reference_to_base64(self) -> None:
        signed_url = "https://bucket.example/reference.png?signature=secret"
        request_data = GenerateImageRequest(
            request_id="request-openrouter-signed-reference",
            prompt="生成图片",
            model="gemini-3.1-flash-image",
            aspect_ratio="1:1",
            image_size="1K",
            input_type="file_url",
            file_urls=[],
            files=[],
            raw_payload={},
            reference_images=[
                ReferenceImageInfo(url=signed_url, mime_type="image/png")
            ],
        )

        asset_fetcher = Mock()
        asset_fetcher.fetch.return_value = FetchedAsset(
            body=b"reference-image",
            content_type="image/png",
            final_url=signed_url,
        )

        references = _build_input_references(request_data, asset_fetcher)

        self.assertEqual(
            references,
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,cmVmZXJlbmNlLWltYWdl"},
                }
            ],
        )
        asset_fetcher.fetch.assert_called_once_with(signed_url)
        self.assertNotIn(signed_url, json.dumps(references))

    def test_openrouter_maps_reference_download_failure(self) -> None:
        request_data = GenerateImageRequest(
            request_id="request-openrouter-reference-failed",
            prompt="生成图片",
            model="gemini-3.1-flash-image",
            aspect_ratio="1:1",
            image_size="1K",
            input_type="file_url",
            file_urls=[],
            files=[],
            raw_payload={},
            reference_images=[
                ReferenceImageInfo(
                    url="https://bucket.example/reference.png?signature=secret",
                    mime_type="image/png",
                )
            ],
        )
        asset_fetcher = Mock()
        asset_fetcher.fetch.side_effect = AssetFetchError("download failed")

        with self.assertRaises(ProviderError) as raised:
            _build_input_references(request_data, asset_fetcher)

        self.assertEqual(raised.exception.category, ErrorCategory.INVALID_ASSET)
        self.assertFalse(raised.exception.counts_toward_circuit)

    def test_providers_reject_unstaged_generation_reference(self) -> None:
        request_data = GenerateImageRequest(
            request_id="request-unstaged-reference",
            prompt="生成图片",
            model="gemini-3.1-flash-image",
            aspect_ratio="1:1",
            image_size="1K",
            input_type="file_url",
            file_urls=["https://assets.example/reference.png"],
            files=[],
            raw_payload={},
        )

        with self.assertRaisesRegex(ProviderError, "尚未完成临时 OSS 暂存"):
            build_gemini_invocation_plan(
                request_data,
                _build_settings(),
                "https://easyrouter.example",
            )
        with self.assertRaisesRegex(ProviderError, "尚未完成临时 OSS 暂存"):
            _build_input_references(request_data, None)

    def test_generate_image_uses_official_images_schema(self) -> None:
        captured_request: httpx.Request | None = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(
                200,
                headers={"x-request-id": "or-request"},
                json={"created": 1, "data": [{"b64_json": "aW1hZ2U="}]},
            )

        request_data = GenerateImageRequest(
            request_id="request-1",
            prompt="生成图片",
            model="gemini-3.1-flash-image",
            aspect_ratio="16:9",
            image_size="2K",
            input_type="file_url",
            file_urls=[],
            files=[],
            raw_payload={},
            reference_images=[
                ReferenceImageInfo(
                    url="https://bucket.example/reference.png?signature=secret",
                    mime_type="image/png",
                )
            ],
        )

        fetched_asset = FetchedAsset(
            body=b"reference-image",
            content_type="image/png",
            final_url="https://bucket.example/reference.png",
        )
        with (
            httpx.Client(transport=httpx.MockTransport(handler)) as client,
            patch("services.providers.openrouter.build_asset_fetcher") as fetcher,
        ):
            fetcher.return_value.fetch.return_value = fetched_asset
            provider = OpenRouterProvider(
                _build_settings(),
                "https://openrouter.example/api/v1",
                "openrouter-key",
                client,
            )
            result = provider.generate_image(
                request_data,
                "gemini-3.1-flash-image",
                "google/gemini-3.1-flash-image",
            )

        self.assertEqual(result.provider, "openrouter")
        self.assertEqual(result.request_id, "or-request")
        self.assertEqual(result.result.assets[0].payload, b"image")
        self.assertIsNotNone(captured_request)
        payload = json.loads(captured_request.content)
        self.assertEqual(captured_request.url.path, "/api/v1/images")
        self.assertEqual(payload["n"], 1)
        self.assertEqual(payload["resolution"], "2K")
        self.assertEqual(payload["aspect_ratio"], "16:9")
        self.assertEqual(
            payload["input_references"][0],
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,cmVmZXJlbmNlLWltYWdl"},
            },
        )

    @patch("services.providers.openrouter.build_asset_fetcher")
    def test_understand_image_uses_chat_completions_schema(
        self,
        build_fetcher: Mock,
    ) -> None:
        captured_request: httpx.Request | None = None
        build_fetcher.return_value.fetch.return_value = FetchedAsset(
            body=b"\x89PNG\r\n\x1a\nreference-image",
            content_type="image/png",
            final_url="https://assets.example/cat.png",
        )

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "图片里有一只猫。"}}]},
            )

        request_data = UnderstandImageRequest(
            request_id="request-2",
            prompt="描述图片",
            model="gemini-3.1-flash-image",
            file_urls=["https://assets.example/cat.png"],
            raw_payload={},
        )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            provider = OpenRouterProvider(
                _build_settings(),
                "https://openrouter.example/api/v1",
                "openrouter-key",
                client,
            )
            result = provider.understand_image(
                request_data,
                "gemini-3.1-flash-image",
                "google/gemini-3.1-flash-image",
            )

        self.assertEqual(result.text, "图片里有一只猫。")
        self.assertIsNotNone(captured_request)
        payload = json.loads(captured_request.content)
        self.assertEqual(captured_request.url.path, "/api/v1/chat/completions")
        self.assertEqual(payload["messages"][0]["content"][0]["type"], "text")
        self.assertEqual(payload["messages"][0]["content"][1]["type"], "image_url")
        self.assertTrue(
            payload["messages"][0]["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )
        self.assertNotIn(
            "https://assets.example/cat.png",
            json.dumps(payload),
        )

    def test_openrouter_error_type_is_normalized(self) -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(
                400,
                json={
                    "error": {
                        "message": "blocked",
                        "metadata": {
                            "error_type": "content_policy_violation",
                        },
                    }
                },
                headers={"x-request-id": "or-error-request"},
            )
        )
        request_data = GenerateImageRequest(
            request_id="request-3",
            prompt="生成图片",
            model="gemini-3.1-flash-image",
            aspect_ratio=None,
            image_size="1K",
            input_type="empty",
            file_urls=[],
            files=[],
            raw_payload={},
        )

        with httpx.Client(transport=transport) as client:
            provider = OpenRouterProvider(
                _build_settings(),
                "https://openrouter.example/api/v1",
                "openrouter-key",
                client,
            )
            with self.assertRaises(ProviderError) as raised:
                provider.generate_image(
                    request_data,
                    "gemini-3.1-flash-image",
                    "google/gemini-3.1-flash-image",
                )

        self.assertEqual(raised.exception.category, ErrorCategory.CONTENT_POLICY)
        self.assertEqual(
            raised.exception.provider_error_type,
            "content_policy_violation",
        )
        self.assertEqual(raised.exception.request_id, "or-error-request")
        self.assertFalse(raised.exception.retryable)

    def test_openrouter_error_message_redacts_remote_and_data_urls(self) -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "failed https://temp.example/image.png?token=secret "
                            "data:image/png;base64,c2VjcmV0"
                        )
                    }
                },
            )
        )
        request_data = GenerateImageRequest(
            request_id="request-redaction",
            prompt="生成图片",
            model="gemini-3.1-flash-image",
            aspect_ratio=None,
            image_size="1K",
            input_type="empty",
            file_urls=[],
            files=[],
            raw_payload={},
        )

        with httpx.Client(transport=transport) as client:
            provider = OpenRouterProvider(
                _build_settings(),
                "https://openrouter.example/api/v1",
                "openrouter-key",
                client,
            )
            with self.assertRaises(ProviderError) as raised:
                provider.generate_image(
                    request_data,
                    "gemini-3.1-flash-image",
                    "google/gemini-3.1-flash-image",
                )

        self.assertEqual(str(raised.exception), "failed <url> <data-url>")


class ProviderErrorMappingTestCase(unittest.TestCase):
    def test_pool_timeout_does_not_count_toward_circuit(self) -> None:
        request = httpx.Request("POST", "https://provider.example")

        error = provider_error_from_httpx(
            "easyrouter", httpx.PoolTimeout("busy", request=request)
        )

        self.assertEqual(error.category, ErrorCategory.LOCAL_CAPACITY)
        self.assertTrue(error.retryable)
        self.assertFalse(error.counts_toward_circuit)


if __name__ == "__main__":
    unittest.main()
