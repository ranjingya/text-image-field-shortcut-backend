from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from services.domain.errors import ErrorCategory, ProviderError
from services.gemini_service import GeminiInvocationPlan, invoke_gemini
from services.http.asset_fetcher import AssetFetcher, AssetFetchError
from services.http.client_factory import HttpClientRegistry
from services.openai_image_service import OpenAIImageInvocationPlan, invoke_openai_image
from services.settings import HttpClientSettings


class HttpClientRegistryTestCase(unittest.TestCase):
    def test_same_name_reuses_client(self) -> None:
        registry = HttpClientRegistry()
        settings = HttpClientSettings()

        first = registry.get("provider", settings)
        second = registry.get("provider", settings)

        self.assertIs(first, second)
        registry.close()

    def test_different_names_use_separate_clients(self) -> None:
        registry = HttpClientRegistry()
        settings = HttpClientSettings()

        provider = registry.get("provider", settings)
        asset = registry.get("asset", settings)

        self.assertIsNot(provider, asset)
        registry.close()


class ProviderHttpxInvocationTestCase(unittest.TestCase):
    def test_gemini_invocation_uses_json_and_bearer_token(self) -> None:
        captured_request: httpx.Request | None = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(200, json={"candidates": []})

        plan = GeminiInvocationPlan(
            api_url="https://provider.example/v1beta/models/test:generateContent",
            api_path="/v1beta/models/test:generateContent",
            model="test",
            prompt="测试",
            prepared_inputs=[],
            request_body={"contents": [{"parts": [{"text": "测试"}]}]},
        )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            response = invoke_gemini(plan, "secret", client=client)

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(captured_request)
        self.assertEqual(captured_request.headers["authorization"], "Bearer secret")
        self.assertIn(b'"contents"', captured_request.content)

    def test_gemini_error_payload_is_preserved_without_signed_url(self) -> None:
        signed_url = (
            "https://temp-images.example.com/temp-references/image.jpg"
            "?x-oss-signature=secret"
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": f"Cannot fetch reference {signed_url}",
                        "type": "upstream_error",
                    }
                },
                headers={"x-request-id": "easyrouter-request"},
            )

        plan = GeminiInvocationPlan(
            api_url="https://provider.example/v1beta/models/test:generateContent",
            api_path="/v1beta/models/test:generateContent",
            model="test",
            prompt="测试",
            prepared_inputs=[],
            request_body={"contents": [{"parts": [{"text": "测试"}]}]},
        )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(ProviderError) as raised:
                invoke_gemini(plan, "secret", client=client)

        error = raised.exception
        self.assertEqual(error.category, ErrorCategory.INVALID_REQUEST)
        self.assertEqual(error.provider_error_type, "upstream_error")
        self.assertEqual(error.request_id, "easyrouter-request")
        self.assertGreater(error.response_bytes or 0, 0)
        self.assertEqual(error.message, "Cannot fetch reference <url>")
        self.assertNotIn("x-oss-signature", error.message)

    def test_gemini_reference_connect_timeout_is_retryable(self) -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "Cannot fetch content from the provided URL. "
                            "Status: URL_UNREACHABLE-UNREACHABLE_CONNECT_TIMEOUT"
                        ),
                        "type": "upstream_error",
                    }
                },
            )
        )
        plan = GeminiInvocationPlan(
            api_url="https://provider.example/v1beta/models/test:generateContent",
            api_path="/v1beta/models/test:generateContent",
            model="test",
            prompt="测试",
            prepared_inputs=[],
            request_body={"contents": []},
        )

        with httpx.Client(transport=transport) as client:
            with self.assertRaises(ProviderError) as raised:
                invoke_gemini(plan, "secret", client=client)

        self.assertEqual(
            raised.exception.category,
            ErrorCategory.UPSTREAM_UNAVAILABLE,
        )
        self.assertTrue(raised.exception.retryable)

    def test_gemini_non_json_error_uses_limited_response_text(self) -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(400, text="invalid image input")
        )
        plan = GeminiInvocationPlan(
            api_url="https://provider.example/v1beta/models/test:generateContent",
            api_path="/v1beta/models/test:generateContent",
            model="test",
            prompt="测试",
            prepared_inputs=[],
            request_body={"contents": []},
        )

        with httpx.Client(transport=transport) as client:
            with self.assertRaises(ProviderError) as raised:
                invoke_gemini(plan, "secret", client=client)

        self.assertEqual(raised.exception.message, "invalid image input")

    def test_openai_image_invocation_uses_images_endpoint(self) -> None:
        captured_request: httpx.Request | None = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(200, json={"data": [{"b64_json": "aW1hZ2U="}]})

        plan = OpenAIImageInvocationPlan(
            api_url="https://provider.example/v1/images/generations",
            model="gpt-image-test",
            prompt="测试",
            size="1024x1024",
            n=1,
            quality="auto",
            request_body={"model": "gpt-image-test", "prompt": "测试"},
        )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            response = invoke_openai_image(plan, "secret", client=client)

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(captured_request)
        self.assertEqual(captured_request.url.path, "/v1/images/generations")


class AssetFetcherTestCase(unittest.TestCase):
    @patch("services.http.asset_fetcher._validate_public_http_url")
    def test_redirect_is_validated_and_downloaded(self, validate_url) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/start":
                return httpx.Response(302, headers={"location": "/image.png"})
            return httpx.Response(
                200,
                headers={"content-type": "image/png"},
                content=b"\x89PNG\r\n\x1a\nimage",
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result = AssetFetcher(client, max_bytes=100, max_redirects=2).fetch(
                "https://assets.example/start"
            )

        self.assertEqual(result.body, b"\x89PNG\r\n\x1a\nimage")
        self.assertEqual(result.content_type, "image/png")
        self.assertEqual(validate_url.call_count, 2)

    @patch("services.http.asset_fetcher._validate_public_http_url")
    def test_response_over_limit_is_rejected(self, _validate_url) -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, content=b"too-large")
        )

        with httpx.Client(transport=transport) as client:
            with self.assertRaisesRegex(AssetFetchError, "资源大小超过限制"):
                AssetFetcher(client, max_bytes=3, max_redirects=0).fetch(
                    "https://assets.example/image.png"
                )


class DirectHttpImportTestCase(unittest.TestCase):
    def test_business_source_only_uses_httpx_for_direct_http(self) -> None:
        forbidden_imports = {"requests", "http.client", "urllib.request"}
        violations: list[str] = []

        for source_root in (Path("api"), Path("services")):
            for path in source_root.rglob("*.py"):
                tree = ast.parse(
                    path.read_text(encoding="utf-8"), filename=str(path)
                )
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported_names = {alias.name for alias in node.names}
                    elif isinstance(node, ast.ImportFrom):
                        imported_names = {node.module or ""}
                    else:
                        continue
                    matched = forbidden_imports.intersection(imported_names)
                    if matched:
                        violations.append(
                            f"{path}: {', '.join(sorted(matched))}"
                        )

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
