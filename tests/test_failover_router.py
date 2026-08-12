from __future__ import annotations

import unittest
from unittest.mock import patch

from services.domain.errors import ErrorCategory, ProviderError
from services.domain.provider import ImageProviderResult, TextProviderResult
from services.model_registry import load_model_registry
from services.domain.requests import GenerateImageRequest, UnderstandImageRequest
from services.response_normalizer import NormalizedGeneratedAsset, NormalizedModelResult
from services.routing import FailoverExhaustedError, FailoverRouter
from services.routing.circuit_breaker import CircuitOpenError, CircuitState
from services.settings import AppSettings, OssSettings, RoutingSettings


def _image_result(provider: str, public_model: str, provider_model: str) -> ImageProviderResult:
    return ImageProviderResult(
        provider=provider,
        public_model=public_model,
        provider_model=provider_model,
        result=NormalizedModelResult(
            raw_response_type="json_base64",
            assets=[
                NormalizedGeneratedAsset(
                    asset_type="image_base64",
                    mime_type="image/png",
                    file_name="image.png",
                    source_kind="bytes",
                    payload=b"image",
                )
            ],
            text_output="",
            raw_meta={},
        ),
    )


def _empty_image_result(
    provider: str, public_model: str, provider_model: str
) -> ImageProviderResult:
    return ImageProviderResult(
        provider=provider,
        public_model=public_model,
        provider_model=provider_model,
        result=NormalizedModelResult(
            raw_response_type="json_text",
            assets=[],
            text_output="",
            raw_meta={},
        ),
    )


class FakeProvider:
    def __init__(self, name: str, image_outcomes: list, text_outcomes: list | None = None) -> None:
        self.name = name
        self.image_outcomes = list(image_outcomes)
        self.text_outcomes = list(text_outcomes or [])
        self.calls: list[tuple[str, str, float | None]] = []

    def generate_image(
        self,
        request: GenerateImageRequest,
        public_model: str,
        provider_model: str,
        timeout_seconds: float | None = None,
    ) -> ImageProviderResult:
        self.calls.append((public_model, provider_model, timeout_seconds))
        outcome = self.image_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def understand_image(
        self,
        request: UnderstandImageRequest,
        public_model: str,
        provider_model: str,
        timeout_seconds: float | None = None,
    ) -> TextProviderResult:
        self.calls.append((public_model, provider_model, timeout_seconds))
        outcome = self.text_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class PrimaryOpenCircuitBreaker:
    state_available = True

    def before_call(self, provider: str, capability: str) -> CircuitState:
        if provider == "easyrouter":
            raise CircuitOpenError(provider, capability, CircuitState.OPEN)
        return CircuitState.CLOSED

    def record_success(
        self,
        provider: str,
        capability: str,
        admission_state: CircuitState,
    ) -> None:
        return None

    def record_failure(
        self, provider: str, capability: str, error: ProviderError
    ) -> bool:
        return False


class CollectingEventReporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, tuple]] = []

    def on_provider_failure(self, *args) -> None:
        self.events.append(("provider_failure", args))

    def on_fallback_used(self, *args) -> None:
        self.events.append(("fallback_used", args))

    def on_circuit_open(self, *args) -> None:
        self.events.append(("circuit_open", args))

    def on_all_providers_failed(self, *args) -> None:
        self.events.append(("all_failed", args))

    def on_primary_success(self, *args) -> None:
        self.events.append(("primary_success", args))


def _build_settings(*, fallback_enabled: bool = True) -> AppSettings:
    return AppSettings(
        oss=OssSettings(endpoint="", region="", bucket_name="", bucket_prefix=""),
        fallback_enabled=fallback_enabled,
        routing=RoutingSettings(
            provider_timeout_seconds=30,
            primary_max_attempts=1,
            fallback_max_attempts=1,
            primary_empty_response_retry_count=1,
        ),
    )


def _build_request(model: str = "gemini-3.1-flash-image-preview") -> GenerateImageRequest:
    return GenerateImageRequest(
        request_id="request-1",
        prompt="生成图片",
        model=model,
        aspect_ratio="1:1",
        image_size="1K",
        input_type="empty",
        file_urls=[],
        files=[],
    )


class FailoverRouterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_model_registry("config/providers.json")

    def test_primary_success_does_not_call_fallback(self) -> None:
        primary_result = _image_result(
            "easyrouter",
            "gemini-3.1-flash-image",
            "gemini-3.1-flash-image",
        )
        primary = FakeProvider("easyrouter", [primary_result])
        fallback = FakeProvider("openrouter", [])
        router = FailoverRouter(
            _build_settings(),
            self.registry,
            {"easyrouter": primary, "openrouter": fallback},
        )

        result = router.generate_image(_build_request())

        self.assertFalse(result.fallback_used)
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(fallback.calls, [])
        self.assertEqual(primary.calls[0][0], "gemini-3.1-flash-image")

    def test_retryable_primary_error_routes_to_openrouter(self) -> None:
        underlying_error = RuntimeError("request failed")
        underlying_error.request_body = b"large-request-body"
        primary_error = ProviderError(
            provider="easyrouter",
            category=ErrorCategory.TIMEOUT,
            message="timeout",
            retryable=True,
            cause=underlying_error,
        )
        fallback_result = _image_result(
            "openrouter",
            "gemini-3.1-flash-image",
            "google/gemini-3.1-flash-image",
        )
        primary = FakeProvider("easyrouter", [primary_error])
        fallback = FakeProvider("openrouter", [fallback_result])
        reporter = CollectingEventReporter()
        router = FailoverRouter(
            _build_settings(),
            self.registry,
            {"easyrouter": primary, "openrouter": fallback},
            event_reporter=reporter,
        )

        result = router.generate_image(_build_request())

        self.assertTrue(result.fallback_used)
        self.assertEqual(fallback.calls[0][1], "google/gemini-3.1-flash-image")
        self.assertGreater(fallback.calls[0][2] or 0, 0)
        self.assertEqual([item.provider for item in result.attempts], ["easyrouter", "openrouter"])
        self.assertEqual(
            [event[0] for event in reporter.events],
            ["provider_failure", "fallback_used"],
        )
        reported_error = reporter.events[0][1][3]
        self.assertIsNot(reported_error, primary_error)
        self.assertIsNone(reported_error.cause)
        self.assertIsNone(reported_error.__traceback__)

    def test_fallback_receives_a_fresh_provider_timeout(self) -> None:
        primary_error = ProviderError(
            provider="easyrouter",
            category=ErrorCategory.TIMEOUT,
            message="timeout",
            retryable=True,
        )
        fallback_result = _image_result(
            "openrouter",
            "gemini-3.1-flash-image",
            "google/gemini-3.1-flash-image",
        )
        primary = FakeProvider("easyrouter", [primary_error])
        fallback = FakeProvider("openrouter", [fallback_result])
        router = FailoverRouter(
            _build_settings(),
            self.registry,
            {"easyrouter": primary, "openrouter": fallback},
        )

        with patch(
            "services.routing.failover.time.monotonic",
            side_effect=[100.0, 100.0, 400.0, 400.0],
        ):
            router.generate_image(_build_request())

        self.assertEqual(primary.calls[0][2], 30.0)
        self.assertEqual(fallback.calls[0][2], 30.0)

    def test_non_retryable_error_does_not_call_fallback(self) -> None:
        primary_error = ProviderError(
            provider="easyrouter",
            category=ErrorCategory.INVALID_REQUEST,
            message="invalid parameter",
            status_code=400,
            request_id="provider-request-1",
            provider_error_type="invalid_request",
            retryable=False,
        )
        primary = FakeProvider("easyrouter", [primary_error])
        fallback = FakeProvider("openrouter", [])
        router = FailoverRouter(
            _build_settings(),
            self.registry,
            {"easyrouter": primary, "openrouter": fallback},
        )

        with self.assertLogs(
            "services.routing.failover", level="WARNING"
        ) as captured_logs:
            with self.assertRaises(ProviderError) as raised:
                router.generate_image(_build_request())

        self.assertEqual(raised.exception.category, ErrorCategory.INVALID_REQUEST)
        self.assertEqual(fallback.calls, [])
        failure_log = "\n".join(captured_logs.output)
        self.assertIn("'statusCode': 400", failure_log)
        self.assertIn("'providerErrorType': 'invalid_request'", failure_log)
        self.assertIn("'providerRequestId': 'provider-request-1'", failure_log)
        self.assertIn("'message': 'invalid parameter'", failure_log)

    def test_empty_primary_response_retries_before_fallback(self) -> None:
        empty = _empty_image_result(
            "easyrouter",
            "gemini-3.1-flash-image",
            "gemini-3.1-flash-image",
        )
        success = _image_result(
            "easyrouter",
            "gemini-3.1-flash-image",
            "gemini-3.1-flash-image",
        )
        primary = FakeProvider("easyrouter", [empty, success])
        fallback = FakeProvider("openrouter", [])
        router = FailoverRouter(
            _build_settings(),
            self.registry,
            {"easyrouter": primary, "openrouter": fallback},
        )

        result = router.generate_image(_build_request())

        self.assertFalse(result.fallback_used)
        self.assertEqual(len(primary.calls), 2)
        self.assertEqual(fallback.calls, [])

    def test_both_providers_fail_raises_exhausted_error(self) -> None:
        primary_error = ProviderError(
            provider="easyrouter",
            category=ErrorCategory.TIMEOUT,
            message="timeout",
            retryable=True,
        )
        fallback_error = ProviderError(
            provider="openrouter",
            category=ErrorCategory.RATE_LIMIT,
            message="limited",
            retryable=True,
        )
        reporter = CollectingEventReporter()
        router = FailoverRouter(
            _build_settings(),
            self.registry,
            {
                "easyrouter": FakeProvider("easyrouter", [primary_error]),
                "openrouter": FakeProvider("openrouter", [fallback_error]),
            },
            event_reporter=reporter,
        )

        with self.assertRaises(FailoverExhaustedError) as raised:
            router.generate_image(_build_request())

        self.assertEqual(len(raised.exception.errors), 2)
        self.assertEqual(
            [event[0] for event in reporter.events],
            ["provider_failure", "provider_failure", "all_failed"],
        )

    def test_non_retryable_fallback_does_not_mask_retryable_primary_error(
        self,
    ) -> None:
        primary_error = ProviderError(
            provider="easyrouter",
            category=ErrorCategory.LOCAL_CAPACITY,
            message="pool busy",
            retryable=True,
        )
        fallback_error = ProviderError(
            provider="openrouter",
            category=ErrorCategory.INVALID_REQUEST,
            message="Unsupported URL, public internet addresses only",
            status_code=400,
            retryable=False,
        )
        router = FailoverRouter(
            _build_settings(),
            self.registry,
            {
                "easyrouter": FakeProvider("easyrouter", [primary_error]),
                "openrouter": FakeProvider("openrouter", [fallback_error]),
            },
        )

        with self.assertRaises(FailoverExhaustedError) as raised:
            router.generate_image(_build_request())

        self.assertEqual(len(raised.exception.errors), 2)
        self.assertTrue(raised.exception.retryable)

    def test_fallback_switch_can_disable_openrouter(self) -> None:
        primary_error = ProviderError(
            provider="easyrouter",
            category=ErrorCategory.TIMEOUT,
            message="timeout",
            retryable=True,
        )
        fallback = FakeProvider("openrouter", [])
        router = FailoverRouter(
            _build_settings(fallback_enabled=False),
            self.registry,
            {"easyrouter": FakeProvider("easyrouter", [primary_error]), "openrouter": fallback},
        )

        with self.assertRaises(ProviderError):
            router.generate_image(_build_request())

        self.assertEqual(fallback.calls, [])

    def test_open_primary_circuit_skips_directly_to_fallback(self) -> None:
        fallback_result = _image_result(
            "openrouter",
            "gemini-3.1-flash-image",
            "google/gemini-3.1-flash-image",
        )
        primary = FakeProvider("easyrouter", [])
        fallback = FakeProvider("openrouter", [fallback_result])
        router = FailoverRouter(
            _build_settings(),
            self.registry,
            {"easyrouter": primary, "openrouter": fallback},
            PrimaryOpenCircuitBreaker(),
        )

        result = router.generate_image(_build_request())

        self.assertTrue(result.fallback_used)
        self.assertEqual(primary.calls, [])
        self.assertEqual(result.attempts[0].attempt, 0)


if __name__ == "__main__":
    unittest.main()
