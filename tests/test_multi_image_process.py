from __future__ import annotations

import base64
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from flask import Flask, request

from api.parsers import parse_generate_image_request
from services.domain.errors import ProviderError
from services.domain.requests import GenerateImageRequest
from services.generation_gate import GenerationGate
from services.gemini_service import GeminiRawResponse
from services.http import FetchedAsset
from services.pipelines.image import (
    _build_batch_prompt,
    _generate_batch_item,
    generate_image_only,
    process_image_request,
)
from services.response_normalizer import (
    NormalizedGeneratedAsset,
    NormalizedModelResult,
    normalize_gemini_response,
)
from services.settings import get_app_settings as load_app_settings


def _build_json_response(payload: dict[str, object]) -> GeminiRawResponse:
    return GeminiRawResponse(
        status_code=200,
        content_type="application/json",
        content_disposition="",
        body=json.dumps(payload).encode("utf-8"),
    )


class MultiImageResponseNormalizerTestCase(unittest.TestCase):
    def test_collects_all_gemini_inline_images_in_response_order(self) -> None:
        raw_response = _build_json_response(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "第一张"},
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": base64.b64encode(b"first").decode(
                                            "ascii"
                                        ),
                                    }
                                },
                                {"text": "第二张"},
                                {
                                    "inline_data": {
                                        "mime_type": "image/jpeg",
                                        "data": base64.b64encode(b"second").decode(
                                            "ascii"
                                        ),
                                    }
                                },
                            ]
                        }
                    }
                ]
            }
        )

        with self.assertNoLogs(
            "services.response_normalizer",
            level="INFO",
        ):
            result = normalize_gemini_response(raw_response)

        self.assertEqual(result.raw_response_type, "json_base64")
        self.assertEqual(
            [asset.payload for asset in result.assets],
            [b"first", b"second"],
        )
        self.assertEqual(
            [asset.mime_type for asset in result.assets],
            ["image/png", "image/jpeg"],
        )
        self.assertEqual(
            [asset.file_name for asset in result.assets],
            ["generated-output-1.png", "generated-output-2.jpg"],
        )

    def test_collects_all_image_urls_in_response_order(self) -> None:
        raw_response = _build_json_response(
            {
                "data": [
                    {"url": "https://assets.example/first.png"},
                    {"url": "https://assets.example/second.png"},
                ]
            }
        )

        with self.assertNoLogs(
            "services.response_normalizer",
            level="INFO",
        ):
            result = normalize_gemini_response(raw_response)

        self.assertEqual(result.raw_response_type, "json_url")
        self.assertEqual(
            [asset.payload for asset in result.assets],
            [
                "https://assets.example/first.png",
                "https://assets.example/second.png",
            ],
        )


class MultiImageProcessPipelineTestCase(unittest.TestCase):
    @patch("services.pipelines.image.get_app_settings")
    @patch("services.pipelines.image.build_failover_router")
    @patch("services.pipelines.image.upload_asset_to_oss")
    def test_uploads_every_image_and_returns_all_urls(
        self,
        upload_asset_to_oss,
        build_failover_router,
        get_app_settings,
    ) -> None:
        request_data = GenerateImageRequest(
            request_id="request-1",
            prompt="生成图片",
            model="gemini-3.1-flash-image",
            aspect_ratio="1:1",
            image_size="1K",
            input_type="empty",
            file_urls=[],
            files=[],
            image_count=2,
        )
        assets = {
            index: NormalizedGeneratedAsset(
                asset_type="image_base64",
                mime_type="image/png",
                file_name=f"generated-output-{index + 1}.png",
                source_kind="bytes",
                payload=payload,
            )
            for index, payload in enumerate((b"first", b"second"))
        }
        started = threading.Barrier(2)
        received_prompts = {}

        def generate_image(item_request, *, deadline=None):
            item_index = int(item_request.request_id.rsplit(":", 1)[1]) - 1
            received_prompts[item_index] = item_request.prompt
            self.assertIsNotNone(deadline)
            started.wait(timeout=1)
            provider_result = SimpleNamespace(
                public_model="gemini-3.1-flash-image",
                provider="easyrouter",
                result=NormalizedModelResult(
                    raw_response_type="json_base64",
                    assets=[assets[item_index]],
                    text_output="",
                    raw_meta={},
                ),
            )
            return SimpleNamespace(
                provider_result=provider_result,
                fallback_used=False,
            )

        settings = SimpleNamespace(
            routing=SimpleNamespace(request_deadline_seconds=30),
            image_generation=SimpleNamespace(
                max_count=5,
                max_concurrency=5,
                queue_timeout_seconds=10,
            )
        )
        get_app_settings.return_value = settings
        build_failover_router.return_value.generate_image.side_effect = generate_image
        upload_asset_to_oss.side_effect = [
            SimpleNamespace(
                object_key="images/first.png",
                object_url="https://bucket.example/images/first.png",
            ),
            SimpleNamespace(
                object_key="images/second.png",
                object_url="https://bucket.example/images/second.png",
            ),
        ]

        with self.assertNoLogs(
            "services.pipelines.image",
            level="INFO",
        ):
            result = process_image_request(request_data)

        self.assertEqual(
            build_failover_router.return_value.generate_image.call_count,
            2,
        )
        self.assertEqual(upload_asset_to_oss.call_count, 2)
        self.assertEqual(
            received_prompts,
            {
                0: (
                    "这是本批次第 1 张，共 2 张。"
                    "若提示词包含分图要求，只执行第 1 张对应的要求；"
                    "仅生成一张完整图片，禁止拼图或显示序号。\n\n"
                    "生成图片"
                ),
                1: (
                    "这是本批次第 2 张，共 2 张。"
                    "若提示词包含分图要求，只执行第 2 张对应的要求；"
                    "仅生成一张完整图片，禁止拼图或显示序号。\n\n"
                    "生成图片"
                ),
            },
        )
        self.assertEqual(request_data.prompt, "生成图片")
        self.assertEqual(result.data["requestedCount"], 2)
        self.assertEqual(result.data["generatedCount"], 2)
        self.assertEqual(
            result.data["ossUrls"],
            [
                "https://bucket.example/images/first.png",
                "https://bucket.example/images/second.png",
            ],
        )
        self.assertEqual(
            result.data["ossUrl"],
            "https://bucket.example/images/first.png",
        )
        self.assertFalse(result.queue_summary.queued)
        self.assertEqual(result.queue_summary.queued_image_count, 0)
        self.assertEqual(result.queue_summary.max_queue_wait_ms, 0.0)
        get_app_settings.assert_called_once_with()

    def test_single_image_keeps_original_prompt(self) -> None:
        self.assertEqual(
            _build_batch_prompt("生成一张产品图", 0, 1),
            "生成一张产品图",
        )

    @patch("services.pipelines.image.get_app_settings")
    def test_rejects_image_count_over_configured_limit(
        self,
        get_app_settings,
    ) -> None:
        request_data = GenerateImageRequest(
            request_id="request-2",
            prompt="生成图片",
            model="gemini-3.1-flash-image",
            aspect_ratio="1:1",
            image_size="1K",
            input_type="empty",
            file_urls=[],
            files=[],
            image_count=6,
        )
        get_app_settings.return_value = SimpleNamespace(
            routing=SimpleNamespace(request_deadline_seconds=30),
            image_generation=SimpleNamespace(
                max_count=5,
                max_concurrency=5,
                queue_timeout_seconds=10,
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "imageCount must be between 1 and 5",
        ):
            process_image_request(request_data)

    def test_binary_endpoint_rejects_multi_image_request(self) -> None:
        request_data = GenerateImageRequest(
            request_id="request-3",
            prompt="生成图片",
            model="gemini-3.1-flash-image",
            aspect_ratio="1:1",
            image_size="1K",
            input_type="empty",
            file_urls=[],
            files=[],
            image_count=2,
        )

        with self.assertRaisesRegex(
            ValueError,
            "only supported by /api/process-image",
        ):
            generate_image_only(request_data)

    @patch("services.pipelines.image.get_app_settings")
    @patch("services.pipelines.image.build_failover_router")
    @patch("services.pipelines.image.upload_asset_to_oss")
    @patch("services.reference_images.build_asset_fetcher")
    @patch("services.reference_images.TemporaryReferenceStore")
    def test_downloads_reference_url_once_before_multi_image_generation(
        self,
        temporary_reference_store,
        build_asset_fetcher,
        upload_asset_to_oss,
        build_failover_router,
        get_app_settings,
    ) -> None:
        request_data = GenerateImageRequest(
            request_id="request-reference",
            prompt="生成图片",
            model="gemini-3.1-flash-image",
            aspect_ratio="1:1",
            image_size="1K",
            input_type="file_url",
            file_urls=["https://assets.example/reference.png"],
            files=[],
            image_count=2,
        )
        settings = SimpleNamespace(
            routing=SimpleNamespace(request_deadline_seconds=30),
            image_generation=SimpleNamespace(
                max_count=5,
                max_concurrency=5,
                queue_timeout_seconds=10,
            ),
        )
        get_app_settings.return_value = settings
        build_asset_fetcher.return_value.fetch.return_value = FetchedAsset(
            body=b"reference-image",
            content_type="image/png",
            final_url="https://assets.example/reference.png",
        )
        temporary_object = SimpleNamespace(
            object_key="temp-references/batch/reference.png",
            signed_url=(
                "https://bucket.example/temp-references/reference.png"
                "?signature=secret"
            ),
            mime_type="image/png",
            content_length=len(b"reference-image"),
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

        def generate_image(item_request, *, deadline=None):
            self.assertEqual(item_request.file_urls, [])
            self.assertEqual(item_request.files, [])
            self.assertEqual(
                item_request.reference_images[0].url,
                temporary_object.signed_url,
            )
            self.assertEqual(
                item_request.reference_images[0].mime_type,
                "image/png",
            )
            temporary_reference_store.return_value.delete_many.assert_not_called()
            return SimpleNamespace(
                provider_result=SimpleNamespace(
                    public_model="gemini-3.1-flash-image",
                    provider="easyrouter",
                    result=NormalizedModelResult(
                        raw_response_type="json_base64",
                        assets=[
                            NormalizedGeneratedAsset(
                                asset_type="image_base64",
                                mime_type="image/png",
                                file_name="generated.png",
                                source_kind="bytes",
                                payload=b"generated",
                            )
                        ],
                        text_output="",
                        raw_meta={},
                    ),
                ),
                fallback_used=False,
            )

        build_failover_router.return_value.generate_image.side_effect = generate_image
        upload_asset_to_oss.side_effect = [
            SimpleNamespace(
                object_key=f"images/{index}.png",
                object_url=f"https://bucket.example/images/{index}.png",
            )
            for index in range(2)
        ]

        process_image_request(request_data)

        build_asset_fetcher.return_value.fetch.assert_called_once_with(
            "https://assets.example/reference.png"
        )
        temporary_reference_store.return_value.upload.assert_called_once()
        uploaded_body, uploaded_mime_type, _ = (
            temporary_reference_store.return_value.upload.call_args.args
        )
        self.assertEqual(uploaded_body, b"reference-image")
        self.assertEqual(uploaded_mime_type, "image/png")
        temporary_reference_store.return_value.delete_many.assert_called_once_with(
            [temporary_object]
        )
        self.assertEqual(
            build_failover_router.return_value.generate_image.call_count,
            2,
        )
        first_request = build_failover_router.return_value.generate_image.call_args_list[
            0
        ].args[0]
        second_request = build_failover_router.return_value.generate_image.call_args_list[
            1
        ].args[0]
        self.assertIs(
            first_request.reference_images,
            second_request.reference_images,
        )


class GenerationGateTestCase(unittest.TestCase):
    def test_available_slot_is_not_reported_as_queued(self) -> None:
        gate = GenerationGate(1)

        with gate.acquire(
            timeout_seconds=0.1,
            request_id="request-immediate",
            image_index=0,
        ) as admission:
            self.assertFalse(admission.queued)
            self.assertEqual(admission.wait_ms, 0.0)

    def test_waited_slot_is_reported_as_queued(self) -> None:
        gate = GenerationGate(1)
        release_slot = threading.Event()
        slot_acquired = threading.Event()

        def hold_slot() -> None:
            with gate.acquire(
                timeout_seconds=0.1,
                request_id="request-holder",
                image_index=0,
            ):
                slot_acquired.set()
                release_slot.wait(timeout=0.2)

        holder = threading.Thread(target=hold_slot)
        holder.start()
        self.assertTrue(slot_acquired.wait(timeout=0.1))
        timer = threading.Timer(0.01, release_slot.set)
        timer.start()

        with gate.acquire(
            timeout_seconds=0.1,
            request_id="request-waited",
            image_index=1,
        ) as admission:
            self.assertTrue(admission.queued)
            self.assertGreater(admission.wait_ms, 0.0)

        holder.join(timeout=0.1)
        timer.join(timeout=0.1)
        self.assertFalse(holder.is_alive())

    def test_queue_timeout_returns_retryable_local_capacity_error(self) -> None:
        gate = GenerationGate(1)

        with gate.acquire(
            timeout_seconds=0.1,
            request_id="request-queue",
            image_index=0,
        ):
            with self.assertRaisesRegex(ProviderError, "排队超时") as raised:
                with gate.acquire(
                    timeout_seconds=0.01,
                    request_id="request-queue",
                    image_index=1,
                ):
                    pass

        self.assertTrue(raised.exception.retryable)


class GenerationBudgetTestCase(unittest.TestCase):
    def test_default_queue_timeout_is_420_seconds(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            settings = load_app_settings()

        self.assertEqual(
            settings.image_generation.queue_timeout_seconds,
            420.0,
        )

    @patch("services.pipelines.image.time.monotonic")
    def test_execution_budget_starts_after_queue_admission(self, monotonic) -> None:
        request_data = GenerateImageRequest(
            request_id="request-budget",
            prompt="生成图片",
            model="gemini-3.1-flash-image",
            aspect_ratio="1:1",
            image_size="1K",
            input_type="empty",
            file_urls=[],
            files=[],
            image_count=1,
        )
        asset = NormalizedGeneratedAsset(
            asset_type="image_base64",
            mime_type="image/png",
            file_name="generated.png",
            source_kind="bytes",
            payload=b"image",
        )
        router = MagicMock()
        router.generate_image.return_value = SimpleNamespace(
            provider_result=SimpleNamespace(
                public_model="gemini-3.1-flash-image",
                provider="easyrouter",
                result=NormalizedModelResult(
                    raw_response_type="json_base64",
                    assets=[asset],
                    text_output="",
                    raw_meta={},
                ),
            ),
            fallback_used=False,
        )
        generation_gate = MagicMock()
        gate_entered = False

        def enter_gate():
            nonlocal gate_entered
            gate_entered = True
            return SimpleNamespace(queued=True, wait_ms=12_000.0)

        generation_gate.acquire.return_value.__enter__.side_effect = enter_gate

        def current_time() -> float:
            self.assertTrue(gate_entered)
            return 100.0

        monotonic.side_effect = current_time

        result = _generate_batch_item(
            router,
            request_data,
            0,
            generation_gate,
            execution_timeout_seconds=390.0,
            queue_timeout_seconds=420.0,
        )

        generation_gate.acquire.assert_called_once_with(
            timeout_seconds=420.0,
            request_id="request-budget",
            image_index=0,
        )
        self.assertEqual(
            router.generate_image.call_args.kwargs["deadline"],
            490.0,
        )
        self.assertTrue(result.queued)
        self.assertEqual(result.queue_wait_ms, 12_000.0)


class MultiImageRequestParserTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)

    def test_parses_json_image_count(self) -> None:
        with self.app.test_request_context(
            json={
                "prompt": "生成图片",
                "imageCount": 3,
            }
        ):
            parsed = parse_generate_image_request(request)

        self.assertEqual(parsed.image_count, 3)
        self.assertEqual(parsed.to_dict()["imageCount"], 3)

    def test_defaults_image_count_to_one(self) -> None:
        with self.app.test_request_context(json={"prompt": "生成图片"}):
            parsed = parse_generate_image_request(request)

        self.assertEqual(parsed.image_count, 1)

    def test_rejects_non_positive_image_count(self) -> None:
        with self.app.test_request_context(
            json={
                "prompt": "生成图片",
                "imageCount": 0,
            }
        ):
            with self.assertRaisesRegex(
                ValueError,
                "imageCount must be a positive integer",
            ):
                parse_generate_image_request(request)


if __name__ == "__main__":
    unittest.main()
