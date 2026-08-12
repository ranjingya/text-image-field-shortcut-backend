from __future__ import annotations

import json
import unittest

from api.request_logging import build_result_log_summary
from services.gemini_service import GeminiInvocationPlan, PreparedReferenceInput
from services.openai_image_service import OpenAIImageInvocationPlan
from services.domain.requests import GenerateImageRequest, UnderstandImageRequest
from services.pipelines.image import GenerationQueueSummary


class LogRedactionTestCase(unittest.TestCase):
    def test_request_summaries_exclude_prompt_and_reference_url(self) -> None:
        prompt = "不应进入日志的提示词"
        reference_url = "https://assets.example/image.png?token=secret"
        generate_request = GenerateImageRequest(
            request_id="request-1",
            prompt=prompt,
            model="gemini-3.1-flash-image",
            aspect_ratio="1:1",
            image_size="1K",
            input_type="file_url",
            file_urls=[reference_url],
            files=[],
        )
        understand_request = UnderstandImageRequest(
            request_id="request-2",
            prompt=prompt,
            model="gemini-3.1-flash-image",
            file_urls=[reference_url],
        )

        serialized = json.dumps(
            [generate_request.to_dict(), understand_request.to_dict()],
            ensure_ascii=False,
        )

        self.assertNotIn(prompt, serialized)
        self.assertNotIn(reference_url, serialized)

    def test_completed_summary_uses_resolved_model_without_oss_url(self) -> None:
        request = GenerateImageRequest(
            request_id="request-3",
            prompt="提示词",
            model="gemini-3-pro-image-preview",
            aspect_ratio="1:1",
            image_size="1K",
            input_type="empty",
            file_urls=[],
            files=[],
        )

        summary = build_result_log_summary(
            success=True,
            status_code=200,
            message="ok",
            normalized_request=request,
            result={
                "model": "gemini-3-pro-image",
                "ossUrl": "https://bucket.example/private.png",
                "ossUrls": ["https://bucket.example/private.png"],
                "provider": "easyrouter",
                "fallbackUsed": False,
            },
            queue_summary=GenerationQueueSummary(
                queued=True,
                queued_image_count=2,
                max_queue_wait_ms=456.78,
            ),
            elapsed_ms=123.45,
        )

        self.assertEqual(summary["requestedModel"], request.model)
        self.assertEqual(summary["resolvedModel"], "gemini-3-pro-image")
        self.assertEqual(summary["totalElapsedMs"], 123.45)
        self.assertEqual(summary["ossObjectCount"], 1)
        self.assertTrue(summary["queued"])
        self.assertEqual(summary["queuedImageCount"], 2)
        self.assertEqual(summary["maxQueueWaitMs"], 456.78)
        self.assertNotIn("ossUrl", summary)

    def test_invocation_plan_summaries_exclude_sensitive_content(self) -> None:
        prompt = "不应进入日志的提示词"
        reference_url = "https://assets.example/image.png?token=secret"
        reference = PreparedReferenceInput(
            source_type="url",
            mime_type="image/png",
            file_name="image.png",
            payload=b"image",
            source_ref=reference_url,
        )
        gemini_plan = GeminiInvocationPlan(
            api_url="https://provider.example/v1/models/test:generateContent",
            api_path="/v1/models/test:generateContent",
            model="test",
            prompt=prompt,
            prepared_inputs=[reference],
            request_body={"contents": [{"parts": [{"text": prompt}]}]},
        )
        openai_plan = OpenAIImageInvocationPlan(
            api_url="https://provider.example/v1/images/generations",
            model="gpt-image-2",
            prompt=prompt,
            size="1024x1024",
            n=1,
            quality="auto",
            request_body={"model": "gpt-image-2", "prompt": prompt},
        )

        serialized = json.dumps(
            [gemini_plan.to_dict(), openai_plan.to_dict()],
            ensure_ascii=False,
        )

        self.assertNotIn(prompt, serialized)
        self.assertNotIn(reference_url, serialized)


if __name__ == "__main__":
    unittest.main()
