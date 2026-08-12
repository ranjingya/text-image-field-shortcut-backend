from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from services.settings import get_app_settings


class OssSettingsTestCase(unittest.TestCase):
    def test_temporary_reference_settings_use_safe_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {"OSS_BUCKET_FOLDER_PREFIX": "images"},
            clear=True,
        ):
            settings = get_app_settings()

        self.assertEqual(settings.oss.bucket_prefix, "images")
        self.assertEqual(
            settings.oss.temporary_reference_prefix,
            "temp-references",
        )
        self.assertEqual(settings.oss.temporary_url_ttl_seconds, 3600)
        self.assertEqual(settings.oss.temporary_custom_domain, "")

    def test_temporary_reference_settings_are_loaded_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OSS_BUCKET_FOLDER_PREFIX": "/images/",
                "OSS_TEMP_FOLDER_PREFIX": "/temporary/",
                "OSS_TEMP_URL_TTL_SECONDS": "1800",
                "OSS_TEMP_CUSTOM_DOMAIN": "https://TEMP-IMAGES.EXAMPLE.COM/",
            },
            clear=True,
        ):
            settings = get_app_settings()

        self.assertEqual(settings.oss.bucket_prefix, "images")
        self.assertEqual(settings.oss.temporary_reference_prefix, "temporary")
        self.assertEqual(settings.oss.temporary_url_ttl_seconds, 1800)
        self.assertEqual(
            settings.oss.temporary_custom_domain,
            "https://temp-images.example.com",
        )

    def test_temporary_custom_domain_must_be_https_origin(self) -> None:
        invalid_domains = (
            "temp-images.example.com",
            "http://temp-images.example.com",
            "https://temp-images.example.com/path",
            "https://temp-images.example.com:443",
            "https://user@temp-images.example.com",
        )

        for custom_domain in invalid_domains:
            with self.subTest(custom_domain=custom_domain):
                with patch.dict(
                    os.environ,
                    {"OSS_TEMP_CUSTOM_DOMAIN": custom_domain},
                    clear=True,
                ):
                    with self.assertRaisesRegex(ValueError, "HTTPS 域名"):
                        get_app_settings()

    def test_temporary_reference_prefix_cannot_overlap_permanent_prefix(self) -> None:
        invalid_prefixes = ("images", "images/references", "temp")

        for temporary_prefix in invalid_prefixes:
            permanent_prefix = "images" if temporary_prefix != "temp" else "temp/images"
            with self.subTest(
                temporary_prefix=temporary_prefix,
                permanent_prefix=permanent_prefix,
            ):
                with patch.dict(
                    os.environ,
                    {
                        "OSS_BUCKET_FOLDER_PREFIX": permanent_prefix,
                        "OSS_TEMP_FOLDER_PREFIX": temporary_prefix,
                    },
                    clear=True,
                ):
                    with self.assertRaisesRegex(ValueError, "不能.*重叠"):
                        get_app_settings()

    def test_temporary_reference_prefix_cannot_be_empty(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OSS_BUCKET_FOLDER_PREFIX": "images",
                "OSS_TEMP_FOLDER_PREFIX": "///",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "不能为空"):
                get_app_settings()


if __name__ == "__main__":
    unittest.main()
