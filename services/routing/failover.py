from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from services.domain.errors import (
    ErrorCategory,
    ProviderError,
    sanitize_provider_error_message,
)
from services.domain.provider import (
    ImageProviderResult,
    ProviderClient,
    TextProviderResult,
)
from services.domain.requests import GenerateImageRequest, UnderstandImageRequest
from services.model_registry import (
    ModelRegistry,
    ModelRegistryError,
    load_model_registry,
)
from services.notifications import FeishuAlertNotifier, RoutingEventReporter
from services.providers.factory import build_provider_clients
from services.routing.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from services.settings import AppSettings
from services.state import build_state_store

logger = logging.getLogger(__name__)

TProviderResult = TypeVar("TProviderResult", ImageProviderResult, TextProviderResult)


@dataclass(frozen=True)
class RouteAttempt:
    provider: str
    provider_model: str
    attempt: int
    success: bool
    elapsed_ms: float
    error_category: str = ""


@dataclass(frozen=True)
class ImageRouteResult:
    provider_result: ImageProviderResult
    fallback_used: bool
    attempts: tuple[RouteAttempt, ...]


@dataclass(frozen=True)
class TextRouteResult:
    provider_result: TextProviderResult
    fallback_used: bool
    attempts: tuple[RouteAttempt, ...]


class FailoverExhaustedError(ProviderError):
    def __init__(self, errors: tuple[ProviderError, ...]) -> None:
        last_error = errors[-1]
        super().__init__(
            provider=last_error.provider,
            category=ErrorCategory.UPSTREAM_UNAVAILABLE,
            message="主服务商和兜底服务商当前均不可用。",
            status_code=503,
            retryable=True,
            request_id=last_error.request_id,
            cause=last_error,
        )
        self.errors = errors


class FailoverRouter:
    """按配置执行主服务商调用、重试和顺序兜底。"""

    def __init__(
        self,
        settings: AppSettings,
        registry: ModelRegistry,
        providers: dict[str, ProviderClient],
        circuit_breaker: CircuitBreaker | None = None,
        event_reporter: RoutingEventReporter | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._providers = providers
        self._configuration = registry.configuration
        self._circuit_breaker = circuit_breaker
        self._event_reporter = event_reporter

    def status(self) -> dict[str, object]:
        """返回不包含密钥和地址的服务商路由状态。

        返回值：
            包含功能开关、状态范围和各服务商熔断快照的字典。
        """
        capabilities = ("image_generation", "image_understanding")
        provider_states: dict[str, object] = {}
        for provider_name in self._configuration.providers:
            capability_states: dict[str, object] = {}
            for capability in capabilities:
                snapshot = (
                    self._circuit_breaker.snapshot(provider_name, capability)
                    if self._circuit_breaker
                    else None
                )
                capability_states[capability] = {
                    "state": snapshot.state if snapshot else "closed",
                    "failureCount": snapshot.failure_count if snapshot else 0,
                    "openCount": snapshot.open_count if snapshot else 0,
                    "openUntil": snapshot.open_until if snapshot else 0.0,
                }
            provider_states[provider_name] = capability_states
        return {
            "fallbackEnabled": self._settings.fallback_enabled,
            "alertEnabled": self._settings.alert.enabled,
            "stateStoreAvailable": bool(
                self._circuit_breaker and self._circuit_breaker.state_available
            ),
            "stateBackend": (
                self._circuit_breaker.state_backend
                if self._circuit_breaker
                else "disabled"
            ),
            "providers": provider_states,
        }

    def generate_image(
        self,
        request: GenerateImageRequest,
    ) -> ImageRouteResult:
        """执行图片生成主备路由。

        参数：
            request: 已完成业务校验的图片生成请求。

        返回值：
            包含服务商结果、是否兜底和完整尝试链路的路由结果。
        """
        public_model = self._registry.resolve(request.model)
        attempts: list[RouteAttempt] = []
        errors: list[ProviderError] = []

        def invoke(
            provider: ProviderClient,
            provider_model: str,
            timeout_seconds: float,
        ) -> ImageProviderResult:
            return provider.generate_image(
                request,
                public_model,
                provider_model,
                timeout_seconds,
            )

        result = self._route(
            capability="image_generation",
            public_model=public_model,
            invoke=invoke,
            is_empty=lambda item: (
                not any(
                    asset.asset_type in {"binary_file", "image_base64", "image_url"}
                    for asset in item.result.assets
                )
            ),
            attempts=attempts,
            errors=errors,
            request_id=request.request_id,
        )
        return ImageRouteResult(
            provider_result=result,
            fallback_used=result.provider != self._configuration.primary_provider,
            attempts=tuple(attempts),
        )

    def understand_image(self, request: UnderstandImageRequest) -> TextRouteResult:
        """执行图片理解主备路由。

        参数：
            request: 已完成业务校验的图片理解请求。

        返回值：
            包含服务商结果、是否兜底和完整尝试链路的路由结果。
        """
        public_model = self._registry.resolve(request.model)
        attempts: list[RouteAttempt] = []
        errors: list[ProviderError] = []

        def invoke(
            provider: ProviderClient,
            provider_model: str,
            timeout_seconds: float,
        ) -> TextProviderResult:
            return provider.understand_image(
                request,
                public_model,
                provider_model,
                timeout_seconds,
            )

        result = self._route(
            capability="image_understanding",
            public_model=public_model,
            invoke=invoke,
            is_empty=lambda item: not item.text.strip(),
            attempts=attempts,
            errors=errors,
            request_id=request.request_id,
        )
        return TextRouteResult(
            provider_result=result,
            fallback_used=result.provider != self._configuration.primary_provider,
            attempts=tuple(attempts),
        )

    def _route(
        self,
        *,
        capability: str,
        public_model: str,
        invoke: Callable[[ProviderClient, str, float], TProviderResult],
        is_empty: Callable[[TProviderResult], bool],
        attempts: list[RouteAttempt],
        errors: list[ProviderError],
        request_id: str,
    ) -> TProviderResult:
        provider_names = [self._configuration.primary_provider]
        if self._settings.fallback_enabled:
            provider_names.extend(self._configuration.fallback_providers)

        for provider_index, provider_name in enumerate(provider_names):
            if provider_index > 0 and not errors[-1].retryable:
                raise errors[-1]
            try:
                provider_model = self._resolve_provider_model(
                    public_model,
                    provider_name,
                    allow_unregistered=provider_index == 0,
                )
            except ModelRegistryError:
                if provider_index == 0:
                    raise
                logger.warning(
                    "provider.route.fallback_skipped: %s",
                    {
                        "provider": provider_name,
                        "publicModel": public_model,
                        "reason": "model_mapping_missing",
                    },
                )
                continue

            if provider_index > 0 and not self._registry.supports(
                public_model, capability
            ):
                continue
            fallback_trigger_category = (
                str(errors[-1].category) if provider_index > 0 and errors else ""
            )

            circuit_admission_state = CircuitState.CLOSED
            if self._circuit_breaker:
                try:
                    circuit_admission_state = self._circuit_breaker.before_call(
                        provider_name, capability
                    )
                except CircuitOpenError as error:
                    errors.append(error)
                    attempts.append(
                        RouteAttempt(
                            provider=provider_name,
                            provider_model=provider_model,
                            attempt=0,
                            success=False,
                            elapsed_ms=0.0,
                            error_category=error.category,
                        )
                    )
                    logger.warning(
                        "provider.route.circuit_skipped: %s",
                        {
                            "provider": provider_name,
                            "capability": capability,
                            "state": error.state,
                        },
                    )
                    continue

            max_attempts = (
                self._settings.routing.primary_max_attempts
                if provider_index == 0
                else self._settings.routing.fallback_max_attempts
            )
            empty_retries_remaining = (
                self._settings.routing.primary_empty_response_retry_count
                if provider_index == 0
                else 0
            )
            maximum_attempts = max_attempts + empty_retries_remaining
            provider_deadline = (
                time.monotonic() + self._settings.routing.provider_timeout_seconds
            )
            attempt_number = 0
            provider_error: ProviderError | None = None
            stop_after_provider = False
            while attempt_number < maximum_attempts:
                attempt_number += 1
                remaining_seconds = provider_deadline - time.monotonic()
                if remaining_seconds <= 0:
                    timeout_error = ProviderError(
                        provider=provider_name,
                        category=ErrorCategory.TIMEOUT,
                        message="当前服务商请求超过时限。",
                        retryable=True,
                        counts_toward_circuit=False,
                    )
                    errors.append(timeout_error)
                    provider_error = timeout_error
                    break
                started_at = time.perf_counter()
                logger.debug(
                    "provider.route.attempt.start: %s",
                    {
                        "provider": provider_name,
                        "publicModel": public_model,
                        "providerModel": provider_model,
                        "attempt": attempt_number,
                    },
                )
                try:
                    result = invoke(
                        self._providers[provider_name],
                        provider_model,
                        remaining_seconds,
                    )
                    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
                    if is_empty(result):
                        empty_error = ProviderError(
                            provider=provider_name,
                            category=ErrorCategory.EMPTY_RESPONSE,
                            message="服务商返回了空内容。",
                            retryable=True,
                        )
                        errors.append(empty_error)
                        provider_error = empty_error
                        attempts.append(
                            RouteAttempt(
                                provider=provider_name,
                                provider_model=provider_model,
                                attempt=attempt_number,
                                success=False,
                                elapsed_ms=elapsed_ms,
                                error_category=empty_error.category,
                            )
                        )
                        if empty_retries_remaining > 0:
                            empty_retries_remaining -= 1
                            continue
                        break

                    attempts.append(
                        RouteAttempt(
                            provider=provider_name,
                            provider_model=provider_model,
                            attempt=attempt_number,
                            success=True,
                            elapsed_ms=elapsed_ms,
                        )
                    )
                    logger.debug(
                        "provider.route.attempt.success: %s",
                        {
                            "provider": provider_name,
                            "publicModel": public_model,
                            "attempt": attempt_number,
                            "elapsedMs": elapsed_ms,
                            "fallbackUsed": provider_index > 0,
                        },
                    )
                    if self._circuit_breaker:
                        self._circuit_breaker.record_success(
                            provider_name,
                            capability,
                            circuit_admission_state,
                        )
                    if self._event_reporter:
                        if provider_index == 0:
                            self._event_reporter.on_primary_success(
                                provider_name,
                                capability,
                                public_model,
                                request_id,
                            )
                        else:
                            self._event_reporter.on_fallback_used(
                                self._configuration.primary_provider,
                                provider_name,
                                capability,
                                public_model,
                                fallback_trigger_category,
                                request_id,
                            )
                    return result
                except ProviderError as error:
                    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
                    detached_error = error.detached()
                    errors.append(detached_error)
                    provider_error = detached_error
                    attempts.append(
                        RouteAttempt(
                            provider=provider_name,
                            provider_model=provider_model,
                            attempt=attempt_number,
                            success=False,
                            elapsed_ms=elapsed_ms,
                            error_category=error.category,
                        )
                    )
                    failure_log = {
                        "provider": provider_name,
                        "publicModel": public_model,
                        "attempt": attempt_number,
                        "elapsedMs": elapsed_ms,
                        "errorCategory": error.category,
                        "statusCode": error.status_code,
                        "providerErrorType": error.provider_error_type,
                        "causeType": (
                            type(error.cause).__name__
                            if error.cause is not None
                            else ""
                        ),
                        "providerRequestId": error.request_id,
                        "message": sanitize_provider_error_message(error, 300),
                    }
                    if error.response_bytes is not None:
                        failure_log["responseBytes"] = error.response_bytes
                    logger.warning(
                        "provider.route.attempt.failed: %s",
                        failure_log,
                    )
                    if not error.retryable:
                        if provider_index == 0:
                            if self._circuit_breaker:
                                circuit_opened = self._circuit_breaker.record_failure(
                                    provider_name, capability, error
                                )
                                self._report_circuit_open(
                                    circuit_opened,
                                    provider_name,
                                    capability,
                                    public_model,
                                    error,
                                    request_id,
                                )
                            self._report_provider_failure(
                                provider_name,
                                capability,
                                public_model,
                                error,
                                request_id,
                            )
                            raise
                        stop_after_provider = True
                        break
                    if attempt_number < max_attempts:
                        if self._wait_for_retry(error, provider_deadline):
                            continue
                    break

            if provider_error and self._circuit_breaker:
                circuit_opened = self._circuit_breaker.record_failure(
                    provider_name, capability, provider_error
                )
                self._report_circuit_open(
                    circuit_opened,
                    provider_name,
                    capability,
                    public_model,
                    provider_error,
                    request_id,
                )
            if provider_error:
                self._report_provider_failure(
                    provider_name,
                    capability,
                    public_model,
                    provider_error,
                    request_id,
                )
            if stop_after_provider:
                break

        if len({error.provider for error in errors}) > 1:
            if self._event_reporter:
                self._event_reporter.on_all_providers_failed(
                    self._configuration.primary_provider,
                    capability,
                    public_model,
                    str(errors[-1].category),
                    request_id,
                )
            raise FailoverExhaustedError(tuple(errors))
        if errors:
            raise errors[-1]
        raise ProviderError(
            provider=self._configuration.primary_provider,
            category=ErrorCategory.CAPABILITY,
            message="当前模型没有可用的服务商映射。",
            retryable=False,
            counts_toward_circuit=False,
        )

    def _resolve_provider_model(
        self, public_model: str, provider: str, *, allow_unregistered: bool
    ) -> str:
        try:
            return self._registry.provider_model(public_model, provider)
        except ModelRegistryError:
            if allow_unregistered and public_model:
                return public_model
            raise

    def _report_provider_failure(
        self,
        provider: str,
        capability: str,
        public_model: str,
        error: ProviderError,
        request_id: str,
    ) -> None:
        if self._event_reporter:
            self._event_reporter.on_provider_failure(
                provider,
                capability,
                public_model,
                error,
                request_id,
            )

    def _report_circuit_open(
        self,
        opened: bool,
        provider: str,
        capability: str,
        public_model: str,
        error: ProviderError,
        request_id: str,
    ) -> None:
        if opened and self._event_reporter:
            self._event_reporter.on_circuit_open(
                provider,
                capability,
                public_model,
                str(error.category),
                request_id,
            )

    @staticmethod
    def _wait_for_retry(error: ProviderError, deadline: float) -> bool:
        delay = error.retry_after_seconds or 0.0
        if delay <= 0:
            return True
        remaining_seconds = deadline - time.monotonic()
        if delay >= remaining_seconds:
            return False
        time.sleep(delay)
        return True


def build_failover_router(settings: AppSettings) -> FailoverRouter:
    """根据应用配置构建故障转移路由器。

    参数：
        settings: 包含配置文件路径、服务商密钥和路由参数的应用设置。

    返回值：
        可执行图片生成与图片理解主备路由的路由器。
    """
    registry = load_model_registry(settings.provider_config_path)
    providers = dict(build_provider_clients(settings, registry.configuration))
    state_store = build_state_store()
    circuit_breaker = CircuitBreaker(state_store, settings.circuit)
    event_reporter = RoutingEventReporter(
        state_store,
        FeishuAlertNotifier(settings.alert, settings.http.notification),
        settings.alert,
    )
    return FailoverRouter(
        settings,
        registry,
        providers,
        circuit_breaker,
        event_reporter,
    )
