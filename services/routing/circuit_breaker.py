from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from services.domain.errors import ErrorCategory, ProviderError
from services.settings import CircuitBreakerSettings
from services.state import StateStore

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(ProviderError):
    def __init__(self, provider: str, capability: str, state: CircuitState) -> None:
        super().__init__(
            provider=provider,
            category=ErrorCategory.UPSTREAM_UNAVAILABLE,
            message="服务商熔断器当前不可用。",
            retryable=True,
            counts_toward_circuit=False,
        )
        self.capability = capability
        self.state = state


@dataclass(frozen=True)
class CircuitSnapshot:
    state: CircuitState
    failure_count: int
    open_count: int
    open_until: float


class CircuitBreaker:
    """按服务商和能力维护应用进程内的熔断状态。"""

    def __init__(
        self,
        store: StateStore,
        settings: CircuitBreakerSettings,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._settings = settings
        self._clock = clock

    @property
    def state_available(self) -> bool:
        return self._store.available

    @property
    def state_backend(self) -> str:
        return self._store.backend_name

    def before_call(self, provider: str, capability: str) -> CircuitState:
        """在服务商调用前检查熔断状态。

        参数：
            provider: 目标服务商名称。
            capability: 图片生成或图片理解能力名称。

        返回值：
            本次调用获得的熔断状态，关闭或半开状态允许调用。
        """
        if not self._store.available:
            return CircuitState.CLOSED
        snapshot = self.snapshot(provider, capability)
        if snapshot.state == CircuitState.CLOSED:
            return CircuitState.CLOSED
        if snapshot.state == CircuitState.OPEN and self._clock() < snapshot.open_until:
            raise CircuitOpenError(provider, capability, CircuitState.OPEN)

        lock_seconds = max(math.ceil(self._settings.open_seconds), 1)
        if not self._store.acquire_lock(
            self._key(provider, capability, "half_open_lock"), lock_seconds
        ):
            raise CircuitOpenError(provider, capability, CircuitState.HALF_OPEN)

        self._write_snapshot(
            provider,
            capability,
            CircuitSnapshot(
                state=CircuitState.HALF_OPEN,
                failure_count=snapshot.failure_count,
                open_count=snapshot.open_count,
                open_until=snapshot.open_until,
            ),
        )
        logger.info(
            "circuit.half_open: %s",
            {"provider": provider, "capability": capability},
        )
        return CircuitState.HALF_OPEN

    def record_success(
        self,
        provider: str,
        capability: str,
        admission_state: CircuitState,
    ) -> None:
        """记录成功调用，并仅由有效的半开探测关闭熔断器。

        参数：
            provider: 成功调用的服务商名称。
            capability: 成功调用的能力名称。
            admission_state: 本次调用在执行前获得的熔断准入状态。

        返回值：
            无。
        """
        if not self._store.available:
            return
        snapshot = self.snapshot(provider, capability)
        if snapshot.state != admission_state:
            logger.debug(
                "circuit.success.stale_ignored: %s",
                {
                    "provider": provider,
                    "capability": capability,
                    "admissionState": admission_state,
                    "currentState": snapshot.state,
                },
            )
            return
        self._store.delete(
            self._key(provider, capability, "snapshot"),
            self._key(provider, capability, "failures"),
            self._key(provider, capability, "open_count"),
            self._key(provider, capability, "half_open_lock"),
        )
        if snapshot.state != CircuitState.CLOSED:
            logger.info(
                "circuit.closed: %s",
                {"provider": provider, "capability": capability},
            )

    def record_failure(
        self, provider: str, capability: str, error: ProviderError
    ) -> bool:
        """记录服务商故障并在达到阈值时打开熔断器。

        参数：
            provider: 发生故障的服务商名称。
            capability: 发生故障的能力名称。
            error: 标准服务商错误。

        返回值：
            本次记录是否导致熔断器打开。
        """
        if not self._store.available or not self._should_count(error):
            return False

        snapshot = self.snapshot(provider, capability)
        immediate_open = error.category in {
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.BILLING,
            ErrorCategory.PERMISSION,
        }
        failure_count = self._store.increment(
            self._key(provider, capability, "failures"),
            self._settings.state_ttl_seconds,
        )
        if snapshot.state == CircuitState.HALF_OPEN:
            immediate_open = True
        if not immediate_open and failure_count < self._settings.failure_threshold:
            return False

        transition_key = self._key(provider, capability, "transition_lock")
        transition_lock_seconds = max(
            min(math.ceil(self._settings.open_seconds), 5), 1
        )
        if not self._store.acquire_lock(
            transition_key, transition_lock_seconds
        ):
            return False
        latest_snapshot = self.snapshot(provider, capability)
        if (
            latest_snapshot.state == CircuitState.OPEN
            and self._clock() < latest_snapshot.open_until
        ):
            self._store.delete(transition_key)
            return False

        open_count = self._store.increment(
            self._key(provider, capability, "open_count"),
            self._settings.state_ttl_seconds,
        )
        open_count = max(open_count, snapshot.open_count + 1, 1)
        open_seconds = min(
            self._settings.open_seconds * (2 ** (open_count - 1)),
            self._settings.max_open_seconds,
        )
        opened_snapshot = CircuitSnapshot(
            state=CircuitState.OPEN,
            failure_count=max(failure_count, 1),
            open_count=open_count,
            open_until=self._clock() + open_seconds,
        )
        self._write_snapshot(provider, capability, opened_snapshot)
        self._store.delete(
            self._key(provider, capability, "half_open_lock"),
            transition_key,
        )
        logger.warning(
            "circuit.opened: %s",
            {
                "provider": provider,
                "capability": capability,
                "failureCount": opened_snapshot.failure_count,
                "openCount": open_count,
                "openSeconds": open_seconds,
                "errorCategory": error.category,
            },
        )
        return True

    def snapshot(self, provider: str, capability: str) -> CircuitSnapshot:
        """读取指定服务商能力的熔断快照。

        参数：
            provider: 服务商名称。
            capability: 能力名称。

        返回值：
            当前熔断状态、故障数和打开时间。
        """
        if not self._store.available:
            return CircuitSnapshot(CircuitState.CLOSED, 0, 0, 0.0)
        raw_value = self._store.get(self._key(provider, capability, "snapshot"))
        if not raw_value:
            return CircuitSnapshot(
                CircuitState.CLOSED,
                self._read_count(self._key(provider, capability, "failures")),
                self._read_count(self._key(provider, capability, "open_count")),
                0.0,
            )
        try:
            payload = json.loads(raw_value)
            return CircuitSnapshot(
                state=CircuitState(payload.get("state", CircuitState.CLOSED)),
                failure_count=int(payload.get("failure_count", 0)),
                open_count=int(payload.get("open_count", 0)),
                open_until=float(payload.get("open_until", 0.0)),
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            logger.error(
                "circuit.snapshot.invalid: %s",
                {"provider": provider, "capability": capability},
            )
            return CircuitSnapshot(CircuitState.CLOSED, 0, 0, 0.0)

    def _write_snapshot(
        self, provider: str, capability: str, snapshot: CircuitSnapshot
    ) -> None:
        self._store.set(
            self._key(provider, capability, "snapshot"),
            json.dumps(
                {
                    "state": snapshot.state,
                    "failure_count": snapshot.failure_count,
                    "open_count": snapshot.open_count,
                    "open_until": snapshot.open_until,
                }
            ),
            self._settings.state_ttl_seconds,
        )

    def _read_count(self, key: str) -> int:
        raw_value = self._store.get(key)
        try:
            return int(raw_value or 0)
        except ValueError:
            return 0

    @staticmethod
    def _key(provider: str, capability: str, suffix: str) -> str:
        return f"circuit:{provider}:{capability}:{suffix}"

    @staticmethod
    def _should_count(error: ProviderError) -> bool:
        return error.counts_toward_circuit and error.category in {
            ErrorCategory.CONNECTION,
            ErrorCategory.TIMEOUT,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.UPSTREAM_UNAVAILABLE,
            ErrorCategory.INVALID_RESPONSE,
            ErrorCategory.EMPTY_RESPONSE,
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.BILLING,
            ErrorCategory.PERMISSION,
        }
