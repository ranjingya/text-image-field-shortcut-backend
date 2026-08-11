from __future__ import annotations

import unittest

from services.domain.errors import ErrorCategory, ProviderError
from services.routing.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from services.settings import CircuitBreakerSettings


class MemoryStateStore:
    available = True

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.locks: set[str] = set()

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str, ttl_seconds: int) -> bool:
        self.values[key] = value
        return True

    def delete(self, *keys: str) -> bool:
        for key in keys:
            self.values.pop(key, None)
            self.locks.discard(key)
        return True

    def increment(self, key: str, ttl_seconds: int) -> int:
        value = int(self.values.get(key, "0")) + 1
        self.values[key] = str(value)
        return value

    def acquire_lock(self, key: str, ttl_seconds: int) -> bool:
        if key in self.locks:
            return False
        self.locks.add(key)
        return True


class CircuitBreakerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 1_000.0
        self.store = MemoryStateStore()
        self.breaker = CircuitBreaker(
            self.store,
            CircuitBreakerSettings(
                failure_threshold=2,
                open_seconds=10,
                max_open_seconds=40,
                state_ttl_seconds=300,
            ),
            clock=lambda: self.now,
        )

    def test_threshold_opens_and_successful_probe_closes_circuit(self) -> None:
        error = ProviderError(
            provider="easyrouter",
            category=ErrorCategory.TIMEOUT,
            message="timeout",
            retryable=True,
        )

        self.assertFalse(
            self.breaker.record_failure("easyrouter", "image_generation", error)
        )
        self.assertTrue(
            self.breaker.record_failure("easyrouter", "image_generation", error)
        )
        with self.assertRaises(CircuitOpenError):
            self.breaker.before_call("easyrouter", "image_generation")

        self.now += 11
        state = self.breaker.before_call("easyrouter", "image_generation")
        self.assertEqual(state, CircuitState.HALF_OPEN)

        self.breaker.record_success(
            "easyrouter", "image_generation", CircuitState.HALF_OPEN
        )
        self.assertEqual(
            self.breaker.snapshot("easyrouter", "image_generation").state,
            CircuitState.CLOSED,
        )

    def test_half_open_allows_only_one_probe(self) -> None:
        error = ProviderError(
            provider="easyrouter",
            category=ErrorCategory.AUTHENTICATION,
            message="invalid key",
            retryable=True,
        )
        self.breaker.record_failure("easyrouter", "image_generation", error)
        self.now += 11

        self.assertEqual(
            self.breaker.before_call("easyrouter", "image_generation"),
            CircuitState.HALF_OPEN,
        )
        with self.assertRaises(CircuitOpenError):
            self.breaker.before_call("easyrouter", "image_generation")

    def test_local_capacity_does_not_open_circuit(self) -> None:
        error = ProviderError(
            provider="easyrouter",
            category=ErrorCategory.LOCAL_CAPACITY,
            message="pool busy",
            retryable=True,
            counts_toward_circuit=False,
        )

        for _ in range(5):
            self.assertFalse(
                self.breaker.record_failure("easyrouter", "image_generation", error)
            )

        self.assertEqual(
            self.breaker.before_call("easyrouter", "image_generation"),
            CircuitState.CLOSED,
        )

    def test_repeated_open_uses_exponential_duration(self) -> None:
        error = ProviderError(
            provider="easyrouter",
            category=ErrorCategory.TIMEOUT,
            message="timeout",
            retryable=True,
        )
        self.breaker.record_failure("easyrouter", "image_generation", error)
        self.breaker.record_failure("easyrouter", "image_generation", error)
        first = self.breaker.snapshot("easyrouter", "image_generation")

        self.now = first.open_until + 1
        self.breaker.before_call("easyrouter", "image_generation")
        self.breaker.record_failure("easyrouter", "image_generation", error)
        second = self.breaker.snapshot("easyrouter", "image_generation")

        self.assertEqual(first.open_until, 1_010.0)
        self.assertEqual(second.open_until, self.now + 20)

    def test_closed_circuit_success_does_not_emit_info_log(self) -> None:
        with self.assertNoLogs(
            "services.routing.circuit_breaker", level="INFO"
        ):
            self.breaker.record_success(
                "easyrouter", "image_generation", CircuitState.CLOSED
            )

    def test_in_flight_success_does_not_close_newly_opened_circuit(self) -> None:
        error = ProviderError(
            provider="easyrouter",
            category=ErrorCategory.RATE_LIMIT,
            message="limited",
            retryable=True,
        )
        admission_state = self.breaker.before_call(
            "easyrouter", "image_generation"
        )
        self.breaker.record_failure("easyrouter", "image_generation", error)
        self.breaker.record_failure("easyrouter", "image_generation", error)

        self.breaker.record_success(
            "easyrouter", "image_generation", admission_state
        )

        snapshot = self.breaker.snapshot(
            "easyrouter", "image_generation"
        )
        self.assertEqual(snapshot.state, CircuitState.OPEN)
        self.assertEqual(snapshot.open_until, 1_010.0)


if __name__ == "__main__":
    unittest.main()
