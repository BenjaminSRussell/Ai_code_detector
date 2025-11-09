"""
HTTP request rate limiter with token bucket algorithm.
"""

import time
from collections import defaultdict
from threading import Lock


class RateLimiter:
    """Token bucket rate limiter for API throttling."""

    def __init__(self, rate_per_second, burst_size=None):
        self._rate = rate_per_second
        self._burst = burst_size or rate_per_second
        self._tokens = defaultdict(lambda: self._burst)
        self._last_update = defaultdict(lambda: time.time())
        self._lock = Lock()

    def allow(self, key):
        """Check if request should be allowed."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update[key]

            # Refill tokens based on elapsed time
            tokens = min(
                self._burst,
                self._tokens[key] + elapsed * self._rate
            )

            if tokens >= 1.0:
                self._tokens[key] = tokens - 1.0
                self._last_update[key] = now
                return True
            else:
                self._tokens[key] = tokens
                return False

    def reset(self, key):
        """Reset rate limit for specific key."""
        with self._lock:
            if key in self._tokens:
                del self._tokens[key]
                del self._last_update[key]
