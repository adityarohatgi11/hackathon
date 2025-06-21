import sys
from pathlib import Path

# Add project root to sys.path for module imports in test environment
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Fallback benchmark fixture (if pytest-benchmark is not installed)
# ---------------------------------------------------------------------------

import pytest
import time


@pytest.fixture
def benchmark(request):  # type: ignore
    """Lightweight replacement for the pytest-benchmark plugin.

    Usage is intentionally identical to the original interface for the subset
    required by our test-suite: a callable wrapper that measures execution time
    and returns the function result.  It records execution time on the fixture
    object for optional inspection (avg, max, etc.).
    """

    class _BenchmarkProxy:
        def __init__(self, call):
            self._call = call
            self.stats = {}

        def __call__(self, func, *args, **kwargs):  # pylint: disable=missing-docstring
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            name = func.__name__
            self.stats.setdefault(name, []).append(elapsed)
            return result

        # Compatibility helpers used occasionally by tests
        @property
        def mean(self):
            all_vals = [v for lst in self.stats.values() for v in lst]
            return sum(all_vals) / len(all_vals) if all_vals else 0.0

        @property
        def max(self):
            all_vals = [v for lst in self.stats.values() for v in lst]
            return max(all_vals) if all_vals else 0.0

    return _BenchmarkProxy(request) 