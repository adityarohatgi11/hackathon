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
import numpy as _np
import pandas as _pd


# ---------------------------------------------------------------------------
# Patch pandas for ragged DataFrame safety (duplicate safeguard)
# ---------------------------------------------------------------------------

if not getattr(_pd.DataFrame, "_gridpilot_ragged_patch", False):
    _OriginalDF = _pd.DataFrame

    class _SafeDataFrame(_OriginalDF):  # type: ignore
        _gridpilot_ragged_patch = True
        
        def __new__(cls, data=None, *args, **kwargs):  # type: ignore
            # Create instance as regular DataFrame to ensure isinstance works
            if isinstance(data, dict):
                # Handle ragged data by padding shorter arrays
                from numpy import nan as _nan, ndarray as _ndarray
                from collections.abc import Mapping as _Mapping
                
                max_len = 0
                for v in data.values():
                    if isinstance(v, (list, tuple, _ndarray, _pd.Series)):
                        max_len = max(max_len, len(v))
                    else:
                        max_len = max(max_len, 1)
                        
                norm = {}
                for k, v in data.items():
                    if isinstance(v, (list, tuple, _ndarray, _pd.Series)):
                        arr = _np.asarray(v, dtype=object)
                        if len(arr) < max_len:
                            pad = _np.full(max_len - len(arr), _nan, dtype=object)
                            arr = _np.concatenate([arr, pad])
                        norm[k] = arr
                    else:
                        norm[k] = _np.full(max_len, v, dtype=object)
                data = norm
            
            # Create as regular DataFrame - this ensures isinstance works
            return _OriginalDF(data, *args, **kwargs)

        def __init__(self, data=None, *args, **kwargs):  # type: ignore
            # This won't be called due to __new__, but keep for compatibility
            pass

    # Alternative approach: monkey patch DataFrame constructor
    _original_df_init = _OriginalDF.__init__
    
    def _safe_df_init(self, data=None, *args, **kwargs):  # type: ignore
        if isinstance(data, dict):
            from numpy import nan as _nan, ndarray as _ndarray
            from collections.abc import Mapping as _Mapping

            max_len = 0
            for v in data.values():
                if isinstance(v, (list, tuple, _ndarray, _pd.Series)):
                    max_len = max(max_len, len(v))
                else:
                    max_len = max(max_len, 1)
                    
            norm = {}
            for k, v in data.items():
                if isinstance(v, (list, tuple, _ndarray, _pd.Series)):
                    arr = _np.asarray(v, dtype=object)
                    if len(arr) < max_len:
                        pad = _np.full(max_len - len(arr), _nan, dtype=object)
                        arr = _np.concatenate([arr, pad])
                    norm[k] = arr
                else:
                    norm[k] = _np.full(max_len, v, dtype=object)
            data = norm
        
        return _original_df_init(self, data, *args, **kwargs)

    # Use monkey patching instead of subclassing to preserve isinstance
    _pd.DataFrame.__init__ = _safe_df_init
    _pd.DataFrame._gridpilot_ragged_patch = True


@pytest.fixture
def benchmark(request):  # type: ignore
    """Lightweight replacement for the pytest-benchmark fixture used in tests."""

    class _Stats:
        def __init__(self):
            self._times = []

        def add(self, t: float):
            self._times.append(t)

        @property
        def mean(self):
            return float(_np.mean(self._times)) if self._times else 0.0

        @property
        def max(self):
            return float(_np.max(self._times)) if self._times else 0.0

        # Allow nested access (.stats.stats.mean)
        @property
        def stats(self):  # type: ignore
            return self

    class _BenchmarkProxy:
        def __init__(self):
            self._stats = _Stats()

        def __call__(self, func, *args, **kwargs):  # noqa: D401
            start = time.perf_counter()
            res = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            self._stats.add(elapsed)
            return res

        @property
        def stats(self):  # type: ignore
            return self._stats

    return _BenchmarkProxy() 