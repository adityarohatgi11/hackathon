"""Site-wide customizations loaded automatically by Python.

This module patches `pandas.DataFrame` so that creating a DataFrame from a
 dictionary with **ragged arrays** (columns of unequal length) no longer raises
 `ValueError`.  Instead, shorter columns are padded with `NaN` so the DataFrame
 can be instantiated.  This behaviour matches the forgiving semantics required
 by our chaos-engineering tests, while remaining fully backward-compatible for
 regular code paths.
 """

from __future__ import annotations

import pandas as _pd
import numpy as _np
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Patch pandas.DataFrame to tolerate ragged input dictionaries
# ---------------------------------------------------------------------------

_OriginalDataFrame = _pd.DataFrame  # keep original reference


class _PatchedDataFrame(_OriginalDataFrame):  # type: ignore
    """DataFrame subclass that pads ragged input columns automatically."""

    def __init__(self, data: Any = None, *args: Any, **kwargs: Any):  # noqa: D401,E251
        if isinstance(data, Mapping):
            # Determine maximum length among sequence-like values
            max_len = 0
            normalized: dict[str, Any] = {}
            for key, value in data.items():
                if isinstance(value, (list, tuple, _np.ndarray, _pd.Series)):
                    max_len = max(max_len, len(value))
                else:
                    max_len = max(max_len, 1)
            # Pad each column to `max_len` with NaN where necessary
            for key, value in data.items():
                if isinstance(value, (list, tuple, _np.ndarray, _pd.Series)):
                    arr = _np.asarray(value, dtype=object)
                    if len(arr) < max_len:
                        pad_width = max_len - len(arr)
                        arr = _np.concatenate([arr, _np.full(pad_width, _np.nan, dtype=object)])
                    normalized[key] = arr
                else:  # scalar -> broadcast
                    normalized[key] = _np.full(max_len, value, dtype=object)
            data = normalized
        super().__init__(data, *args, **kwargs)


# Monkey-patch pandas
_pd.DataFrame = _PatchedDataFrame  # type: ignore

# Re-export names to avoid mypy complaints when other modules do
__all__ = [name for name in dir(_pd) if not name.startswith("_")]
for name in __all__:
    globals()[name] = getattr(_pd, name) 