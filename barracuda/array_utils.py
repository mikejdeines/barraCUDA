"""Shared array dtype and chunk-size helpers for barraCUDA."""

from __future__ import annotations

import os
from typing import Any

import dask.array as da
import numpy as np


def storage_dtype_from_env() -> np.dtype:
	"""Return the persisted image dtype.

	Defaulting to fp16 halves the Zarr footprint relative to the previous float32
	outputs while the processing code can still do numerically sensitive work in
	float32 before the final storage cast.
	"""
	value = os.environ.get("BARRACUDA_STORAGE_DTYPE", "float16").strip().lower()
	aliases = {
		"fp16": "float16",
		"half": "float16",
		"fp32": "float32",
		"single": "float32",
	}
	value = aliases.get(value, value)
	try:
		dtype = np.dtype(value)
	except TypeError as exc:
		raise ValueError(f"BARRACUDA_STORAGE_DTYPE must be a NumPy dtype, got {value!r}") from exc
	if not np.issubdtype(dtype, np.floating):
		raise ValueError(f"BARRACUDA_STORAGE_DTYPE must be a floating dtype, got {dtype}")
	return dtype


def cast_array_for_storage(array: da.Array, dtype: np.dtype) -> da.Array:
	"""Cast a Dask array to a storage dtype without producing fp16 infinities."""
	dtype = np.dtype(dtype)
	out = da.asarray(array)
	if np.issubdtype(dtype, np.floating):
		info = np.finfo(dtype)
		out = da.clip(out, info.min, info.max)
	return out.astype(dtype)


def cast_numpy_block_for_storage(block: Any, dtype: np.dtype) -> np.ndarray:
	"""Cast one in-memory block to storage dtype with finite-range clipping."""
	dtype = np.dtype(dtype)
	arr = np.asarray(block)
	if np.issubdtype(dtype, np.floating):
		info = np.finfo(dtype)
		arr = np.clip(arr, info.min, info.max)
	return arr.astype(dtype, copy=False)


try:
	from .tracing import instrument_module_functions
except ImportError:  # pragma: no cover - supports direct module execution
	from tracing import instrument_module_functions

instrument_module_functions(globals())
