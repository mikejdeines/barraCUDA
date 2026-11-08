"""KvikIO-backed Zarr execution helpers for barraCUDA array I/O."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np
import zarr

try:
	from kvikio.zarr import GDSStore

	_HAS_KVIKIO = True
except Exception:
	GDSStore = None
	_HAS_KVIKIO = False


def kvikio_zarr_available() -> bool:
	"""Return True when KvikIO's Zarr store can be imported."""
	return bool(_HAS_KVIKIO and GDSStore is not None)


def kvikio_zarr_enabled() -> bool:
	"""Return whether the optional KvikIO Zarr path should be attempted."""
	value = os.environ.get("BARRACUDA_KVIKIO_ZARR", "1").strip().lower()
	return value not in {"0", "false", "no", "off"}


def package_versions() -> dict[str, str]:
	"""Best-effort runtime versions for diagnostic logging."""
	versions: dict[str, str] = {}
	try:
		from importlib.metadata import version
	except Exception:
		return versions
	for package in ("kvikio", "cucim", "cudf", "cuml", "cupy"):
		try:
			versions[package] = version(package)
		except Exception:
			continue
	return versions


def _env_int(name: str, default: int, min_value: int = 1) -> int:
	value = os.environ.get(name)
	if value is None:
		return int(default)
	try:
		parsed = int(value)
	except ValueError as exc:
		raise ValueError(f"{name} must be an integer, got {value!r}") from exc
	return max(int(min_value), int(parsed))


def _normalize_component(component: str | Path | None) -> str:
	if component is None:
		return ""
	return str(component).strip("/")


def _regular_chunk_shape(chunks: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
	if not chunks:
		return ()
	return tuple(max(1, int(axis_chunks[0])) for axis_chunks in chunks)


def _open_gds_group(root_path: str | Path, mode: str) -> zarr.Group:
	if not kvikio_zarr_available():
		raise RuntimeError("kvikio.zarr.GDSStore is not available.")
	store = GDSStore(str(root_path))
	return zarr.open_group(store=store, mode=mode)


def write_dask_array_to_zarr(
	array: np.ndarray | da.Array,
	root_path: str | Path,
	component: str | Path,
	overwrite: bool = True,
	max_workers: int | None = None,
	num_gpus_per_task: float | None = None,
) -> bool:
	"""Compute a Dask array chunk-by-chunk and write it through KvikIO GDSStore.

	Returns True when the KvikIO path was used. Returns False when KvikIO is
	missing or disabled so callers can fall back to ``dask.array.to_zarr``.
	``num_gpus_per_task`` is accepted for API compatibility with the previous
	distributed writer; GPU scheduling is now left to the local RAPIDS/CuPy stack.
	"""
	_ = num_gpus_per_task
	if not kvikio_zarr_enabled() or not kvikio_zarr_available():
		return False

	arr = da.asarray(array)
	chunk_shape = _regular_chunk_shape(arr.chunks)
	if arr.ndim == 0 or not chunk_shape:
		return False

	root = _open_gds_group(root_path, mode="a")
	if max_workers is None:
		max_workers = _env_int("BARRACUDA_KVIKIO_WORKERS", max(1, os.cpu_count() or 1))
	else:
		max_workers = max(1, int(max_workers))

	with dask.config.set(scheduler="threads", num_workers=max_workers):
		da.to_zarr(arr, root.store, component=_normalize_component(component), overwrite=overwrite)
	return True


def copy_zarr_array(
	source_root_path: str | Path,
	source_component: str | Path,
	dest_root_path: str | Path,
	dest_component: str | Path,
	overwrite: bool = True,
	max_workers: int | None = None,
) -> bool:
	"""Copy one local Zarr array to another through KvikIO GDSStore."""
	if not kvikio_zarr_enabled() or not kvikio_zarr_available():
		return False

	source_root = _open_gds_group(source_root_path, mode="r")
	source_component_clean = _normalize_component(source_component)
	source_arr = da.from_zarr(source_root.store, component=source_component_clean)
	dest_root = _open_gds_group(dest_root_path, mode="a")

	if max_workers is None:
		max_workers = _env_int("BARRACUDA_KVIKIO_WORKERS", max(1, os.cpu_count() or 1))
	else:
		max_workers = max(1, int(max_workers))
	with dask.config.set(scheduler="threads", num_workers=max_workers):
		da.to_zarr(source_arr, dest_root.store, component=_normalize_component(dest_component), overwrite=overwrite)
	return True


try:
	from .tracing import instrument_module_functions
except ImportError:  # pragma: no cover - supports direct module execution
	from tracing import instrument_module_functions

instrument_module_functions(globals())
