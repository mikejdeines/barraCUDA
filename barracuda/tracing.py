"""Lightweight function-entry tracing for barraCUDA OOM debugging."""

from __future__ import annotations

import functools
import inspect
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable


_TRACE_ENV = "BARRACUDA_TRACE_FUNCTIONS"
_TRACE_ARGS_ENV = "BARRACUDA_TRACE_ARGS"
_TRACE_RETURNS_ENV = "BARRACUDA_TRACE_RETURNS"


def _trace_enabled() -> bool:
	value = os.environ.get(_TRACE_ENV, "1").strip().lower()
	return value not in {"0", "false", "no", "off"}


def _trace_args_enabled() -> bool:
	value = os.environ.get(_TRACE_ARGS_ENV, "1").strip().lower()
	return value not in {"0", "false", "no", "off"}


def _trace_returns_enabled() -> bool:
	value = os.environ.get(_TRACE_RETURNS_ENV, "1").strip().lower()
	return value not in {"0", "false", "no", "off"}


def _rss_mb() -> float | None:
	status_path = Path("/proc/self/status")
	try:
		for line in status_path.read_text().splitlines():
			if line.startswith("VmRSS:"):
				parts = line.split()
				if len(parts) >= 2:
					return float(parts[1]) / 1024.0
	except Exception:
		return None
	return None


def _gpu_memory_summary() -> str | None:
	cp = sys.modules.get("cupy")
	if cp is None:
		return None
	try:
		free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
	except Exception:
		return None
	total_mb = float(total_bytes) / (1024.0 * 1024.0)
	free_mb = float(free_bytes) / (1024.0 * 1024.0)
	used_mb = max(0.0, total_mb - free_mb)
	return f"gpu_used={used_mb:.1f}MB gpu_free={free_mb:.1f}MB"


def _memory_summary() -> str:
	parts: list[str] = []
	rss = _rss_mb()
	if rss is not None:
		parts.append(f"rss={rss:.1f}MB")
	gpu = _gpu_memory_summary()
	if gpu is not None:
		parts.append(gpu)
	return " ".join(parts) if parts else "mem=unavailable"


def _summarize_value(value: Any) -> str:
	try:
		if isinstance(value, (str, Path)):
			text = str(value)
			if len(text) > 120:
				text = text[:117] + "..."
			return repr(text)
		if isinstance(value, (int, float, bool, type(None))):
			return repr(value)
		shape = getattr(value, "shape", None)
		dtype = getattr(value, "dtype", None)
		chunks = getattr(value, "chunks", None)
		if shape is not None:
			shape_text = tuple(int(v) for v in shape)
			dtype_text = f", dtype={dtype}" if dtype is not None else ""
			chunks_text = ", chunks=set" if chunks is not None else ""
			return f"{type(value).__name__}(shape={shape_text}{dtype_text}{chunks_text})"
		if isinstance(value, dict):
			keys = list(value.keys())
			return f"dict(len={len(value)}, keys={keys[:5]!r})"
		if isinstance(value, (list, tuple, set, frozenset)):
			return f"{type(value).__name__}(len={len(value)})"
	except Exception:
		pass
	return type(value).__name__


def _summarize_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
	if not _trace_args_enabled():
		return ""
	parts: list[str] = []
	for idx, value in enumerate(args[:6]):
		parts.append(f"arg{idx}={_summarize_value(value)}")
	if len(args) > 6:
		parts.append(f"... {len(args) - 6} more positional")
	for key, value in list(kwargs.items())[:8]:
		parts.append(f"{key}={_summarize_value(value)}")
	if len(kwargs) > 8:
		parts.append(f"... {len(kwargs) - 8} more kwargs")
	return " " + ", ".join(parts) if parts else ""


def trace_function(func: Callable[..., Any]) -> Callable[..., Any]:
	"""Wrap a function with flushed print statements for OOM localization."""
	if getattr(func, "__barracuda_traced__", False):
		return func

	@functools.wraps(func)
	def wrapped(*args: Any, **kwargs: Any) -> Any:
		if not _trace_enabled():
			return func(*args, **kwargs)

		call_id = f"{os.getpid()}:{threading.get_ident()}:{time.perf_counter_ns()}"
		name = f"{func.__module__}.{func.__qualname__}"
		start = time.perf_counter()
		print(
			f"[barracuda-trace] ENTER {name} call={call_id} {_memory_summary()}{_summarize_call(args, kwargs)}",
			flush=True,
		)
		try:
			result = func(*args, **kwargs)
		except BaseException as exc:
			elapsed = time.perf_counter() - start
			print(
				f"[barracuda-trace] ERROR {name} call={call_id} elapsed={elapsed:.3f}s "
				f"{_memory_summary()} {type(exc).__name__}: {exc}",
				flush=True,
			)
			raise
		elapsed = time.perf_counter() - start
		return_text = ""
		if _trace_returns_enabled():
			return_text = f" return={_summarize_value(result)}"
		print(
			f"[barracuda-trace] EXIT {name} call={call_id} elapsed={elapsed:.3f}s "
			f"{_memory_summary()}{return_text}",
			flush=True,
		)
		return result

	wrapped.__barracuda_traced__ = True  # type: ignore[attr-defined]
	return wrapped


def instrument_module_functions(module_globals: dict[str, Any]) -> None:
	"""Wrap every function defined in the caller's module."""
	module_name = str(module_globals.get("__name__", ""))
	for name, value in list(module_globals.items()):
		if not inspect.isfunction(value):
			continue
		if getattr(value, "__module__", None) != module_name:
			continue
		if name.startswith("__"):
			continue
		module_globals[name] = trace_function(value)
