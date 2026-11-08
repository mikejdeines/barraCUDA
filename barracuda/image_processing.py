"""Image stack processing for MERFISH pre-decoding workflows."""

from __future__ import annotations

import re
import inspect
import time
import os
import threading
from pathlib import Path
from typing import Any, Callable

import dask
import dask.array as da
import numpy as np
import scipy.ndimage as ndi
import zarr
from ome_zarr.io import parse_url

try:
	from .array_utils import (
		cast_numpy_block_for_storage,
		storage_dtype_from_env,
	)
	from .kvikio_zarr import copy_zarr_array, package_versions, write_dask_array_to_zarr
except ImportError:  # pragma: no cover - supports direct module execution
	from array_utils import (
		cast_numpy_block_for_storage,
		storage_dtype_from_env,
	)
	from kvikio_zarr import copy_zarr_array, package_versions, write_dask_array_to_zarr

try:
	from kvikio.zarr import GDSStore
	_HAS_KVIKIO = True
except Exception:
	GDSStore = None
	_HAS_KVIKIO = False

try:
	import cupy as cp
	from cucim.skimage.filters import gaussian as _gaussian_gpu
	from cucim.skimage.registration import phase_cross_correlation as _phase_cross_correlation_gpu
	from cucim.skimage.restoration import richardson_lucy as _richardson_lucy_gpu
	from cupyx.scipy.signal import fftconvolve as _fftconvolve_gpu
	from cupyx.scipy.ndimage import median_filter as _median_filter_gpu
	from cupyx.scipy.ndimage import shift as _shift_gpu

	_HAS_CUCIM = True
except Exception:
	cp = None
	_fftconvolve_gpu = None
	_median_filter_gpu = None
	_HAS_CUCIM = False

_AXIS_TYPES: dict[str, str] = {
	"t": "time",
	"c": "channel",
	"z": "space",
	"y": "space",
	"x": "space",
	"m": "space",
}

# Cache decoded PSF stores across FOVs to avoid repeated Zarr I/O.
_PSF_BANK_CACHE: dict[str, tuple[da.Array, dict[int, da.Array], dict[int, da.Array]]] = {}

# Flush pooled GPU allocations periodically to balance throughput and VRAM pressure.
_GPU_FREE_EVERY_N_CALLS = 8
_GPU_FREE_COUNTER = 0


def _env_int(name: str, default: int) -> int:
	value = os.environ.get(name)
	if value is None:
		return int(default)
	try:
		return int(value)
	except ValueError as exc:
		raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_int_with_fallback(name: str, fallback_name: str, default: int) -> int:
	if os.environ.get(name) is not None:
		return _env_int(name, default)
	return _env_int(fallback_name, default)


# Conservative default Z/Y/X chunk for GPU Richardson-Lucy on dask arrays.
_RL_GPU_CHUNK_SHAPE: tuple[int, int, int] = (
	max(1, _env_int("BARRACUDA_RL_CHUNK_Z", 18)),
	max(1, _env_int("BARRACUDA_RL_CHUNK_Y", 768)),
	max(1, _env_int("BARRACUDA_RL_CHUNK_X", 768)),
)
_GPU_DEVICE_LOCK = threading.Lock()
_GPU_DEVICE_ROUND_ROBIN = 0

# Conservative default Z/Y/X chunk for GPU spatial filters.  These filters are
# separable but allocate temporary device arrays, so host-RAM-sized Dask chunks
# can still OOM VRAM.
_GPU_FILTER_CHUNK_SHAPE: tuple[int, int, int] = (
	max(1, _env_int_with_fallback("BARRACUDA_FILTER_CHUNK_Z", "BARRACUDA_CHUNK_Z", 18)),
	max(1, _env_int_with_fallback("BARRACUDA_FILTER_CHUNK_Y", "BARRACUDA_CHUNK_Y", 768)),
	max(1, _env_int_with_fallback("BARRACUDA_FILTER_CHUNK_X", "BARRACUDA_CHUNK_X", 768)),
)

# Stitching chunks are intentionally smaller because every output chunk may
# touch several shifted tiles plus their support masks.
_STITCH_CHUNK_SHAPE: tuple[int, int, int] = (
	max(1, _env_int("BARRACUDA_STITCH_CHUNK_Z", 8)),
	max(1, _env_int("BARRACUDA_STITCH_CHUNK_Y", 512)),
	max(1, _env_int("BARRACUDA_STITCH_CHUNK_X", 512)),
)

_FALLBACK_PROCESS_CHUNK_SHAPE: tuple[int, int, int] = (18, 2048, 2048)
_STORAGE_DTYPE = storage_dtype_from_env()

_BACKGROUND_SAMPLE_STRIDE_YX = max(1, int(os.environ.get("BARRACUDA_BACKGROUND_SAMPLE_STRIDE_YX", "4")))

# Maximum allowed shift corrections per spatial axis.
# Don't let anything shift in z.
_MAX_SHIFT_Z_PX: float = 0
_MAX_SHIFT_YX_PX: float = 20.0


def profile_add(profile: dict[str, float] | None, key: str, value: float) -> None:
	if profile is None:
		return
	profile[key] = float(profile.get(key, 0.0) + float(value))


def print_profile_summary(profile: dict[str, float], prefix: str = "[image_processing]") -> None:
	keys = [
		"total_s",
		"read_s",
		"stitch_graph_s",
		"stitch_materialize_s",
		"process_compute_s",
		"materialize_s",
		"rl_compute_s",
		"gpu_transfer_in_s",
		"gpu_transfer_out_s",
		"write_s",
	]
	metrics = ", ".join(
		f"{key}={profile[key]:.2f}s"
		for key in keys
		if float(profile.get(key, 0.0)) > 0.0
	)
	print(f"{prefix} timing summary: {metrics}", flush=True)


def open_zarr_group_with_optional_kvikio(path: str | Path, mode: str) -> zarr.Group:
	"""Open a Zarr group using KvikIO GDSStore when available.

	For local filesystem stores, KvikIO can use GPUDirect Storage-enabled I/O
	paths and avoid host bounce buffers for device-bound chunk transfers.
	Falls back to the previous parse_url/open_group path when unavailable.
	"""
	path_str = str(path)
	# Prefer GDSStore for both read and write so chunk I/O can use KvikIO when available.
	if _HAS_KVIKIO and GDSStore is not None:
		try:
			store = GDSStore(path_str)
			return zarr.open_group(store=store, mode=mode)
		except Exception:
			pass

	loc = parse_url(path_str, mode=mode)
	if loc is None:
		raise ValueError(f"Could not open OME-Zarr path: {path_str}")
	return zarr.open_group(store=loc.store, mode=mode)


def available_gpu_count() -> int:
	if not _HAS_CUCIM or cp is None:
		return 0
	try:
		return max(0, int(cp.cuda.runtime.getDeviceCount()))
	except Exception:
		return 0


def next_gpu_device_id(gpu_count: int) -> int:
	global _GPU_DEVICE_ROUND_ROBIN
	if gpu_count <= 1:
		return 0
	with _GPU_DEVICE_LOCK:
		device_id = int(_GPU_DEVICE_ROUND_ROBIN % gpu_count)
		_GPU_DEVICE_ROUND_ROBIN += 1
	return device_id


def clear_fft_plan_cache() -> None:
	if not _HAS_CUCIM:
		return
	try:
		cp.fft.config.get_plan_cache().clear()
	except Exception:
		pass


def free_gpu_memory() -> None:
	if not _HAS_CUCIM:
		return
	try:
		cp.get_default_memory_pool().free_all_blocks()
		cp.get_default_pinned_memory_pool().free_all_blocks()
	except Exception:
		pass


def maybe_free_gpu_memory(force: bool = False) -> None:
	global _GPU_FREE_COUNTER
	if not _HAS_CUCIM:
		return
	if force or _GPU_FREE_EVERY_N_CALLS <= 0:
		free_gpu_memory()
		_GPU_FREE_COUNTER = 0
		return
	_GPU_FREE_COUNTER += 1
	if _GPU_FREE_COUNTER >= _GPU_FREE_EVERY_N_CALLS:
		free_gpu_memory()
		_GPU_FREE_COUNTER = 0


def richardson_lucy_gpu_raw_init(image_gpu: Any, psf_gpu: Any, num_iter: int, clip: bool) -> Any:
	kwargs: dict[str, Any] = {"num_iter": num_iter, "clip": clip}
	image_gpu = cp.clip(cp.asarray(image_gpu, dtype=cp.float32), a_min=0.0, a_max=None)
	try:
		signature = inspect.signature(_richardson_lucy_gpu)
	except Exception:
		signature = None

	if signature is not None:
		for param_name in ("init_image", "init", "x0", "initial", "estimate"):
			if param_name in signature.parameters:
				kwargs[param_name] = image_gpu
				return _richardson_lucy_gpu(image_gpu, psf_gpu, **kwargs)

	if _fftconvolve_gpu is None:
		return _richardson_lucy_gpu(image_gpu, psf_gpu, **kwargs)

	eps = cp.float32(1e-6)
	image = image_gpu
	psf = cp.asarray(psf_gpu, dtype=cp.float32)
	psf_sum = cp.sum(psf)
	if psf_sum <= eps:
		raise ValueError("PSF sum is zero; cannot run Richardson-Lucy deconvolution.")
	psf = psf / psf_sum
	psf_mirror = psf[(slice(None, None, -1),) * psf.ndim]

	estimate = cp.clip(image, a_min=0.0, a_max=None)
	for _ in range(int(num_iter)):
		conv = _fftconvolve_gpu(estimate, psf, mode="same")
		relative_blur = image / (conv + eps)
		estimate *= _fftconvolve_gpu(relative_blur, psf_mirror, mode="same")
		if clip:
			estimate = cp.clip(estimate, 0.0, 1.0)

	return estimate


def bounded_spatial_chunk_shape(
	shape_zyx: tuple[int, int, int],
	base_chunk_zyx: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
	"""Return a Z/Y/X chunk shape capped by the array bounds."""
	base = _FALLBACK_PROCESS_CHUNK_SHAPE if base_chunk_zyx is None else base_chunk_zyx
	return (
		max(1, min(int(shape_zyx[0]), int(base[0]))),
		max(1, min(int(shape_zyx[1]), int(base[1]))),
		max(1, min(int(shape_zyx[2]), int(base[2]))),
	)


def batch_size_xyz_to_zyx(batch_size_xyz: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
	if batch_size_xyz is None:
		return None
	values = tuple(int(v) for v in batch_size_xyz)
	if len(values) != 3:
		raise ValueError(f"batch_size_xyz must be a length-3 (x, y, z) tuple, got {batch_size_xyz!r}")
	if any(v <= 0 for v in values):
		raise ValueError(f"batch_size_xyz values must be positive, got {batch_size_xyz!r}")
	x_size, y_size, z_size = values
	return (int(z_size), int(y_size), int(x_size))


def to_dask_tczyx(
	image: np.ndarray | da.Array,
	chunk_shape_zyx: tuple[int, int, int] | None = None,
) -> da.Array:
	arr = da.asarray(image)
	if arr.ndim != 5:
		raise ValueError(f"Expected TCZYX input with ndim=5, got shape {arr.shape}")
	chunk_z, chunk_y, chunk_x = bounded_spatial_chunk_shape(
		(int(arr.shape[2]), int(arr.shape[3]), int(arr.shape[4])),
		base_chunk_zyx=chunk_shape_zyx,
	)
	return arr.rechunk((1, 1, chunk_z, chunk_y, chunk_x))


def rechunk_tczyx_for_write(
	array: np.ndarray | da.Array,
	write_chunk_shape_zyx: tuple[int, int, int] | None,
) -> da.Array:
	"""Return TCZYX data rechunked for coarse, low-file-count Zarr writes."""
	arr = da.asarray(array)
	if arr.ndim != 5:
		raise ValueError(f"Expected TCZYX input with ndim=5, got shape {arr.shape}")
	if write_chunk_shape_zyx is None:
		return arr
	chunk_z, chunk_y, chunk_x = bounded_spatial_chunk_shape(
		(int(arr.shape[2]), int(arr.shape[3]), int(arr.shape[4])),
		base_chunk_zyx=write_chunk_shape_zyx,
	)
	return arr.rechunk((1, 1, chunk_z, chunk_y, chunk_x))


def normalize_spatial_param(value: float | tuple[float, float, float], name: str) -> tuple[float, float, float]:
	if isinstance(value, tuple):
		values = tuple(float(v) for v in value)
	else:
		values = (float(value), float(value), float(value))
	if len(values) != 3:
		raise ValueError(f"{name} must be a scalar or a length-3 ZYX tuple, got {value!r}")
	return values


def gaussian_overlap_depth(sigma: float | tuple[float, float, float]) -> tuple[int, int, int]:
	# Four sigma covers essentially all Gaussian support while keeping halo work bounded.
	return tuple(max(0, int(np.ceil(4.0 * s))) for s in normalize_spatial_param(sigma, "sigma"))


def _clip_negative_highpass_values(array: Any) -> Any:
	"""Match MERlin's high-pass convention: Gaussian subtraction cannot go below zero."""
	if cp is not None and isinstance(array, cp.ndarray):
		cp.maximum(array, 0.0, out=array)
		return array
	np.maximum(array, 0.0, out=array)
	return array


def median_overlap_depth(size: int | tuple[int, int, int]) -> tuple[int, int, int]:
	if isinstance(size, tuple):
		sizes = tuple(int(v) for v in size)
	else:
		sizes = (int(size), int(size), int(size))
	if len(sizes) != 3:
		raise ValueError(f"median size must be a scalar or a length-3 ZYX tuple, got {size!r}")
	return tuple(max(0, int(v) // 2) for v in sizes)


def gaussian_backend(array: np.ndarray, sigma: float | tuple[float, float, float]):
	if not _HAS_CUCIM:
		raise RuntimeError(
			"GPU backend is required but cuCIM/CuPy is unavailable. "
			"Install/enable GPU dependencies and run with --nv."
		)
	gpu = result = out = None
	try:
		gpu = cp.asarray(array)
		result = _gaussian_gpu(gpu, sigma=sigma, preserve_range=True)
		out = result.astype(cp.float32, copy=False)
	finally:
		del gpu, result
		maybe_free_gpu_memory()
	if out is None:
		raise RuntimeError("GPU gaussian filter failed to produce an output.")
	return out


def richardson_lucy_backend(
	array: np.ndarray | da.Array,
	psf: np.ndarray | da.Array,
	num_iter: int,
	clip: bool,
	chunk_shape: tuple[int, int, int] | None = None,
	timing_stats: dict[str, float] | None = None,
) -> np.ndarray | da.Array:
	psf_np = da.asarray(psf).compute().astype(np.float32, copy=False)

	if isinstance(array, da.Array):
		arr = da.asarray(array).astype(np.float32)
		chunk_z, chunk_y, chunk_x = bounded_spatial_chunk_shape(
			shape_zyx=(int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])),
			base_chunk_zyx=chunk_shape or _RL_GPU_CHUNK_SHAPE,
		)
		arr = arr.rechunk((chunk_z, chunk_y, chunk_x))

		chunk_sizes = tuple(int(c[0]) for c in arr.chunks)
		depth = (
			max(0, min(int(psf_np.shape[0] // 2), chunk_sizes[0] - 1)),
			max(0, min(int(psf_np.shape[1] // 2), chunk_sizes[1] - 1)),
			max(0, min(int(psf_np.shape[2] // 2), chunk_sizes[2] - 1)),
		)
		rl_workers = max(1, available_gpu_count())

		def decon_block(block: np.ndarray, _psf: np.ndarray = psf_np) -> np.ndarray:
			if not _HAS_CUCIM:
				raise RuntimeError(
					"GPU backend is required but cuCIM/CuPy is unavailable. "
					"Install/enable GPU dependencies and run with --nv."
				)
			gpu = gpu_psf = result = out = None
			device_id = next_gpu_device_id(rl_workers)
			try:
				with cp.cuda.Device(int(device_id)):
					t0 = time.perf_counter()
					gpu = cp.asarray(block)
					gpu_psf = cp.asarray(_psf)
					profile_add(timing_stats, "gpu_transfer_in_s", time.perf_counter() - t0)

					t1 = time.perf_counter()
					result = richardson_lucy_gpu_raw_init(gpu, gpu_psf, num_iter=num_iter, clip=clip)
					profile_add(timing_stats, "rl_compute_s", time.perf_counter() - t1)
					out = result.astype(cp.float32, copy=False)
			finally:
				del gpu, gpu_psf, result
				with cp.cuda.Device(int(device_id)):
					maybe_free_gpu_memory()
			if out is None:
				raise RuntimeError("GPU Richardson-Lucy block failed to produce an output.")
			return out

		if depth == (0, 0, 0):
			kwargs: dict[str, Any] = {"dtype": np.float32}
			if _HAS_CUCIM and cp is not None:
				kwargs["meta"] = cp.empty((0,), dtype=cp.float32)
			with dask.config.set(scheduler="threads", num_workers=rl_workers):
				return da.map_blocks(decon_block, arr, **kwargs)

		kwargs = {
			"depth": depth,
			"boundary": "reflect",
			"trim": True,
			"dtype": np.float32,
		}
		if _HAS_CUCIM and cp is not None:
			kwargs["meta"] = cp.empty((0,), dtype=cp.float32)
		with dask.config.set(scheduler="threads", num_workers=rl_workers):
			return da.map_overlap(decon_block, arr, **kwargs)

	if not _HAS_CUCIM:
		raise RuntimeError(
			"GPU backend is required but cuCIM/CuPy is unavailable. "
			"Install/enable GPU dependencies and run with --nv."
		)
	gpu = gpu_psf = result = out = None
	try:
		t0 = time.perf_counter()
		gpu = cp.asarray(array)
		gpu_psf = cp.asarray(psf_np)
		profile_add(timing_stats, "gpu_transfer_in_s", time.perf_counter() - t0)

		t1 = time.perf_counter()
		result = richardson_lucy_gpu_raw_init(gpu, gpu_psf, num_iter=num_iter, clip=clip)
		profile_add(timing_stats, "rl_compute_s", time.perf_counter() - t1)
		out = result.astype(cp.float32, copy=False)
	finally:
		del gpu, gpu_psf, result
		maybe_free_gpu_memory()
	if out is None:
		raise RuntimeError("GPU Richardson-Lucy failed to produce an output.")
	return out


def shift_backend(array: np.ndarray, shift_zyx: np.ndarray) -> np.ndarray:
	if not _HAS_CUCIM:
		raise RuntimeError(
			"GPU backend is required but cuCIM/CuPy is unavailable. "
			"Install/enable GPU dependencies and run with --nv."
		)
	gpu = gpu_shift = shifted = out = None
	try:
		gpu = cp.asarray(array)
		shift_np = np.asarray(shift_zyx, dtype=np.float32)
		if gpu.ndim == 4:
			# CZYX volume: keep channel axis fixed and shift spatial axes only.
			shift_np = np.concatenate(([0.0], shift_np))
		gpu_shift = cp.asarray(shift_np, dtype=cp.float32)
		shifted = _shift_gpu(gpu, shift=gpu_shift, order=1, mode="constant", cval=0.0, prefilter=False)
		out = shifted.astype(cp.float32, copy=False)
	finally:
		del gpu, gpu_shift, shifted
		maybe_free_gpu_memory()
	if out is None:
		raise RuntimeError("GPU shift failed to produce an output.")
	return out


def median_backend(array: np.ndarray, size: int | tuple[int, int, int]) -> np.ndarray:
	if _HAS_CUCIM:
		gpu = result = out = None
		try:
			gpu = cp.asarray(array)
			result = _median_filter_gpu(gpu, size=size)
			out = result.astype(cp.float32, copy=False)
		finally:
			del gpu, result
			maybe_free_gpu_memory()
		if out is None:
			raise RuntimeError("GPU median filter failed to produce an output.")
		return out

	return ndi.median_filter(array, size=size).astype(np.float32, copy=False)


def phase_correlation_backend(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
	if not _HAS_CUCIM:
		raise RuntimeError(
			"GPU backend is required but cuCIM/CuPy is unavailable. "
			"Install/enable GPU dependencies and run with --nv."
		)
	ref_gpu = mov_gpu = shift_zyx = shift_np = None
	try:
		ref_gpu = cp.asarray(reference)
		mov_gpu = cp.asarray(moving)
		shift_zyx, _, _ = _phase_cross_correlation_gpu(ref_gpu, mov_gpu)
		shift_np = cp.asnumpy(shift_zyx).astype(np.float32, copy=False)
	finally:
		del ref_gpu, mov_gpu, shift_zyx
		maybe_free_gpu_memory()
	if shift_np is None:
		raise RuntimeError("GPU phase correlation failed to produce an output.")
	return shift_np


def _get_primary_dataset(group: zarr.Group, require_multiscales: bool = True) -> zarr.Array:
	"""Return the highest-resolution dataset from an image or bare PSF group."""
	multiscales = get_multiscales(group)
	datasets = multiscales[0].get("datasets", []) if multiscales else []
	if require_multiscales and not multiscales:
		raise ValueError(f"Group has no 'multiscales' metadata: {group.name}")
	if require_multiscales and not datasets:
		raise ValueError(f"No datasets listed in multiscales for group: {group.name}")

	dataset_path = datasets[0].get("path", "0") if datasets and isinstance(datasets[0], dict) else "0"
	if dataset_path not in group:
		raise ValueError(f"Dataset path '{dataset_path}' not found in group: {group.name}")

	dataset = group[dataset_path]
	if not isinstance(dataset, zarr.Array):
		raise ValueError(f"Expected zarr.Array at '{group.name}/{dataset_path}'.")
	return dataset


def read_psf_from_group(root_group: zarr.Group) -> da.Array:
	data = da.from_zarr(_get_primary_dataset(root_group, require_multiscales=False))
	data = da.squeeze(data).astype(np.float32)
	if data.ndim != 3:
		raise ValueError(f"Expected PSF with ZYX dimensions, got shape {data.shape}.")
	return data.rechunk(data.shape)


def load_psf_bank(psf_path: str | Path) -> tuple[da.Array, dict[int, da.Array], dict[int, da.Array]]:
	psf_dir = Path(psf_path)
	if not psf_dir.is_dir():
		raise ValueError(f"psf_path must be a directory containing PSF .zarr files: {psf_dir}")

	zarr_files = sorted([p for p in psf_dir.iterdir() if p.suffix == ".zarr"])
	if not zarr_files:
		raise ValueError(f"No .zarr PSF files found in directory: {psf_dir}")

	psf_by_wavelength_nm: dict[int, da.Array] = {}
	psf_by_channel_index: dict[int, da.Array] = {}
	default_psf: da.Array | None = None

	for psf_file in zarr_files:
		try:
			root = open_zarr_group_with_optional_kvikio(psf_file, mode="r")
		except Exception:
			continue

		root_psf: da.Array | None = None
		try:
			root_psf = read_psf_from_group(root)
		except Exception:
			root_psf = None

		if root_psf is not None:
			if default_psf is None or "default" in psf_file.stem.lower():
				default_psf = root_psf

			wavelength_value = (
				root.attrs.get("wavelength_nm")
				or root.attrs.get("emission_wavelength_nm")
				or root.attrs.get("excitation_wavelength_nm")
			)
			if wavelength_value is None:
				match_nm = re.search(r"(\d+(?:\.\d+)?)\s*nm", psf_file.stem.lower())
				if match_nm:
					wavelength_value = match_nm.group(1)
			if wavelength_value is not None:
				try:
					psf_by_wavelength_nm[int(round(float(wavelength_value)))] = root_psf
				except Exception:
					pass

			channel_value = root.attrs.get("channel_index") or root.attrs.get("channel")
			if channel_value is None:
				match_ch = re.search(r"(?:^|[_-])(?:ch|channel)(\d+)(?:$|[_-])", psf_file.stem.lower())
				if match_ch:
					channel_value = match_ch.group(1)
			if channel_value is not None:
				try:
					psf_by_channel_index[int(channel_value)] = root_psf
				except Exception:
					pass

		psf_wavelength_map = root.attrs.get("psf_by_wavelength_nm", {})
		if isinstance(psf_wavelength_map, dict):
			for wavelength_key, group_name in psf_wavelength_map.items():
				try:
					group = root[group_name]
					psf_by_wavelength_nm[int(round(float(wavelength_key)))] = read_psf_from_group(group)
				except Exception:
					continue

		psf_channel_map = root.attrs.get("psf_by_channel", {})
		if isinstance(psf_channel_map, dict):
			for channel_key, group_name in psf_channel_map.items():
				try:
					group = root[group_name]
					psf_by_channel_index[int(channel_key)] = read_psf_from_group(group)
				except Exception:
					continue

	if default_psf is None:
		raise ValueError(f"Could not find a readable default PSF image node in directory: {psf_dir}")

	return default_psf, psf_by_wavelength_nm, psf_by_channel_index


def get_psf_bank_cached(psf_path: str | Path) -> tuple[da.Array, dict[int, da.Array], dict[int, da.Array]]:
	cache_key = str(Path(psf_path).expanduser().resolve())
	if cache_key not in _PSF_BANK_CACHE:
		_PSF_BANK_CACHE[cache_key] = load_psf_bank(psf_path)
	return _PSF_BANK_CACHE[cache_key]


def resolve_psfs_for_channels(
	psf_path: str | Path,
	channel_metadata: list[dict[str, Any]] | None,
	channel_count: int,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
	default_psf, psf_by_wavelength_nm, psf_by_channel_idx = get_psf_bank_cached(psf_path)
	channel_wavelengths_nm = extract_channel_wavelengths_nm(channel_metadata, channel_count)

	resolved_psf_by_channel: dict[int, da.Array] = {}
	for channel_idx in range(channel_count):
		if channel_idx in psf_by_channel_idx:
			resolved_psf_by_channel[channel_idx] = psf_by_channel_idx[channel_idx]
			continue
		wavelength_nm = channel_wavelengths_nm[channel_idx]
		if wavelength_nm is not None:
			selected = select_psf_for_wavelength(wavelength_nm, psf_by_wavelength_nm)
			if selected is not None:
				resolved_psf_by_channel[channel_idx] = selected

	default_psf_np = da.asarray(default_psf).compute().astype(np.float32, copy=False)
	psf_by_channel_np: dict[int, np.ndarray] = {
		int(k): da.asarray(v).compute().astype(np.float32, copy=False)
		for k, v in resolved_psf_by_channel.items()
	}
	return default_psf_np, psf_by_channel_np


def extract_channel_wavelengths_nm(
	channel_metadata: list[dict[str, Any]] | None,
	channel_count: int,
) -> list[int | None]:
	wavelengths: list[int | None] = [None] * channel_count
	if not channel_metadata:
		return wavelengths

	for idx, meta in enumerate(channel_metadata):
		if idx >= channel_count:
			break
		if not isinstance(meta, dict):
			continue
		value = (
			meta.get("wavelength_nm")
			or meta.get("emission_wavelength_nm")
			or meta.get("excitation_wavelength_nm")
		)
		if value is None:
			continue
		try:
			wavelengths[idx] = int(round(float(value)))
		except Exception:
			continue

	return wavelengths


def select_psf_for_wavelength(
	image_wavelength_nm: int,
	psf_by_wavelength_nm: dict[int, da.Array],
) -> da.Array | None:
	if image_wavelength_nm in psf_by_wavelength_nm:
		return psf_by_wavelength_nm[image_wavelength_nm]
	candidates = sorted([w for w in psf_by_wavelength_nm.keys() if w >= image_wavelength_nm])
	if not candidates:
		return None
	return psf_by_wavelength_nm[candidates[0]]


def align_beads(fov_image: np.ndarray | da.Array, bead_channel: int | None) -> da.Array:
	image = to_dask_tczyx(fov_image)
	t_dim, c_dim, _, _, _ = image.shape
	if t_dim <= 1 or bead_channel is None:
		return image.astype(np.float32)
	if not (0 <= bead_channel < c_dim):
		raise ValueError(f"bead_channel={bead_channel} out of bounds for C={c_dim}")

	bead_stack = image[:, bead_channel].compute()
	reference = bead_stack[0]
	aligned_t: list[da.Array] = []

	for t_idx in range(t_dim):
		moving_beads = bead_stack[t_idx]
		shift_zyx = phase_correlation_backend(reference, moving_beads)
		moving_all_channels = image[t_idx]
		kwargs: dict[str, Any] = {"dtype": np.float32}
		if _HAS_CUCIM and cp is not None:
			kwargs["meta"] = cp.empty((0,), dtype=cp.float32)
		with dask.config.set(scheduler="single-threaded"):
			shifted = da.map_blocks(
				lambda block: shift_backend(block, shift_zyx), moving_all_channels, **kwargs,
			)
		aligned_t.append(shifted)

	return da.stack(aligned_t, axis=0).astype(np.float32)


def _volume_map_kwargs() -> dict[str, Any]:
	kwargs: dict[str, Any] = {"dtype": np.float32}
	if _HAS_CUCIM and cp is not None:
		kwargs["meta"] = cp.empty((0,), dtype=cp.float32)
	return kwargs


def _cast_block_to_backend_float32(block: Any) -> Any:
	if _HAS_CUCIM and cp is not None:
		return cp.asarray(block, dtype=cp.float32)
	return np.asarray(block, dtype=np.float32)


def _identity_volume(vol: da.Array) -> da.Array:
	with dask.config.set(scheduler="single-threaded"):
		return da.map_blocks(
			lambda block: _cast_block_to_backend_float32(block),
			vol,
			**_volume_map_kwargs(),
		)


def _apply_volume_func(block: Any, func: Callable[[Any], Any]) -> Any:
	backend_block = _cast_block_to_backend_float32(block)
	return func(backend_block).astype(np.float32, copy=False)


def _gpu_filter_chunk_shape_for_image(
	image: np.ndarray | da.Array,
) -> tuple[int, int, int]:
	arr = da.asarray(image)
	if arr.ndim != 5:
		raise ValueError(f"Expected TCZYX input with ndim=5, got shape {arr.shape}")
	return bounded_spatial_chunk_shape(
		shape_zyx=(int(arr.shape[2]), int(arr.shape[3]), int(arr.shape[4])),
		base_chunk_zyx=_GPU_FILTER_CHUNK_SHAPE,
	)


def apply_per_volume(
	image: np.ndarray | da.Array,
	func: Callable[[Any], Any],
	skip_tc: tuple[int, int] | None = None,
	overlap_depth: tuple[int, int, int] | None = None,
	boundary: str | float = "reflect",
	chunk_shape_zyx: tuple[int, int, int] | None = None,
) -> da.Array:
	arr = to_dask_tczyx(image, chunk_shape_zyx=chunk_shape_zyx)
	t_dim, c_dim, _, _, _ = arr.shape
	use_overlap = overlap_depth is not None and any(int(d) > 0 for d in overlap_depth)
	out_t: list[da.Array] = []
	for t_idx in range(t_dim):
		out_c: list[da.Array] = []
		for c_idx in range(c_dim):
			vol = arr[t_idx, c_idx]
			if skip_tc is not None and (int(t_idx), int(c_idx)) == skip_tc:
				out_c.append(_identity_volume(vol))
				continue
			with dask.config.set(scheduler="single-threaded"):
				if use_overlap:
					mapped = da.map_overlap(
						lambda block, _func=func: _apply_volume_func(block, _func),
						vol,
						depth=overlap_depth,
						boundary=boundary,
						trim=True,
						**_volume_map_kwargs(),
					)
				else:
					mapped = da.map_blocks(
						lambda block, _func=func: _apply_volume_func(block, _func),
						vol,
						**_volume_map_kwargs(),
					)
			out_c.append(mapped)
		out_t.append(da.stack(out_c, axis=0))
	return da.stack(out_t, axis=0).astype(np.float32)


def highpass_filter(
	image: np.ndarray | da.Array,
	sigma: float | tuple[float, float, float],
	skip_tc: tuple[int, int] | None = None,
	chunk_shape_zyx: tuple[int, int, int] | None = None,
) -> da.Array:
	def highpass(block: np.ndarray) -> np.ndarray:
		highpassed = block - gaussian_backend(block, sigma=sigma)
		return _clip_negative_highpass_values(highpassed)

	overlap_depth = gaussian_overlap_depth(sigma)
	return apply_per_volume(
		image,
		highpass,
		skip_tc=skip_tc,
		overlap_depth=overlap_depth,
		boundary="reflect",
		chunk_shape_zyx=chunk_shape_zyx or _gpu_filter_chunk_shape_for_image(image),
	)


def deconvolve_psf(
	image: np.ndarray | da.Array,
	psf_path: str | Path,
	channel_metadata: list[dict[str, Any]] | None = None,
	num_iter: int = 10,
	clip: bool = False,
	precomputed_psfs: tuple[np.ndarray, dict[int, np.ndarray]] | None = None,
	timing_stats: dict[str, float] | None = None,
	skip_tc: tuple[int, int] | None = None,
	chunk_shape_zyx: tuple[int, int, int] | None = None,
) -> da.Array:
	arr = to_dask_tczyx(image, chunk_shape_zyx=chunk_shape_zyx or _gpu_filter_chunk_shape_for_image(image))
	t_dim, c_dim, _, _, _ = arr.shape

	if precomputed_psfs is None:
		default_psf_np, psf_by_channel_np = resolve_psfs_for_channels(
			psf_path=psf_path,
			channel_metadata=channel_metadata,
			channel_count=int(c_dim),
		)
	else:
		default_psf_np, psf_by_channel_np = precomputed_psfs

	clear_fft_plan_cache()
	try:
		out_t: list[da.Array] = []
		for t_idx in range(t_dim):
			out_c: list[da.Array] = []
			for c_idx in range(c_dim):
				vol = arr[t_idx, c_idx]
				if skip_tc is not None and (int(t_idx), int(c_idx)) == skip_tc:
					out_c.append(_identity_volume(vol))
					continue
				psf_c = psf_by_channel_np.get(c_idx, default_psf_np)
				mapped = richardson_lucy_backend(
					vol,
					psf_c,
					num_iter=num_iter,
					clip=clip,
					timing_stats=timing_stats,
				)
				out_c.append(mapped)
			out_t.append(da.stack(out_c, axis=0))
		return da.stack(out_t, axis=0).astype(np.float32)
	finally:
		clear_fft_plan_cache()
		maybe_free_gpu_memory(force=True)


def lowpass_filter(
	image: np.ndarray | da.Array,
	sigma: float | tuple[float, float, float],
	skip_tc: tuple[int, int] | None = None,
	chunk_shape_zyx: tuple[int, int, int] | None = None,
) -> da.Array:
	def lowpass(block: np.ndarray) -> np.ndarray:
		return gaussian_backend(block, sigma=sigma)
	overlap_depth = gaussian_overlap_depth(sigma)
	return apply_per_volume(
		image,
		lowpass,
		skip_tc=skip_tc,
		overlap_depth=overlap_depth,
		boundary="reflect",
		chunk_shape_zyx=chunk_shape_zyx or _gpu_filter_chunk_shape_for_image(image),
	)


def median_filter_image(
	image: np.ndarray | da.Array,
	size: int | tuple[int, int, int] = 3,
	chunk_shape_zyx: tuple[int, int, int] | None = None,
) -> da.Array:
	"""Apply a 3D median filter to each (T, C) volume in a TCZYX stack."""
	def median(block: np.ndarray) -> np.ndarray:
		return median_backend(block, size=size)

	overlap_depth = median_overlap_depth(size)
	return apply_per_volume(
		image,
		median,
		overlap_depth=overlap_depth,
		boundary="reflect",
		chunk_shape_zyx=chunk_shape_zyx or _gpu_filter_chunk_shape_for_image(image),
	)


def process_stack(
	fov_image: np.ndarray | da.Array,
	bead_channel: int | None,
	psf_path: str | Path,
	output_path: str | Path,
	channel_metadata: list[dict[str, Any]] | None = None,
	highpass_sigma: float | tuple[float, float, float] = 3.0,
	lowpass_sigma: float | tuple[float, float, float] = 1.0,
	deconvolution_iterations: int = 10,
	precomputed_psfs: tuple[np.ndarray, dict[int, np.ndarray]] | None = None,
	timing_stats: dict[str, float] | None = None,
	dapi_channel: int | None = None,
	dapi_timepoint: int | None = None,
	is_confocal: bool = False,
	chunk_shape_zyx: tuple[int, int, int] | None = None,
) -> da.Array:
	input_image = to_dask_tczyx(fov_image, chunk_shape_zyx=chunk_shape_zyx)
	t_dim, c_dim, _, _, _ = input_image.shape

	if (dapi_channel is None) != (dapi_timepoint is None):
		raise ValueError("dapi_channel and dapi_timepoint must be provided together.")

	skip_tc: tuple[int, int] | None = None
	if dapi_channel is not None and dapi_timepoint is not None:
		if dapi_channel < 0 or dapi_timepoint < 0:
			raise ValueError("dapi_channel and dapi_timepoint must be >= 0.")
		if dapi_channel >= int(c_dim):
			raise ValueError(f"dapi_channel={dapi_channel} out of bounds for C={int(c_dim)}")
		if dapi_timepoint >= int(t_dim):
			raise ValueError(f"dapi_timepoint={dapi_timepoint} out of bounds for T={int(t_dim)}")
		skip_tc = (int(dapi_timepoint), int(dapi_channel))

	if input_image.shape[0] <= 1 or bead_channel is None:
		aligned = input_image.astype(np.float32)
	else:
		aligned = align_beads(fov_image=input_image, bead_channel=bead_channel)
	highpassed = highpass_filter(aligned, sigma=highpass_sigma, skip_tc=skip_tc, chunk_shape_zyx=chunk_shape_zyx)
	if is_confocal:
		highpassed = median_filter_image(highpassed, size=3, chunk_shape_zyx=chunk_shape_zyx)
	deconvolved = deconvolve_psf(
		highpassed,
		psf_path=psf_path,
		channel_metadata=channel_metadata,
		num_iter=deconvolution_iterations,
		precomputed_psfs=precomputed_psfs,
		timing_stats=timing_stats,
		skip_tc=skip_tc,
		chunk_shape_zyx=chunk_shape_zyx,
	)
	return lowpass_filter(deconvolved, sigma=lowpass_sigma, skip_tc=skip_tc, chunk_shape_zyx=chunk_shape_zyx)


def extract_channel_metadata_from_group(group: zarr.Group) -> list[dict[str, Any]] | None:
	omero_meta = group.attrs.get("omero")
	if isinstance(omero_meta, dict):
		channels = omero_meta.get("channels")
		if isinstance(channels, list):
			metadata = [ch for ch in channels if isinstance(ch, dict)]
			if metadata:
				return metadata

	channel_meta = group.attrs.get("channel_metadata")
	if isinstance(channel_meta, list):
		metadata = [x for x in channel_meta if isinstance(x, dict)]
		if metadata:
			return metadata

	return None


def extract_global_transform_from_group(
	group: zarr.Group,
	tile_meta: dict[str, Any] | None,
) -> dict[str, Any]:
	meta = dict(tile_meta) if isinstance(tile_meta, dict) else {}

	for key in (
		"stage_x", "stage_y", "stage_z",
		"physical_pixel_size_x", "physical_pixel_size_y", "physical_pixel_size_z",
	):
		if key in meta and meta[key] is not None:
			continue
		value = group.attrs.get(key)
		if value is not None:
			meta[key] = value

	meta.setdefault("stage_x", 0.0)
	meta.setdefault("stage_y", 0.0)
	meta.setdefault("stage_z", 0.0)
	meta.setdefault("physical_pixel_size_x", 1.0)
	meta.setdefault("physical_pixel_size_y", 1.0)
	meta.setdefault("physical_pixel_size_z", 1.0)

	return {
		"stage_x": float(meta["stage_x"]),
		"stage_y": float(meta["stage_y"]),
		"stage_z": float(meta["stage_z"]),
		"physical_pixel_size_x": float(meta["physical_pixel_size_x"]),
		"physical_pixel_size_y": float(meta["physical_pixel_size_y"]),
		"physical_pixel_size_z": float(meta["physical_pixel_size_z"]),
	}


def _device_scalar_float(value: Any) -> float:
	try:
		return float(value.item())
	except Exception:
		if _HAS_CUCIM and cp is not None:
			try:
				return float(cp.asnumpy(value))
			except Exception:
				pass
		return float(value)


def _phase_corr_fft_nd(
	a: np.ndarray,
	b: np.ndarray,
	use_gpu: bool,
) -> tuple[np.ndarray, float]:
	"""FFT phase correlation with the previous 3-point peak refinement."""
	if use_gpu and _HAS_CUCIM and cp is not None:
		xp = cp
	else:
		xp = np

	a_x = xp.asarray(a, dtype=xp.float32)
	b_x = xp.asarray(b, dtype=xp.float32)

	fa = xp.fft.fftn(a_x)
	fb = xp.fft.fftn(b_x)
	r = fa * xp.conj(fb)
	r = r / (xp.abs(r) + xp.float32(1e-8))
	corr = xp.fft.ifftn(r).real

	max_idx = xp.argmax(corr)
	peak_coords_raw = tuple(int(_device_scalar_float(v)) for v in xp.unravel_index(max_idx, corr.shape))
	center = corr[peak_coords_raw]

	shift_values: list[float] = []
	for axis, axis_size in enumerate(corr.shape):
		coord = int(peak_coords_raw[axis])
		peak_i = coord
		if peak_i > int(axis_size) // 2:
			peak_i -= int(axis_size)

		coord_m = list(peak_coords_raw)
		coord_p = list(peak_coords_raw)
		coord_m[axis] = (coord - 1) % int(axis_size)
		coord_p[axis] = (coord + 1) % int(axis_size)
		cm = corr[tuple(coord_m)]
		cp1 = corr[tuple(coord_p)]
		delta = (cp1 - cm) / (xp.float32(2.0) * (xp.float32(2.0) * center - cm - cp1) + xp.float32(1e-8))
		shift_values.append(float(peak_i) + _device_scalar_float(delta))

	peak_score = _device_scalar_float(center)
	return np.asarray(shift_values, dtype=np.float32), float(peak_score)


def phase_corr_gpu_2d(
	a: np.ndarray,
	b: np.ndarray,
	use_gpu: bool = True,
) -> tuple[np.ndarray, float]:
	"""Estimate subpixel (y, x) shift to align image b onto image a and return peak score."""
	if use_gpu:
		try:
			return _phase_corr_fft_nd(a, b, use_gpu=True)
		except Exception:
			pass
	return _phase_corr_fft_nd(a, b, use_gpu=False)


def estimate_overlap_shift_adjacent_slices(
	patch_i: np.ndarray,
	patch_j: np.ndarray,
	use_gpu: bool = True,
	max_shift_z_px: float = _MAX_SHIFT_Z_PX,
	max_shift_yx_px: float = _MAX_SHIFT_YX_PX,
) -> np.ndarray:
	"""Estimate (z, y, x) shift using only overlap slices and adjacent z-slice candidates.

	Z corrections are capped tightly at ``max_shift_z_px`` (default 2 px) because
	tiles are co-planar and large axial offsets almost always indicate noise rather
	than real displacement.  Y/X corrections are capped separately at
	``max_shift_yx_px`` (default 20 px).
	"""
	if patch_i.ndim != 3 or patch_j.ndim != 3:
		return np.zeros(3, dtype=np.float32)

	od = int(min(patch_i.shape[0], patch_j.shape[0]))
	if od < 2:
		return np.zeros(3, dtype=np.float32)

	best_shift = np.zeros(3, dtype=np.float32)
	best_score = -np.inf

	# Only test integer dz candidates within the tight z limit.
	max_dz_int = int(np.floor(max_shift_z_px))
	dz_candidates = list(range(-max_dz_int, max_dz_int + 1))

	for dz in dz_candidates:
		dy_vals: list[float] = []
		dx_vals: list[float] = []
		scores: list[float] = []

		for zi in range(od):
			zj = zi + dz
			if zj < 0 or zj >= od:
				continue

			slice_i = normalize_patch(patch_i[zi])
			slice_j = normalize_patch(patch_j[zj])
			if slice_i.size == 0 or slice_j.size == 0:
				continue

			shift_yx, score = phase_corr_gpu_2d(slice_i, slice_j, use_gpu=use_gpu)
			dy_vals.append(float(shift_yx[0]))
			dx_vals.append(float(shift_yx[1]))
			scores.append(float(score))

		if not scores:
			continue

		cand_shift = np.asarray(
			[
				float(dz),
				float(np.median(dy_vals)),
				float(np.median(dx_vals)),
			],
			dtype=np.float32,
		)

		# Clip z and yx axes independently with their respective limits.
		cand_shift[0] = float(np.clip(cand_shift[0], -max_shift_z_px, max_shift_z_px))
		cand_shift[1] = float(np.clip(cand_shift[1], -max_shift_yx_px, max_shift_yx_px))
		cand_shift[2] = float(np.clip(cand_shift[2], -max_shift_yx_px, max_shift_yx_px))

		cand_score = float(np.mean(scores))
		if cand_score > best_score:
			best_score = cand_score
			best_shift = cand_shift

	return best_shift.astype(np.float32, copy=False)


def normalize_patch(patch: np.ndarray) -> np.ndarray:
	arr = np.asarray(patch, dtype=np.float32)
	arr = arr - float(np.mean(arr))
	std = float(np.std(arr))
	if std > 1e-6:
		arr = arr / std
	return arr


def shift_block_tczyx_gpu(block: np.ndarray, shift_zyx: np.ndarray) -> np.ndarray:
	if cp is None or _shift_gpu is None:
		raise RuntimeError("CuPy + cupyx shift backend is required for stitching transforms.")
	gpu = shift_vec = shifted = None
	try:
		gpu = cp.asarray(block, dtype=cp.float32)
		shift_vec = cp.asarray([0.0, 0.0, float(shift_zyx[0]), float(shift_zyx[1]), float(shift_zyx[2])], dtype=cp.float32)
		shifted = _shift_gpu(
			gpu,
			shift=shift_vec,
			order=1,
			mode="constant",
			cval=0.0,
			prefilter=False,
		)
		return cp.asnumpy(shifted).astype(np.float32, copy=False)
	finally:
		del gpu, shift_vec, shifted
		maybe_free_gpu_memory()


def estimate_tile_backgrounds_tczyx(arr: da.Array) -> np.ndarray:
	"""Return one 5th-percentile background value per (T, C) tile volume."""
	tile = to_dask_tczyx(arr).astype(np.float32)
	t_dim = int(tile.shape[0])
	c_dim = int(tile.shape[1])
	background_tasks: list[da.Array] = []
	for t in range(t_dim):
		for c in range(c_dim):
			vol = tile[t, c]
			if _BACKGROUND_SAMPLE_STRIDE_YX > 1:
				sample = vol[:, ::_BACKGROUND_SAMPLE_STRIDE_YX, ::_BACKGROUND_SAMPLE_STRIDE_YX]
			else:
				sample = vol
			background_tasks.append(da.percentile(sample.ravel(), 5).astype(np.float32))
	if not background_tasks:
		return np.zeros((t_dim, c_dim), dtype=np.float32)
	with dask.config.set(scheduler="threads", num_workers=1):
		values = dask.compute(*background_tasks)
	return np.asarray(values, dtype=np.float32).reshape((t_dim, c_dim))


def _shift_or_identity_tczyx_block(block: np.ndarray, frac_zyx: np.ndarray) -> np.ndarray:
	frac = np.asarray(frac_zyx, dtype=np.float32)
	if np.all(np.abs(frac) <= 1e-6):
		return np.asarray(block, dtype=np.float32)
	return shift_block_tczyx_gpu(np.asarray(block, dtype=np.float32), frac)


def stitch_tiles_blockwise_tczyx(
	tile_arrays: list[da.Array],
	tile_backgrounds: list[np.ndarray],
	final_positions: np.ndarray,
	shape: tuple[int, int, int, int, int],
	chunk_shape_zyx: tuple[int, int, int] | None = None,
) -> da.Array:
	"""Build a stitched TCZYX mosaic without retaining shifted full-tile graphs."""
	if not tile_arrays:
		raise ValueError("No tile arrays were provided for blockwise stitching.")
	if len(tile_arrays) != len(tile_backgrounds):
		raise ValueError("tile_arrays and tile_backgrounds must have the same length.")

	chunk_zyx = bounded_spatial_chunk_shape(
		shape[2:],
		base_chunk_zyx=chunk_shape_zyx or _STITCH_CHUNK_SHAPE,
	)
	template = da.empty(shape, chunks=(1, 1, chunk_zyx[0], chunk_zyx[1], chunk_zyx[2]), dtype=np.float32)
	tile_shapes = [tuple(int(v) for v in tile.shape) for tile in tile_arrays]
	final_positions_np = np.asarray(final_positions, dtype=np.float32)

	def render_block(dummy: np.ndarray, block_info: dict | None = None) -> np.ndarray:
		_ = dummy
		if block_info is None:
			raise RuntimeError("Dask did not provide block_info for stitch rendering.")
		location = block_info[None]["array-location"]
		t0, t1 = (int(location[0][0]), int(location[0][1]))
		c0, c1 = (int(location[1][0]), int(location[1][1]))
		z0, z1 = (int(location[2][0]), int(location[2][1]))
		y0, y1 = (int(location[3][0]), int(location[3][1]))
		x0, x1 = (int(location[4][0]), int(location[4][1]))

		out = np.zeros((t1 - t0, c1 - c0, z1 - z0, y1 - y0, x1 - x0), dtype=np.float32)
		weights = np.zeros_like(out)

		for tile_idx, tile in enumerate(tile_arrays):
			_, _, tile_z, tile_y, tile_x = tile_shapes[tile_idx]
			pos_z = float(final_positions_np[tile_idx, 0])
			pos_y = float(final_positions_np[tile_idx, 1])
			pos_x = float(final_positions_np[tile_idx, 2])
			int_z = int(np.floor(pos_z))
			int_y = int(np.floor(pos_y))
			int_x = int(np.floor(pos_x))
			frac = np.asarray(
				[pos_z - float(int_z), pos_y - float(int_y), pos_x - float(int_x)],
				dtype=np.float32,
			)

			gz0 = max(z0, int_z)
			gy0 = max(y0, int_y)
			gx0 = max(x0, int_x)
			gz1 = min(z1, int_z + tile_z)
			gy1 = min(y1, int_y + tile_y)
			gx1 = min(x1, int_x + tile_x)
			if gz0 >= gz1 or gy0 >= gy1 or gx0 >= gx1:
				continue

			lz0 = gz0 - int_z
			ly0 = gy0 - int_y
			lx0 = gx0 - int_x
			lz1 = gz1 - int_z
			ly1 = gy1 - int_y
			lx1 = gx1 - int_x

			read_z0 = max(0, int(np.floor(float(lz0) - float(frac[0]))) - 2)
			read_y0 = max(0, int(np.floor(float(ly0) - float(frac[1]))) - 2)
			read_x0 = max(0, int(np.floor(float(lx0) - float(frac[2]))) - 2)
			read_z1 = min(tile_z, int(np.ceil(float(lz1) - float(frac[0]))) + 2)
			read_y1 = min(tile_y, int(np.ceil(float(ly1) - float(frac[1]))) + 2)
			read_x1 = min(tile_x, int(np.ceil(float(lx1) - float(frac[2]))) + 2)
			if read_z0 >= read_z1 or read_y0 >= read_y1 or read_x0 >= read_x1:
				continue

			with dask.config.set(scheduler="single-threaded"):
				raw = tile[
					t0:t1,
					c0:c1,
					read_z0:read_z1,
					read_y0:read_y1,
					read_x0:read_x1,
				].compute()
			raw = np.asarray(raw, dtype=np.float32)
			bg = tile_backgrounds[tile_idx][t0:t1, c0:c1].reshape((t1 - t0, c1 - c0, 1, 1, 1))
			raw = (raw - bg).astype(np.float32, copy=False)

			shifted = _shift_or_identity_tczyx_block(raw, frac)
			if np.all(np.abs(frac) <= 1e-6):
				shifted_weights = np.ones(raw.shape, dtype=np.float32)
			else:
				shifted_weights = _shift_or_identity_tczyx_block(np.ones(raw.shape, dtype=np.float32), frac)
				shifted_weights = np.clip(shifted_weights, 0.0, 1.0).astype(np.float32, copy=False)

			crop_z0 = lz0 - read_z0
			crop_y0 = ly0 - read_y0
			crop_x0 = lx0 - read_x0
			crop_z1 = crop_z0 + (gz1 - gz0)
			crop_y1 = crop_y0 + (gy1 - gy0)
			crop_x1 = crop_x0 + (gx1 - gx0)
			out_z0 = gz0 - z0
			out_y0 = gy0 - y0
			out_x0 = gx0 - x0
			out_z1 = out_z0 + (gz1 - gz0)
			out_y1 = out_y0 + (gy1 - gy0)
			out_x1 = out_x0 + (gx1 - gx0)

			out[:, :, out_z0:out_z1, out_y0:out_y1, out_x0:out_x1] += shifted[
				:, :, crop_z0:crop_z1, crop_y0:crop_y1, crop_x0:crop_x1
			]
			weights[:, :, out_z0:out_z1, out_y0:out_y1, out_x0:out_x1] += shifted_weights[
				:, :, crop_z0:crop_z1, crop_y0:crop_y1, crop_x0:crop_x1
			]

		np.divide(out, weights, out=out, where=weights > 0.0)
		out[weights <= 0.0] = 0.0
		return out.astype(np.float32, copy=False)

	return da.map_blocks(
		render_block,
		template,
		dtype=np.float32,
		chunks=template.chunks,
		meta=np.empty((0, 0, 0, 0, 0), dtype=np.float32),
	)


def fuse_tiles_with_phasecorr_dapi(
	series_groups: list[tuple[zarr.Group, str]],
	tile_metadata_by_group: dict[str, dict[str, Any]],
	dapi_channel: int,
	dapi_timepoint: int,
	stitch_chunk_shape_zyx: tuple[int, int, int] | None = None,
) -> da.Array:
	"""Stitch tiles by stage-prior + DAPI FFT phase correlation and return TCZYX dask array.

	Before placement each tile's per-(T, C) volume has its 5th-percentile
	background subtracted so that tiles acquired at different illumination
	levels share a common baseline.  This is the primary fix for visible
	rectangular seam artefacts.  Z-axis shift corrections are capped to
	``_MAX_SHIFT_Z_PX`` (2 px); Y/X corrections are capped to
	``_MAX_SHIFT_YX_PX`` (20 px).
	"""
	if not series_groups:
		raise ValueError("No image series groups were provided for stitching.")

	tile_arrays: list[da.Array] = []
	tile_backgrounds: list[np.ndarray] = []
	stage_positions_px: list[np.ndarray] = []
	dapi_volumes: list[da.Array] = []

	for group, group_path in series_groups:
		group_key = group_path.strip("/") or "/"
		tile_meta = tile_metadata_by_group.get(group_key)
		transform = extract_global_transform_from_group(group, tile_meta)

		arr = read_image_data_from_group(group).astype(np.float32)
		if arr.ndim != 5:
			raise ValueError(
				f"Expected each stitched input tile to be TCZYX (5D), got {arr.shape} for '{group_path}'."
			)
		t_dim = int(arr.shape[0])
		c_dim = int(arr.shape[1])
		if dapi_timepoint < 0 or dapi_timepoint >= t_dim:
			raise ValueError(f"dapi_timepoint={dapi_timepoint} out of bounds for group '{group_path}' (T={t_dim})")
		if dapi_channel < 0 or dapi_channel >= c_dim:
			raise ValueError(f"dapi_channel={dapi_channel} out of bounds for group '{group_path}' (C={c_dim})")

		backgrounds = estimate_tile_backgrounds_tczyx(arr)
		tile_arrays.append(arr)
		tile_backgrounds.append(backgrounds)
		dapi_volumes.append(
			da.clip(arr[dapi_timepoint, dapi_channel].astype(np.float32) - float(backgrounds[dapi_timepoint, dapi_channel]), 0.0, None)
		)

		pps_z = float(transform["physical_pixel_size_z"]) if float(transform["physical_pixel_size_z"]) > 0 else 1.0
		pps_y = float(transform["physical_pixel_size_y"]) if float(transform["physical_pixel_size_y"]) > 0 else 1.0
		pps_x = float(transform["physical_pixel_size_x"]) if float(transform["physical_pixel_size_x"]) > 0 else 1.0
		stage_positions_px.append(
			np.asarray([
				float(transform["stage_z"]) / pps_z,
				float(transform["stage_y"]) / pps_y,
				float(transform["stage_x"]) / pps_x,
			], dtype=np.float32)
		)

	if len(tile_arrays) == 1:
		return to_dask_tczyx(tile_arrays[0])

	initial_positions = np.stack(stage_positions_px, axis=0)
	initial_positions -= np.min(initial_positions, axis=0, keepdims=True)

	use_gpu_phasecorr = available_gpu_count() > 0
	tile_offsets = np.zeros_like(initial_positions, dtype=np.float32)

	for j in range(1, len(tile_arrays)):
		dj = int(dapi_volumes[j].shape[0])
		hj = int(dapi_volumes[j].shape[1])
		wj = int(dapi_volumes[j].shape[2])
		best_i = None
		best_overlap = -1

		for i in range(j):
			di = int(dapi_volumes[i].shape[0])
			hi = int(dapi_volumes[i].shape[1])
			wi = int(dapi_volumes[i].shape[2])

			zi, yi, xi = initial_positions[i]
			zj, yj, xj = initial_positions[j]
			z0 = max(float(zi), float(zj))
			y0 = max(float(yi), float(yj))
			x0 = max(float(xi), float(xj))
			z1 = min(float(zi + di), float(zj + dj))
			y1 = min(float(yi + hi), float(yj + hj))
			x1 = min(float(xi + wi), float(xj + wj))
			overlap_d = int(np.floor(z1 - z0))
			overlap_h = int(np.floor(y1 - y0))
			overlap_w = int(np.floor(x1 - x0))
			overlap = overlap_d * overlap_h * overlap_w
			if overlap > best_overlap and overlap_d >= 8 and overlap_h >= 32 and overlap_w >= 32:
				best_overlap = overlap
				best_i = i

		if best_i is None:
			# Fall back to stage-only placement when no overlap is available.
			tile_offsets[j] = tile_offsets[j - 1]
			continue

		i = best_i
		zi, yi, xi = initial_positions[i]
		zj, yj, xj = initial_positions[j]
		di = int(dapi_volumes[i].shape[0])
		hi = int(dapi_volumes[i].shape[1])
		wi = int(dapi_volumes[i].shape[2])
		dj = int(dapi_volumes[j].shape[0])
		hj = int(dapi_volumes[j].shape[1])
		wj = int(dapi_volumes[j].shape[2])

		z0 = max(float(zi), float(zj))
		y0 = max(float(yi), float(yj))
		x0 = max(float(xi), float(xj))
		z1 = min(float(zi + di), float(zj + dj))
		y1 = min(float(yi + hi), float(yj + hj))
		x1 = min(float(xi + wi), float(xj + wj))
		od = int(np.floor(z1 - z0))
		oh = int(np.floor(y1 - y0))
		ow = int(np.floor(x1 - x0))

		iz0 = int(np.floor(z0 - zi))
		iy0 = int(np.floor(y0 - yi))
		ix0 = int(np.floor(x0 - xi))
		jz0 = int(np.floor(z0 - zj))
		jy0 = int(np.floor(y0 - yj))
		jx0 = int(np.floor(x0 - xj))

		patch_i = np.asarray(dapi_volumes[i][iz0:iz0 + od, iy0:iy0 + oh, ix0:ix0 + ow].compute(), dtype=np.float32)
		patch_j = np.asarray(dapi_volumes[j][jz0:jz0 + od, jy0:jy0 + oh, jx0:jx0 + ow].compute(), dtype=np.float32)
		shift_zyx = estimate_overlap_shift_adjacent_slices(
			patch_i,
			patch_j,
			use_gpu=use_gpu_phasecorr,
			max_shift_z_px=_MAX_SHIFT_Z_PX,
			max_shift_yx_px=_MAX_SHIFT_YX_PX,
		)

		# If j must be shifted by shift_zyx to match i, its global offset moves opposite.
		# Clamp z and yx independently before storing the offset.
		clamped = np.asarray([
			float(np.clip(shift_zyx[0], -_MAX_SHIFT_Z_PX, _MAX_SHIFT_Z_PX)),
			float(np.clip(shift_zyx[1], -_MAX_SHIFT_YX_PX, _MAX_SHIFT_YX_PX)),
			float(np.clip(shift_zyx[2], -_MAX_SHIFT_YX_PX, _MAX_SHIFT_YX_PX)),
		], dtype=np.float32)
		tile_offsets[j] = tile_offsets[i] - clamped

	initial_extent_zyx = np.zeros(3, dtype=np.float32)
	for idx, tile in enumerate(tile_arrays):
		d = float(int(tile.shape[2]))
		h = float(int(tile.shape[3]))
		w = float(int(tile.shape[4]))
		initial_extent_zyx[0] = max(initial_extent_zyx[0], float(initial_positions[idx, 0]) + d)
		initial_extent_zyx[1] = max(initial_extent_zyx[1], float(initial_positions[idx, 1]) + h)
		initial_extent_zyx[2] = max(initial_extent_zyx[2], float(initial_positions[idx, 2]) + w)

	final_positions = initial_positions + tile_offsets
	for axis in range(3):
		low = float(np.min(final_positions[:, axis]))
		high = -np.inf
		for idx, tile in enumerate(tile_arrays):
			tile_span = float(int(tile.shape[2 + axis]))
			high = max(high, float(final_positions[idx, axis]) + tile_span)
		span = float(initial_extent_zyx[axis])
		t_min = -low
		t_max = span - high
		if t_min <= t_max:
			t = 0.0
			if t < t_min:
				t = t_min
			elif t > t_max:
				t = t_max
		else:
			t = t_min
		final_positions[:, axis] = final_positions[:, axis] + float(t)

	max_z = 0.0
	max_y = 0.0
	max_x = 0.0
	for idx, tile in enumerate(tile_arrays):
		d = float(int(tile.shape[2]))
		h = float(int(tile.shape[3]))
		w = float(int(tile.shape[4]))
		max_z = max(max_z, float(final_positions[idx, 0]) + d)
		max_y = max(max_y, float(final_positions[idx, 1]) + h)
		max_x = max(max_x, float(final_positions[idx, 2]) + w)

	out_t = int(tile_arrays[0].shape[0])
	out_c = int(tile_arrays[0].shape[1])
	out_z = max(int(np.ceil(max_z)), int(np.ceil(float(initial_extent_zyx[0]))))
	out_y = max(int(np.ceil(max_y)), int(np.ceil(float(initial_extent_zyx[1]))))
	out_x = max(int(np.ceil(max_x)), int(np.ceil(float(initial_extent_zyx[2]))))

	shape = (out_t, out_c, out_z, out_y, out_x)
	return stitch_tiles_blockwise_tczyx(
		tile_arrays=tile_arrays,
		tile_backgrounds=tile_backgrounds,
		final_positions=final_positions,
		shape=shape,
		chunk_shape_zyx=stitch_chunk_shape_zyx,
	)


def get_multiscales(group: zarr.Group) -> list | None:
	"""Return the multiscales list from a group's attrs.
	OME-NGFF 0.4 writes:  group.attrs["multiscales"] = [...]
	OME-NGFF 0.5 writes:  group.attrs["ome"]["multiscales"] = [...]
	Both layouts must be handled so that series enumeration and data reading
	work regardless of which ome-zarr version wrote the store.
	"""
	# Flat layout (0.4)
	ms = group.attrs.get("multiscales")
	if isinstance(ms, list) and ms:
		return ms
	# Nested layout (0.5)
	ome = group.attrs.get("ome")
	if isinstance(ome, dict):
		ms = ome.get("multiscales")
		if isinstance(ms, list) and ms:
			return ms
	return None


def enumerate_series_groups(in_root: zarr.Group) -> list[tuple[zarr.Group, str]]:
	"""Return (group, relative_path) pairs for every image series in the store.
	Handles three layouts in priority order:
	1. bioformats2raw — reads series paths from OME/.zattrs["series"]
	2. Single image at root — root itself has "multiscales"
	3. Flat numeric children — sub-groups named "0", "1", … that have "multiscales"
	"""
	results: list[tuple[zarr.Group, str]] = []

	# Bioformats2raw (OME group carries "series" list)
	if "OME" in in_root:
		ome_group = in_root["OME"]
		if isinstance(ome_group, zarr.Group):
			series_paths: list[str] = ome_group.attrs.get("series", [])
			if isinstance(series_paths, list) and series_paths:
				for series_path in series_paths:
					series_path = str(series_path).strip("/")
					if series_path in in_root:
						candidate = in_root[series_path]
						if isinstance(candidate, zarr.Group) and get_multiscales(candidate) is not None:
							results.append((candidate, series_path))
				if results:
					return results

	# Single image at root
	if get_multiscales(in_root) is not None:
		return [(in_root, "/")]

	# Flat numeric children
	for key in sorted(in_root.keys()):
		candidate = in_root[key]
		if isinstance(candidate, zarr.Group) and get_multiscales(candidate) is not None:
			results.append((candidate, key))

	return results


def read_image_data_from_group(group: zarr.Group) -> da.Array:
	"""Read the highest-resolution multiscale level from a group as a Dask array."""
	return da.from_zarr(_get_primary_dataset(group))


# ---------------------------------------------------------------------------
# Collection metadata
# ---------------------------------------------------------------------------


def write_image_collection_metadata(root_group: zarr.Group, image_group_paths: list[str]) -> None:
	"""Write bioformats2raw v3 collection metadata so _enumerate_series_groups can find all FOVs.

	Mirrors convert_to_ngff._write_image_collection_metadata exactly:
	- root .zattrs  → {"bioformats2raw.layout": 3}   (flat, NOT nested under "ome")
	- OME/.zattrs   → {"series": ["fovs/fov_0000", ...]}  (flat, NOT nested under "ome")
	"""
	if not image_group_paths:
		return

	root_group.attrs["bioformats2raw.layout"] = 3

	ome_group = root_group.require_group("OME")
	ome_group.attrs["series"] = image_group_paths


def _block_to_storage_numpy(block: Any) -> np.ndarray:
	if _HAS_CUCIM and cp is not None and isinstance(block, cp.ndarray):
		block = cp.asnumpy(block)
	return cast_numpy_block_for_storage(block, _STORAGE_DTYPE)


def process_zarr(
	input_zarr_path: str | Path,
	bead_channel: int | None,
	psf_path: str | Path,
	output_path: str | Path,
	intermediate_path: str | Path | None = None,
	highpass_sigma: float | tuple[float, float, float] = 3.0,
	lowpass_sigma: float | tuple[float, float, float] = 1.0,
	deconvolution_iterations: int = 10,
	dapi_channel: int | None = None,
	dapi_timepoint: int | None = None,
	timing: bool = False,
	write_max_workers: int | None = None,
	is_confocal: bool = False,
	batch_size_xyz: tuple[int, int, int] | None = None,
	processed_write_batch_size_xyz: tuple[int, int, int] | None = None,
) -> Path:
	"""Fuse all tiles, process the stitched image, and save to one output OME-Zarr."""
	in_path = Path(input_zarr_path)
	out_path = Path(output_path)
	if out_path.suffix != ".zarr":
		out_path = out_path.with_suffix(".zarr")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	batch_shape_zyx = batch_size_xyz_to_zyx(batch_size_xyz)
	processed_write_shape_zyx = batch_size_xyz_to_zyx(processed_write_batch_size_xyz)

	in_root = open_zarr_group_with_optional_kvikio(in_path, mode="r")
	in_uses_kvikio = bool(
		_HAS_KVIKIO and GDSStore is not None and isinstance(getattr(in_root, "store", None), GDSStore)
	)
	print(f"[image_processing] kvikio input_store={in_uses_kvikio} path={in_path}", flush=True)
	series_groups = enumerate_series_groups(in_root)
	if not series_groups:
		raise ValueError(f"No image nodes found in input OME-Zarr: {in_path}")

	root_channel_metadata = extract_channel_metadata_from_group(in_root)
	tile_metadata = in_root.attrs.get("tile_metadata", [])
	tile_metadata_by_group: dict[str, dict[str, Any]] = {}
	if isinstance(tile_metadata, list):
		for tile_entry in tile_metadata:
			if not isinstance(tile_entry, dict):
				continue
			group_name = tile_entry.get("ngff_group")
			if isinstance(group_name, str) and group_name:
				tile_metadata_by_group[group_name.strip("/") or "/"] = tile_entry

	out_root = open_zarr_group_with_optional_kvikio(out_path, mode="w")
	out_uses_kvikio = bool(
		_HAS_KVIKIO and GDSStore is not None and isinstance(getattr(out_root, "store", None), GDSStore)
	)
	print(f"[image_processing] kvikio output_store={out_uses_kvikio} path={out_path}", flush=True)
	rt_versions = package_versions()
	if rt_versions:
		print(
			"[image_processing] rapids_io "
			f"kvikio={rt_versions.get('kvikio', 'unknown')} "
			f"cucim={rt_versions.get('cucim', 'unknown')} "
			f"cudf={rt_versions.get('cudf', 'unknown')} "
			f"cuml={rt_versions.get('cuml', 'unknown')}",
			flush=True,
		)
	intermediate_root = out_root
	intermediate_store_path = out_path
	if intermediate_path is not None:
		inter_path = Path(intermediate_path)
		if inter_path.suffix != ".zarr":
			inter_path = inter_path.with_suffix(".zarr")
		inter_path.parent.mkdir(parents=True, exist_ok=True)
		intermediate_root = open_zarr_group_with_optional_kvikio(inter_path, mode="w")
		intermediate_store_path = inter_path
		inter_uses_kvikio = bool(
			_HAS_KVIKIO and GDSStore is not None and isinstance(getattr(intermediate_root, "store", None), GDSStore)
		)
		print(f"[image_processing] kvikio intermediate_store={inter_uses_kvikio} path={inter_path}", flush=True)

	timing_stats: dict[str, float] | None = {} if timing else None
	t_pipeline = time.perf_counter()
	t_stitch_graph = time.perf_counter()
	stitch_dapi_channel = int(dapi_channel) if dapi_channel is not None else 0
	stitch_dapi_timepoint = int(dapi_timepoint) if dapi_timepoint is not None else 0
	stitched_image = fuse_tiles_with_phasecorr_dapi(
		series_groups=series_groups,
		tile_metadata_by_group=tile_metadata_by_group,
		dapi_channel=stitch_dapi_channel,
		dapi_timepoint=stitch_dapi_timepoint,
		stitch_chunk_shape_zyx=batch_shape_zyx,
	)
	stitched_tczyx = to_dask_tczyx(stitched_image, chunk_shape_zyx=batch_shape_zyx).astype(np.float32)
	if stitched_tczyx.ndim != 5:
		raise ValueError(f"Expected stitched TCZYX output with ndim=5, got shape {stitched_tczyx.shape}")
	profile_add(timing_stats, "stitch_graph_s", time.perf_counter() - t_stitch_graph)

	# Break the tile-stitching graph before building the filtering/deconvolution
	# graph so downstream processing reads a plain chunked Zarr array.
	tmp_stitched_path = "_tmp_stitched_tczyx"
	t_stitch_materialize = time.perf_counter()
	gpu_count = available_gpu_count()
	stitched_with_kvikio = write_dask_array_to_zarr(
		stitched_tczyx,
		root_path=intermediate_store_path,
		component=tmp_stitched_path,
		overwrite=True,
		max_workers=1,
		num_gpus_per_task=1.0 if gpu_count > 0 else 0.0,
	)
	if not stitched_with_kvikio:
		with dask.config.set(scheduler="threads", num_workers=1):
			da.to_zarr(stitched_tczyx, intermediate_root.store, component=tmp_stitched_path, overwrite=True)
	profile_add(timing_stats, "stitch_materialize_s", time.perf_counter() - t_stitch_materialize)
	stitched_image = da.from_zarr(intermediate_root[tmp_stitched_path]).astype(np.float32)
	del stitched_tczyx
	clear_fft_plan_cache()
	maybe_free_gpu_memory(force=True)

	channel_metadata = root_channel_metadata
	if channel_metadata is None and series_groups:
		channel_metadata = extract_channel_metadata_from_group(series_groups[0][0])

	precomputed_psfs = resolve_psfs_for_channels(
		psf_path=psf_path,
		channel_metadata=channel_metadata,
		channel_count=int(stitched_image.shape[1]),
	)

	t_process = time.perf_counter()
	processed = process_stack(
		fov_image=stitched_image,
		bead_channel=bead_channel,
		psf_path=psf_path,
		output_path=out_path,
		channel_metadata=channel_metadata,
		highpass_sigma=highpass_sigma,
		lowpass_sigma=lowpass_sigma,
		deconvolution_iterations=deconvolution_iterations,
		precomputed_psfs=precomputed_psfs,
		timing_stats=timing_stats,
		dapi_channel=dapi_channel,
		dapi_timepoint=dapi_timepoint,
		is_confocal=is_confocal,
		chunk_shape_zyx=batch_shape_zyx,
	)
	profile_add(timing_stats, "process_compute_s", time.perf_counter() - t_process)
	processed_tczyx = to_dask_tczyx(
		processed,
		chunk_shape_zyx=batch_shape_zyx or _gpu_filter_chunk_shape_for_image(processed),
	).astype(np.float32)
	if processed_tczyx.ndim != 5:
		raise ValueError(f"Expected processed TCZYX output with ndim=5, got shape {processed_tczyx.shape}")
	processed_storage = processed_tczyx.map_blocks(
		_block_to_storage_numpy,
		dtype=_STORAGE_DTYPE,
		meta=np.empty((0,), dtype=_STORAGE_DTYPE),
	)
	processed_storage = rechunk_tczyx_for_write(
		processed_storage,
		write_chunk_shape_zyx=processed_write_shape_zyx,
	)
	if processed_write_shape_zyx is not None:
		print(
			"[image_processing] processed_write_chunks="
			f"{tuple(tuple(int(v) for v in axis_chunks) for axis_chunks in processed_storage.chunks)}",
			flush=True,
		)
	tmp_processed_path = "_tmp_processed_tczyx"
	t_materialize = time.perf_counter()
	materialized_with_kvikio = write_dask_array_to_zarr(
		processed_storage,
		root_path=intermediate_store_path,
		component=tmp_processed_path,
		overwrite=True,
		max_workers=max(1, gpu_count),
		num_gpus_per_task=1.0 if gpu_count > 0 else 0.0,
	)
	if not materialized_with_kvikio:
		with dask.config.set(scheduler="threads", num_workers=1):
			da.to_zarr(processed_storage, intermediate_root.store, component=tmp_processed_path, overwrite=True)
	profile_add(timing_stats, "materialize_s", time.perf_counter() - t_materialize)
	processed_cached = da.from_zarr(intermediate_root[tmp_processed_path])

	write_workers = max(1, int(write_max_workers)) if write_max_workers is not None else 1

	stitched_group_path = "stitched"
	target_group = out_root.require_group(stitched_group_path)
	stitched_transform = extract_global_transform_from_group(series_groups[0][0], tile_metadata_by_group.get(series_groups[0][1].strip("/") or "/"))
	t_write = time.perf_counter()
	copied_with_kvikio = copy_zarr_array(
		source_root_path=intermediate_store_path,
		source_component=tmp_processed_path,
		dest_root_path=out_path,
		dest_component=f"{stitched_group_path}/0",
		overwrite=True,
		max_workers=write_workers,
	)
	if not copied_with_kvikio:
		with dask.config.set(scheduler="threads", num_workers=write_workers):
			da.to_zarr(processed_cached, out_root.store, component=f"{stitched_group_path}/0", overwrite=True)
	target_group.attrs["multiscales"] = [
		{
			"version": "0.4",
			"axes": [{"name": ax, "type": _AXIS_TYPES.get(ax, "space")} for ax in "tczyx"],
			"datasets": [{"path": "0"}],
		}
	]
	profile_add(timing_stats, "write_s", time.perf_counter() - t_write)
	if tmp_processed_path in intermediate_root:
		del intermediate_root[tmp_processed_path]
	if tmp_stitched_path in intermediate_root:
		del intermediate_root[tmp_stitched_path]
	clear_fft_plan_cache()
	maybe_free_gpu_memory(force=True)

	out_root.attrs["processed_fov_count"] = 1
	out_root.attrs["processed_fov_groups"] = [
		{
			"fov_index": 0,
			"source_node_index": -1,
			"source_node_group": "stitched",
			"source_local_index": 0,
			"output_group": stitched_group_path,
			"source_tile_metadata": {
				"ngff_group": "stitched",
				"stage_x": 0.0,
				"stage_y": 0.0,
				"stage_z": 0.0,
				"physical_pixel_size_x": float(stitched_transform["physical_pixel_size_x"]),
				"physical_pixel_size_y": float(stitched_transform["physical_pixel_size_y"]),
				"physical_pixel_size_z": float(stitched_transform["physical_pixel_size_z"]),
			},
		},
	]
	if root_channel_metadata is not None:
		out_root.attrs["channel_metadata"] = root_channel_metadata
	if isinstance(tile_metadata, list):
		out_root.attrs["source_tile_metadata"] = tile_metadata

	# Write bioformats2raw collection metadata so downstream readers can
	# discover every FOV group via OME/"series" — matching the layout that
	# convert_to_ngff produces and that _enumerate_series_groups expects.
	write_image_collection_metadata(out_root, [stitched_group_path])
	profile_add(timing_stats, "total_s", time.perf_counter() - t_pipeline)
	if timing and timing_stats is not None:
		print_profile_summary(timing_stats)

	return out_path


try:
	from .tracing import instrument_module_functions
except ImportError:  # pragma: no cover - supports direct module execution
	from tracing import instrument_module_functions

instrument_module_functions(globals())
