"""Decode MERFISH image stacks into spot-level gene calls."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import zarr

try:
	from kvikio.zarr import GDSStore
	_HAS_KVIKIO = True
except Exception:
	GDSStore = None
	_HAS_KVIKIO = False

try:
	import cupy as cp
	from cucim.skimage.measure import label as _label_gpu

	_HAS_CUCIM = True
except Exception:
	cp = None
	_label_gpu = None
	_HAS_CUCIM = False

try:
	import cudf

	_HAS_CUDF = True
except Exception:
	cudf = None
	_HAS_CUDF = False

try:
	from cuml.neighbors import NearestNeighbors as _CuMLNearestNeighbors

	_HAS_CUML = True
except Exception:
	_CuMLNearestNeighbors = None
	_HAS_CUML = False

def _env_int(name: str, default: int, min_value: int = 1) -> int:
	value = os.environ.get(name)
	if value is None:
		return int(default)
	try:
		parsed = int(value)
	except ValueError as exc:
		raise ValueError(f"{name} must be an integer, got {value!r}") from exc
	return max(int(min_value), int(parsed))


_FALLBACK_DECODE_CHUNK_SHAPE: tuple[int, int, int] = (
	_env_int("BARRACUDA_DECODE_CHUNK_Z", 4),
	_env_int("BARRACUDA_DECODE_CHUNK_Y", 512),
	_env_int("BARRACUDA_DECODE_CHUNK_X", 512),
)
_DECODE_PIXELS_PER_BATCH = _env_int("BARRACUDA_DECODE_PIXELS_PER_BATCH", 65_536)
_EXTRACT_GROUP_BY_ROUND_CHANNEL = bool(_env_int("BARRACUDA_EXTRACT_GROUP_BY_ROUND_CHANNEL", 0, min_value=0))
_DECODE_TILE_SHAPE_ZYX: tuple[int, int, int] = (
	_env_int("BARRACUDA_DECODE_TILE_Z", 0, min_value=0),
	_env_int("BARRACUDA_DECODE_TILE_Y", 2048),
	_env_int("BARRACUDA_DECODE_TILE_X", 2048),
)
_DECODE_TILE_OVERLAP_ZYX: tuple[int, int, int] = (
	_env_int("BARRACUDA_DECODE_TILE_OVERLAP_Z", 0, min_value=0),
	_env_int("BARRACUDA_DECODE_TILE_OVERLAP_Y", 64, min_value=0),
	_env_int("BARRACUDA_DECODE_TILE_OVERLAP_X", 64, min_value=0),
)


def _read_codebook_csv(path: str | Path) -> pd.DataFrame:
	"""Read a codebook CSV, preferring cuDF I/O when RAPIDS is available."""
	if _HAS_CUDF and cudf is not None:
		try:
			return cudf.read_csv(str(path)).to_pandas()
		except Exception:
			pass
	return pd.read_csv(path)


def _write_records_csv(records: list[dict[str, Any]], columns: list[str], output_path: str | Path) -> Path:
	"""Write records to CSV, preferring cuDF for GPU-friendly tabular output."""
	out = Path(output_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	if _HAS_CUDF and cudf is not None:
		try:
			cudf.DataFrame(records, columns=columns).to_csv(str(out), index=False)
			return out
		except Exception:
			pass
	pd.DataFrame(records, columns=columns).to_csv(out, index=False)
	return out


def _as_cupy_array(value: Any) -> Any:
	if cp is None:
		return value
	if hasattr(value, "to_cupy"):
		return value.to_cupy()
	return cp.asarray(value)


def _profile_add(profile: dict[str, float] | None, key: str, value: float) -> None:
	if profile is None:
		return
	profile[key] = float(profile.get(key, 0.0) + float(value))


def _print_profile_summary(profile: dict[str, float], prefix: str = "[decode_spots]") -> None:
	keys = [
		"total_s",
		"read_s",
		"normalize_s",
		"assign_s",
		"assign_dask_to_numpy_s",
		"assign_gpu_transfer_in_s",
		"assign_cuml_compute_s",
		"assign_gpu_transfer_out_s",
		"combine_s",
		"merge_filter_save_s",
	]
	metrics = ", ".join(
		f"{key}={profile[key]:.2f}s"
		for key in keys
		if float(profile.get(key, 0.0)) > 0.0
	)
	print(f"{prefix} timing summary: {metrics}", flush=True)


def _bounded_spatial_chunk_shape(
	shape_zyx: tuple[int, int, int],
	base_chunk_zyx: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
	base = _FALLBACK_DECODE_CHUNK_SHAPE if base_chunk_zyx is None else base_chunk_zyx
	return (
		max(1, min(int(shape_zyx[0]), int(base[0]))),
		max(1, min(int(shape_zyx[1]), int(base[1]))),
		max(1, min(int(shape_zyx[2]), int(base[2]))),
	)


def _gpu_free_memory_bytes() -> int | None:
	if not _HAS_CUCIM or cp is None:
		return None
	try:
		free_bytes, _ = cp.cuda.runtime.memGetInfo()
		if int(free_bytes) > 0:
			return int(free_bytes)
	except Exception:
		pass
	return None


def _gpu_volume_fits(
	shape: tuple[int, ...],
	dtype: Any,
	expansion_factor: float,
	vram_fraction: float,
) -> bool:
	free_vram = _gpu_free_memory_bytes()
	if free_vram is None:
		return False
	itemsize = max(1, int(np.dtype(dtype).itemsize))
	n_items = int(np.prod(tuple(int(v) for v in shape), dtype=np.int64))
	estimated = int(float(n_items * itemsize) * float(expansion_factor))
	return estimated <= int(float(free_vram) * float(vram_fraction))


def _to_dask_tczyx(
	image: np.ndarray | da.Array,
	chunk_shape_zyx: tuple[int, int, int] | None = None,
) -> da.Array:
	"""Convert input image to a Dask TCZYX array."""
	arr = da.asarray(image)
	if arr.ndim != 5:
		raise ValueError(f"Expected TCZYX array (ndim=5), got shape {arr.shape}")
	chunk_z, chunk_y, chunk_x = _bounded_spatial_chunk_shape(
		(int(arr.shape[2]), int(arr.shape[3]), int(arr.shape[4])),
		base_chunk_zyx=chunk_shape_zyx,
	)
	return arr.rechunk((1, 1, chunk_z, chunk_y, chunk_x))


# ---------------------------------------------------------------------------
# OME-Zarr layout helpers
# ---------------------------------------------------------------------------


def _get_multiscales(group: zarr.Group) -> list | None:
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


def _enumerate_series_groups(in_root: zarr.Group) -> list[tuple[zarr.Group, str]]:
	"""Return (group, relative_path) pairs for every image series in the store.

	Handles three layouts in priority order:
	1. bioformats2raw — reads series paths from OME/.zattrs["series"]
	2. Single image at root — root itself has "multiscales"
	3. Flat numeric children — sub-groups named "0", "1", … that have "multiscales"

	Using Reader(loc)() to discover sub-groups in a bioformats2raw store does
	not work because the Reader only yields image nodes for groups that have
	"multiscales" in their own attrs.  The root of a bioformats2raw store has
	no "multiscales", so the generator returns an empty sequence.  Direct
	enumeration from OME/.zattrs["series"] is the correct approach.
	"""
	results: list[tuple[zarr.Group, str]] = []

	# --- Layout 1: bioformats2raw (OME group carries "series" list) ----------
	if "OME" in in_root:
		ome_group = in_root["OME"]
		if isinstance(ome_group, zarr.Group):
			series_paths: list[str] = ome_group.attrs.get("series", [])
			if isinstance(series_paths, list) and series_paths:
				for series_path in series_paths:
					series_path = str(series_path).strip("/")
					if series_path in in_root:
						candidate = in_root[series_path]
						if isinstance(candidate, zarr.Group) and _get_multiscales(candidate) is not None:
							results.append((candidate, series_path))
				if results:
					return results

	# --- Layout 2: single image at root -------------------------------------
	if _get_multiscales(in_root) is not None:
		return [(in_root, "/")]

	# --- Layout 3: flat numeric children ------------------------------------
	for key in sorted(in_root.keys()):
		candidate = in_root[key]
		if isinstance(candidate, zarr.Group) and _get_multiscales(candidate) is not None:
			results.append((candidate, key))

	return results


def _read_image_data_from_group(group: zarr.Group) -> da.Array:
	"""Read the highest-resolution multiscale level from a group as a Dask array."""
	multiscales = _get_multiscales(group)
	if not multiscales:
		raise ValueError(f"Group has no 'multiscales' metadata: {group.name}")

	datasets = multiscales[0].get("datasets", [])
	if not datasets:
		raise ValueError(f"No datasets listed in multiscales for group: {group.name}")

	dataset_path = datasets[0].get("path", "0") if isinstance(datasets[0], dict) else "0"
	if dataset_path not in group:
		raise ValueError(f"Dataset path '{dataset_path}' not found in group: {group.name}")

	return da.from_zarr(group[dataset_path])


def _open_root(path: str | Path) -> zarr.Group:
	"""Open an OME-Zarr store as a root zarr.Group."""
	path_str = str(path)
	if _HAS_KVIKIO and GDSStore is not None:
		try:
			return zarr.open_group(store=GDSStore(path_str), mode="r")
		except Exception:
			pass
	root = zarr.open_group(path_str, mode="r")
	return root


def _iter_ome_zarr_tczyx(path: str | Path) -> list[tuple[str, da.Array]]:
	"""Iterate all TCZYX FOV images from an OME-Zarr path.

	Supports both standard image nodes and outputs produced by process_zarr_fovs
	(e.g. `fovs/fov_XXXX` groups).

	The original implementation used Reader(loc)() which returns an empty
	sequence for bioformats2raw stores because the root has no "multiscales"
	attr and Reader does not follow OME/"series" indirection.  This version
	enumerates series groups directly.
	"""
	root = _open_root(path)
	series = _enumerate_series_groups(root)
	if not series:
		raise ValueError(f"No image nodes found in OME-Zarr path: {path}")

	fovs: list[tuple[str, da.Array]] = []
	for group, group_path in series:
		arr = _read_image_data_from_group(group).astype(np.float32)

		if arr.ndim == 5:
			fovs.append((group_path, _to_dask_tczyx(arr)))
		elif arr.ndim == 6:
			for local_idx in range(int(arr.shape[0])):
				fovs.append((
					f"{group_path}/idx_{local_idx:04d}",
					_to_dask_tczyx(arr[local_idx]),
				))
		else:
			raise ValueError(
				f"Expected image data to be 5D or 6D, got shape {arr.shape} at '{group_path}'."
			)

	return fovs


# ---------------------------------------------------------------------------
# Coordinate / metadata helpers
# ---------------------------------------------------------------------------


def _extract_group_global_transform(meta: dict[str, Any] | None) -> dict[str, float]:
	"""Extract stage offsets and pixel sizes for global coordinate mapping."""
	if not isinstance(meta, dict):
		return {
			"stage_x": 0.0,
			"stage_y": 0.0,
			"stage_z": 0.0,
			"pps_x": 1.0,
			"pps_y": 1.0,
			"pps_z": 1.0,
		}

	return {
		"stage_x": float(meta.get("stage_x", 0.0) or 0.0),
		"stage_y": float(meta.get("stage_y", 0.0) or 0.0),
		"stage_z": float(meta.get("stage_z", 0.0) or 0.0),
		"pps_x": float(meta.get("physical_pixel_size_x", 1.0) or 1.0),
		"pps_y": float(meta.get("physical_pixel_size_y", 1.0) or 1.0),
		"pps_z": float(meta.get("physical_pixel_size_z", 1.0) or 1.0),
	}


def _load_group_global_transforms(path: str | Path) -> dict[str, dict[str, float]]:
	"""Load group->global transform mapping from OME-Zarr attrs."""
	root = _open_root(path)
	mapping: dict[str, dict[str, float]] = {}

	tile_metadata = root.attrs.get("tile_metadata", [])
	if isinstance(tile_metadata, list):
		for entry in tile_metadata:
			if not isinstance(entry, dict):
				continue
			group_name = entry.get("ngff_group")
			if not isinstance(group_name, str):
				continue
			mapping[group_name.strip("/") or "/"] = _extract_group_global_transform(entry)

	processed_groups = root.attrs.get("processed_fov_groups", [])
	if isinstance(processed_groups, list):
		for entry in processed_groups:
			if not isinstance(entry, dict):
				continue
			group_name = entry.get("output_group")
			if not isinstance(group_name, str):
				continue
			source_meta = entry.get("source_tile_metadata")
			mapping[group_name.strip("/") or "/"] = _extract_group_global_transform(source_meta)

	return mapping


def _load_experiment_map(experiment_json: str | Path | list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Load experiment mapping records with keys: readout, round, channel."""
	if isinstance(experiment_json, list):
		records = experiment_json
	else:
		with open(experiment_json, "r", encoding="utf-8") as handle:
			parsed = json.load(handle)
		if isinstance(parsed, dict):
			if "mappings" in parsed:
				mappings = parsed.get("mappings")
				if not isinstance(mappings, list):
					raise ValueError("Expected 'mappings' to be a list of mapping records.")
				records = mappings
			else:
				records = [parsed]
		else:
			records = parsed

	required = {"readout", "round", "channel"}
	for rec in records:
		if not isinstance(rec, dict):
			raise ValueError(f"Each experiment record must be a dict, found: {type(rec)!r}")
		if not required.issubset(rec):
			raise ValueError(f"Each experiment record must contain keys {required}, found: {rec}")
	return records


def _infer_negative_controls(genes: list[str]) -> set[str]:
	"""Infer blank/negative control gene names from common naming patterns."""
	negatives: set[str] = set()
	for gene in genes:
		low = gene.lower()
		if "blank" in low or "neg" in low or "control" in low:
			negatives.add(gene)
	return negatives



# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------


def _normalization_tmp_dir() -> Path | None:
	value = os.environ.get("BARRACUDA_NORMALIZATION_TMPDIR")
	if value is None or value.strip() == "":
		return None
	path = Path(value)
	path.mkdir(parents=True, exist_ok=True)
	return path


def _linear_percentile_partition(values: np.ndarray, q: float) -> float:
	"""Compute NumPy-compatible linear percentile by in-place partition."""
	n_values = int(values.shape[0])
	if n_values <= 0:
		return 0.0
	if n_values == 1:
		return float(values[0])
	rank = (float(n_values) - 1.0) * (float(q) / 100.0)
	lo = int(np.floor(rank))
	hi = int(np.ceil(rank))
	if lo == hi:
		values.partition(lo)
		return float(values[lo])
	values.partition((lo, hi))
	lower = float(values[lo])
	upper = float(values[hi])
	return lower + (upper - lower) * float(rank - lo)


def _compute_exact_background_scale_for_volumes(
	volumes: list[da.Array],
	eps: float,
) -> tuple[float, float]:
	"""Compute exact per-bit percentiles without keeping the full bit in RAM."""
	if not volumes:
		return 0.0, float(eps)

	total_values = 0
	for volume in volumes:
		vol = da.asarray(volume)
		if vol.ndim != 3:
			raise ValueError(f"Expected ZYX volume for percentile normalization, got shape {vol.shape}")
		total_values += int(np.prod(tuple(int(v) for v in vol.shape), dtype=np.int64))
	if total_values <= 0:
		return 0.0, float(eps)

	tmp_dir = _normalization_tmp_dir()
	tmp = tempfile.NamedTemporaryFile(
		prefix="barracuda_norm_",
		suffix=".float32",
		dir=str(tmp_dir) if tmp_dir is not None else None,
		delete=False,
	)
	tmp_path = Path(tmp.name)
	tmp.close()

	try:
		values = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(total_values,))
		offset = 0
		for volume in volumes:
			vol = da.asarray(volume).astype(np.float32)
			for delayed_block in vol.to_delayed().ravel():
				with dask.config.set(scheduler="single-threaded"):
					block = np.asarray(delayed_block.compute(), dtype=np.float32).ravel()
				if block.size == 0:
					continue
				block = block[np.isfinite(block)]
				if block.size == 0:
					continue
				values[offset:offset + int(block.size)] = block
				offset += int(block.size)
		values.flush()
		del values

		if offset <= 0:
			return 0.0, float(eps)

		values = np.memmap(tmp_path, dtype=np.float32, mode="r+", shape=(offset,))
		bg = _linear_percentile_partition(values, 5.0)
		p95 = _linear_percentile_partition(values, 95.0)
		del values
		return bg, max(float(p95) - float(bg), float(eps))
	finally:
		try:
			tmp_path.unlink()
		except FileNotFoundError:
			pass


def _normalize_bit_indices(
	bit_indices: list[int] | np.ndarray | None,
	bit_count: int,
	context: str,
) -> np.ndarray:
	if bit_indices is None:
		return np.arange(int(bit_count), dtype=np.int64)

	indices = np.asarray(bit_indices, dtype=np.int64).reshape(-1)
	if indices.size == 0:
		raise ValueError(f"{context} must include at least one experiment bit.")
	if np.any(indices < 0) or np.any(indices >= int(bit_count)):
		bad = indices[(indices < 0) | (indices >= int(bit_count))]
		raise ValueError(
			f"{context} contains bit indices outside the valid range [0, {int(bit_count) - 1}]: "
			+ ", ".join(str(int(v)) for v in np.unique(bad))
		)
	return indices


def _batch_size_xyz_to_zyx(batch_size_xyz: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
	if batch_size_xyz is None:
		return None
	values = tuple(int(v) for v in batch_size_xyz)
	if len(values) != 3:
		raise ValueError(f"batch_size_xyz must be a length-3 (x, y, z) tuple, got {batch_size_xyz!r}")
	if any(v <= 0 for v in values):
		raise ValueError(f"batch_size_xyz values must be positive, got {batch_size_xyz!r}")
	x_size, y_size, z_size = values
	return (int(z_size), int(y_size), int(x_size))


def _experiment_bit_indices_from_mapping(
	mapping: list[dict[str, Any]],
	t_dim: int,
	c_dim: int,
) -> np.ndarray:
	"""Return flattened TC bit indices used by the experiment mapping."""
	bit_indices: list[int] = []
	for rec in mapping:
		round_idx = int(rec["round"])
		channel_idx = int(rec["channel"])
		readout = str(rec.get("readout", "<unknown>"))
		if round_idx < 0 or round_idx >= int(t_dim):
			raise ValueError(
				f"Experiment readout {readout!r} references round {round_idx}, "
				f"but image has {int(t_dim)} rounds/timepoints."
			)
		if channel_idx < 0 or channel_idx >= int(c_dim):
			raise ValueError(
				f"Experiment readout {readout!r} references channel {channel_idx}, "
				f"but image has {int(c_dim)} channels."
			)
		bit_indices.append(int(round_idx * int(c_dim) + channel_idx))
	return _normalize_bit_indices(bit_indices, int(t_dim) * int(c_dim), "experiment mapping")


def _selected_bits_to_tczyx(
	selected_bits: da.Array,
	selected_bit_indices: np.ndarray,
	t_dim: int,
	c_dim: int,
	z_dim: int,
	y_dim: int,
	x_dim: int,
) -> da.Array:
	"""Restore selected normalized flat TC bits into a TCZYX array, zeroing unused bits."""
	bit_count = int(t_dim) * int(c_dim)
	all_bits = np.arange(bit_count, dtype=np.int64)
	if selected_bit_indices.shape[0] == bit_count and np.array_equal(selected_bit_indices, all_bits):
		return selected_bits.reshape((int(t_dim), int(c_dim), int(z_dim), int(y_dim), int(x_dim)))

	zero_bit = da.zeros(
		(int(z_dim), int(y_dim), int(x_dim)),
		dtype=selected_bits.dtype,
		chunks=selected_bits.chunks[1:],
	)
	bits_by_index = {
		int(bit_idx): selected_bits[pos]
		for pos, bit_idx in enumerate(selected_bit_indices.tolist())
	}
	flat_bits = [bits_by_index.get(bit_idx, zero_bit) for bit_idx in range(bit_count)]
	return da.stack(flat_bits, axis=0).reshape((int(t_dim), int(c_dim), int(z_dim), int(y_dim), int(x_dim)))


def _normalize_decode_tile_shape(
	shape_zyx: tuple[int, int, int],
	tile_shape_zyx: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
	requested = _DECODE_TILE_SHAPE_ZYX if tile_shape_zyx is None else tuple(int(v) for v in tile_shape_zyx)
	if len(requested) != 3:
		raise ValueError(f"tile_shape_zyx must be a length-3 ZYX tuple, got {tile_shape_zyx!r}")
	out: list[int] = []
	for axis_len, tile_len in zip(shape_zyx, requested, strict=True):
		axis_len_i = max(1, int(axis_len))
		tile_len_i = axis_len_i if int(tile_len) <= 0 else int(tile_len)
		out.append(max(1, min(axis_len_i, tile_len_i)))
	return (int(out[0]), int(out[1]), int(out[2]))


def _normalize_decode_tile_overlap(
	tile_shape_zyx: tuple[int, int, int],
	tile_overlap_zyx: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
	requested = _DECODE_TILE_OVERLAP_ZYX if tile_overlap_zyx is None else tuple(int(v) for v in tile_overlap_zyx)
	if len(requested) != 3:
		raise ValueError(f"tile_overlap_zyx must be a length-3 ZYX tuple, got {tile_overlap_zyx!r}")
	out: list[int] = []
	for tile_len, overlap in zip(tile_shape_zyx, requested, strict=True):
		out.append(max(0, min(max(0, int(tile_len) - 1), int(overlap))))
	return (int(out[0]), int(out[1]), int(out[2]))


def _iter_decode_spatial_tiles(
	shape_zyx: tuple[int, int, int],
	tile_shape_zyx: tuple[int, int, int] | None = None,
	tile_overlap_zyx: tuple[int, int, int] | None = None,
) -> list[dict[str, tuple[int, int, int]]]:
	"""Return core and halo read windows for tile-wise stitched-image decoding."""
	shape = tuple(max(1, int(v)) for v in shape_zyx)
	tile_shape = _normalize_decode_tile_shape(shape, tile_shape_zyx)
	overlap = _normalize_decode_tile_overlap(tile_shape, tile_overlap_zyx)

	tiles: list[dict[str, tuple[int, int, int]]] = []
	for z0 in range(0, shape[0], tile_shape[0]):
		z1 = min(shape[0], z0 + tile_shape[0])
		for y0 in range(0, shape[1], tile_shape[1]):
			y1 = min(shape[1], y0 + tile_shape[1])
			for x0 in range(0, shape[2], tile_shape[2]):
				x1 = min(shape[2], x0 + tile_shape[2])
				read_start = (
					max(0, z0 - overlap[0]),
					max(0, y0 - overlap[1]),
					max(0, x0 - overlap[2]),
				)
				read_stop = (
					min(shape[0], z1 + overlap[0]),
					min(shape[1], y1 + overlap[1]),
					min(shape[2], x1 + overlap[2]),
				)
				tiles.append({
					"core_start": (z0, y0, x0),
					"core_stop": (z1, y1, x1),
					"read_start": read_start,
					"read_stop": read_stop,
				})
	return tiles


def normalize_pixels(
	image: np.ndarray | da.Array,
	eps: float = 1e-8,
	bit_indices: list[int] | np.ndarray | None = None,
	chunk_shape_zyx: tuple[int, int, int] | None = None,
) -> da.Array:
	"""Normalize pixel traces using MERlin-style bit scaling + unit normalization."""
	backgrounds, scale_factors = compute_global_normalization_factors(
		images=[image],
		eps=eps,
		bit_indices=bit_indices,
		chunk_shape_zyx=chunk_shape_zyx,
	)
	return normalize_pixels_with_factors(
		image=image,
		backgrounds=backgrounds,
		scale_factors=scale_factors,
		eps=eps,
		bit_indices=bit_indices,
		chunk_shape_zyx=chunk_shape_zyx,
	)


def compute_global_normalization_factors(
	images: list[np.ndarray | da.Array],
	eps: float = 1e-8,
	bit_indices: list[int] | np.ndarray | None = None,
	chunk_shape_zyx: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Compute exact shared per-bit normalization factors before spatial tiling.

	Each selected experiment bit gets one background and one scale factor, computed
	from all pixels for that bit across all input image groups.  Decode batches then
	reuse these fixed factors; normalization is never recomputed per spatial batch.
	"""
	if not images:
		raise ValueError("Expected at least one image to compute global normalization factors.")

	arrays = [_to_dask_tczyx(image, chunk_shape_zyx=chunk_shape_zyx).astype(np.float32) for image in images]
	t_dim = int(arrays[0].shape[0])
	c_dim = int(arrays[0].shape[1])
	bit_count = int(t_dim * c_dim)
	selected_bit_indices = _normalize_bit_indices(bit_indices, bit_count, "normalization bit_indices")

	for arr in arrays:
		if int(arr.shape[0]) != t_dim or int(arr.shape[1]) != c_dim:
			raise ValueError(
				"All image groups must have the same T/C shape for shared per-bit normalization "
				f"(expected T={t_dim}, C={c_dim}; found T={int(arr.shape[0])}, C={int(arr.shape[1])})."
			)

	backgrounds: list[float] = []
	scales: list[float] = []
	for flat_idx in selected_bit_indices:
		t_idx = int(flat_idx) // c_dim
		c_idx = int(flat_idx) % c_dim
		bg, sf = _compute_exact_background_scale_for_volumes(
			[
				arr[t_idx, c_idx]
				for arr in arrays
			],
			eps=eps,
		)
		backgrounds.append(bg)
		scales.append(sf)

	return (
		np.asarray(backgrounds, dtype=np.float32),
		np.asarray(scales, dtype=np.float32),
	)


def normalize_pixels_with_factors(
	image: np.ndarray | da.Array,
	backgrounds: np.ndarray,
	scale_factors: np.ndarray,
	eps: float = 1e-8,
	bit_indices: list[int] | np.ndarray | None = None,
	chunk_shape_zyx: tuple[int, int, int] | None = None,
) -> da.Array:
	"""Normalize a TCZYX tile using precomputed shared per-bit factors."""
	arr = _to_dask_tczyx(image, chunk_shape_zyx=chunk_shape_zyx).astype(np.float32)
	t_dim, c_dim, z_dim, y_dim, x_dim = arr.shape
	bit_count = int(t_dim * c_dim)
	selected_bit_indices = _normalize_bit_indices(bit_indices, bit_count, "normalization bit_indices")
	selected_bit_count = int(selected_bit_indices.shape[0])

	backgrounds_np = np.asarray(backgrounds, dtype=np.float32).reshape(-1)
	scales_np = np.asarray(scale_factors, dtype=np.float32).reshape(-1)
	if backgrounds_np.shape[0] != selected_bit_count or scales_np.shape[0] != selected_bit_count:
		raise ValueError(
			"Precomputed normalization factors do not match tile bit count "
			f"(expected {selected_bit_count}, got backgrounds={backgrounds_np.shape[0]}, "
			f"scales={scales_np.shape[0]})."
		)

	pixel_traces = arr.reshape((bit_count, z_dim * y_dim * x_dim))[selected_bit_indices]
	backgrounds_da = da.from_array(backgrounds_np, chunks=(selected_bit_count,))
	scale_factors_da = da.from_array(np.maximum(scales_np, float(eps)), chunks=(selected_bit_count,))

	scaled_pixel_traces = (pixel_traces - backgrounds_da[:, None]) / scale_factors_da[:, None]
	pixel_magnitudes = da.sqrt(da.sum(scaled_pixel_traces * scaled_pixel_traces, axis=0))
	zero_mask = pixel_magnitudes <= eps
	pixel_magnitudes_safe = da.where(zero_mask, 1.0, pixel_magnitudes)

	normalized_pixel_traces = scaled_pixel_traces / pixel_magnitudes_safe[None, :]
	normalized_pixel_traces = da.where(zero_mask[None, :], 0.0, normalized_pixel_traces)

	normalized_bits_first = _selected_bits_to_tczyx(
		selected_bits=normalized_pixel_traces.reshape((selected_bit_count, z_dim, y_dim, x_dim)),
		selected_bit_indices=selected_bit_indices,
		t_dim=int(t_dim),
		c_dim=int(c_dim),
		z_dim=int(z_dim),
		y_dim=int(y_dim),
		x_dim=int(x_dim),
	)
	return _to_dask_tczyx(normalized_bits_first.astype(np.float32), chunk_shape_zyx=chunk_shape_zyx)


def map_codebook_to_experiment(
	codebook: pd.DataFrame,
	experiment_json: str | Path | list[dict[str, Any]],
) -> list[dict[str, Any]]:
	"""Map codebook readout columns to experiment channel/timepoint coordinates."""
	experiment = _load_experiment_map(experiment_json)
	experiment_lookup = {str(rec["readout"]): rec for rec in experiment}

	non_readout_columns = {"name", "gene", "id", "transcript_id"}
	readout_columns = [col for col in codebook.columns if col not in non_readout_columns]

	mapping: list[dict[str, Any]] = []
	missing: list[str] = []
	for readout in readout_columns:
		record = experiment_lookup.get(str(readout))
		if record is None:
			missing.append(str(readout))
			continue
		mapping.append(
			{
				"readout": str(readout),
				"round": int(record["round"]),
				"channel": int(record["channel"]),
			}
		)

	if missing:
		raise ValueError("Missing experiment mapping for readouts: " + ", ".join(sorted(missing)))

	return mapping


def infer_gene_dominant_channels(
	codebook: pd.DataFrame,
	mapping: list[dict[str, Any]],
) -> np.ndarray:
	"""Infer a dominant experiment channel for each codebook row/gene."""
	readouts = [str(m["readout"]) for m in mapping]
	channels = np.asarray([int(m["channel"]) for m in mapping], dtype=np.int32)
	if channels.size == 0:
		return np.full((int(codebook.shape[0]),), -1, dtype=np.int32)

	code_values = np.abs(codebook[readouts].to_numpy(dtype=np.float32))
	unique_channels = np.unique(channels)
	channel_scores = np.zeros((code_values.shape[0], unique_channels.size), dtype=np.float32)

	for idx, channel in enumerate(unique_channels):
		mask = channels == channel
		if np.any(mask):
			channel_scores[:, idx] = np.sum(code_values[:, mask], axis=1, dtype=np.float32)

	total_score = np.sum(channel_scores, axis=1)
	dominant_idx = np.argmax(channel_scores, axis=1)
	dominant_channels = unique_channels[dominant_idx].astype(np.int32, copy=False)
	dominant_channels[total_score <= 0.0] = -1
	return dominant_channels


def infer_gene_dominant_rounds(
	codebook: pd.DataFrame,
	mapping: list[dict[str, Any]],
) -> np.ndarray:
	"""Infer a dominant barcode round/timepoint for each codebook row/gene."""
	readouts = [str(m["readout"]) for m in mapping]
	rounds = np.asarray([int(m["round"]) for m in mapping], dtype=np.int32)
	if rounds.size == 0:
		return np.full((int(codebook.shape[0]),), -1, dtype=np.int32)

	code_values = np.abs(codebook[readouts].to_numpy(dtype=np.float32))
	unique_rounds = np.unique(rounds)
	round_scores = np.zeros((code_values.shape[0], unique_rounds.size), dtype=np.float32)

	for idx, round_value in enumerate(unique_rounds):
		mask = rounds == round_value
		if np.any(mask):
			round_scores[:, idx] = np.sum(code_values[:, mask], axis=1, dtype=np.float32)

	total_score = np.sum(round_scores, axis=1)
	dominant_idx = np.argmax(round_scores, axis=1)
	dominant_rounds = unique_rounds[dominant_idx].astype(np.int32, copy=False)
	dominant_rounds[total_score <= 0.0] = -1
	return dominant_rounds


def _decode_metrics_block(
	block: np.ndarray,
	code_vectors: np.ndarray,
	one_bit_threshold: float,
) -> np.ndarray:
	"""Decode one BZYX bit block and return [assigned, euclidean, cosine] ZYX metrics."""
	if block.ndim != 4:
		raise ValueError(f"Expected BZYX block, got shape {block.shape}")

	out_shape = tuple(int(v) for v in block.shape[1:])
	if block.size == 0:
		out = np.empty((3,) + out_shape, dtype=np.float32)
		out[0] = -1.0
		out[1:] = np.nan
		return out

	b_dim = int(block.shape[0])
	code_np = np.asarray(code_vectors, dtype=np.float32)
	chunk_size = max(1, int(_DECODE_PIXELS_PER_BATCH))

	if _HAS_CUCIM and cp is not None:
		block_gpu = pixels = code_gpu = out = None
		try:
			block_gpu = cp.asarray(block, dtype=cp.float32)
			pixels = block_gpu.reshape((b_dim, -1)).T
			code_gpu = cp.asarray(code_np, dtype=cp.float32)
			out = cp.empty((3, int(pixels.shape[0])), dtype=cp.float32)
			out[0].fill(-1.0)
			out[1:].fill(cp.nan)

			if _HAS_CUML and _CuMLNearestNeighbors is not None:
				try:
					nearest = _CuMLNearestNeighbors(n_neighbors=1, metric="euclidean")
					nearest.fit(code_gpu)
					for start in range(0, int(pixels.shape[0]), chunk_size):
						end = min(start + chunk_size, int(pixels.shape[0]))
						distances, indices = nearest.kneighbors(
							pixels[start:end],
							n_neighbors=1,
							return_distance=True,
						)
						chunk_min_dist = _as_cupy_array(distances).reshape((-1,)).astype(cp.float32, copy=False)
						chunk_best_idx = _as_cupy_array(indices).reshape((-1,)).astype(cp.int32, copy=False)
						pass_mask = chunk_min_dist < float(one_bit_threshold)
						out[0, start:end] = cp.where(pass_mask, chunk_best_idx.astype(cp.float32), cp.float32(-1.0))
						out[1, start:end] = chunk_min_dist
						out[2, start:end] = (cp.float32(0.5) * chunk_min_dist * chunk_min_dist).astype(
							cp.float32,
							copy=False,
						)

					out_np = cp.asnumpy(out.reshape((3,) + out_shape))
					return out_np.astype(np.float32, copy=False)
				except Exception:
					out[0].fill(-1.0)
					out[1:].fill(cp.nan)

			for start in range(0, int(pixels.shape[0]), chunk_size):
				end = min(start + chunk_size, int(pixels.shape[0]))
				chunk = pixels[start:end]
				chunk_cosine = cp.matmul(chunk, code_gpu.T)
				chunk_best_idx = cp.argmax(chunk_cosine, axis=1).astype(cp.int32, copy=False)
				chunk_best_cosine = chunk_cosine[cp.arange(chunk_cosine.shape[0]), chunk_best_idx]
				chunk_min_dist = cp.sqrt(cp.clip(2.0 - 2.0 * chunk_best_cosine, a_min=0.0, a_max=None))
				pass_mask = chunk_min_dist < float(one_bit_threshold)
				out[0, start:end] = cp.where(pass_mask, chunk_best_idx.astype(cp.float32), cp.float32(-1.0))
				out[1, start:end] = chunk_min_dist.astype(cp.float32, copy=False)
				out[2, start:end] = (1.0 - chunk_best_cosine).astype(cp.float32, copy=False)

			out_np = cp.asnumpy(out.reshape((3,) + out_shape))
			return out_np.astype(np.float32, copy=False)
		finally:
			del block_gpu, pixels, code_gpu, out
			try:
				cp.get_default_memory_pool().free_all_blocks()
				cp.get_default_pinned_memory_pool().free_all_blocks()
			except Exception:
				pass

	block_np = np.asarray(block, dtype=np.float32)
	pixels_np = block_np.reshape((b_dim, -1)).T
	out_np = np.empty((3, int(pixels_np.shape[0])), dtype=np.float32)
	out_np[0] = -1.0
	out_np[1:] = np.nan

	for start in range(0, int(pixels_np.shape[0]), chunk_size):
		end = min(start + chunk_size, int(pixels_np.shape[0]))
		chunk_np = pixels_np[start:end]
		chunk_cosine_np = np.matmul(chunk_np, code_np.T)
		chunk_best_idx_np = np.argmax(chunk_cosine_np, axis=1).astype(np.int32, copy=False)
		chunk_best_cosine_np = chunk_cosine_np[np.arange(chunk_cosine_np.shape[0]), chunk_best_idx_np]
		chunk_min_dist_np = np.sqrt(np.clip(2.0 - 2.0 * chunk_best_cosine_np, a_min=0.0, a_max=None))
		pass_mask_np = chunk_min_dist_np < float(one_bit_threshold)
		out_np[0, start:end] = np.where(pass_mask_np, chunk_best_idx_np.astype(np.float32), -1.0)
		out_np[1, start:end] = chunk_min_dist_np.astype(np.float32, copy=False)
		out_np[2, start:end] = (1.0 - chunk_best_cosine_np).astype(np.float32, copy=False)

	return out_np.reshape((3,) + out_shape)


def assign_barcodes(
	normalized_image: np.ndarray | da.Array,
	codebook: pd.DataFrame,
	experiment_json: str | Path | list[dict[str, Any]],
	one_bit_threshold: float = 0.5176,
	precomputed_assignment: dict[str, Any] | None = None,
	timing_stats: dict[str, float] | None = None,
) -> tuple[da.Array, da.Array, da.Array, da.Array]:
	"""Assign nearest codebook index to each voxel using Euclidean distance.

	Returns an integer-assignment map where `-1` means unassigned.
	"""
	image = _to_dask_tczyx(normalized_image).astype(np.float32)

	if precomputed_assignment is None:
		mapping = map_codebook_to_experiment(codebook, experiment_json)
		readouts = [m["readout"] for m in mapping]

		for readout in readouts:
			if readout not in codebook.columns:
				raise ValueError(f"Readout {readout} not found in codebook columns")

		code_vectors = codebook[readouts].to_numpy(dtype=np.float32)
		code_norms = np.linalg.norm(code_vectors, axis=1, keepdims=True)
		code_norms = np.maximum(code_norms, 1e-8)
		code_vectors = code_vectors / code_norms

		gene_column = "name" if "name" in codebook.columns else "gene"
		if gene_column not in codebook.columns:
			raise ValueError("Codebook must include a 'name' or 'gene' column for gene labels.")
		gene_arr = codebook[gene_column].astype(str).to_numpy(dtype=object)

		mapping_records = mapping
	else:
		mapping_records = precomputed_assignment["mapping"]
		code_vectors = precomputed_assignment.get("code_vectors")
		if code_vectors is None:
			raise ValueError("precomputed_assignment must include 'code_vectors'.")
		code_vectors = np.asarray(code_vectors, dtype=np.float32)
		gene_arr = precomputed_assignment["gene_arr"]
	_ = gene_arr

	bits = []
	for m in mapping_records:
		bits.append(image[m["round"], m["channel"]])
	bit_stack = da.stack(bits, axis=0).astype(np.float32)

	b_dim, z_dim, y_dim, x_dim = bit_stack.shape
	chunks_zyx = tuple(int(c[0]) for c in bit_stack.chunks[1:])
	bit_stack = bit_stack.rechunk((int(b_dim), chunks_zyx[0], chunks_zyx[1], chunks_zyx[2]))
	chunks_zyx_da = bit_stack.chunks[1:]

	t0 = time.perf_counter()
	block_kwargs = {
		"code_vectors": np.asarray(code_vectors, dtype=np.float32),
		"one_bit_threshold": float(one_bit_threshold),
	}
	metrics_da = da.map_blocks(
		_decode_metrics_block,
		bit_stack,
		dtype=np.float32,
		chunks=((3,),) + chunks_zyx_da,
		drop_axis=0,
		new_axis=0,
		**block_kwargs,
	)
	assigned_da = metrics_da[0].astype(np.int32)
	euclidean_da = metrics_da[1].astype(np.float32)
	cosine_da = metrics_da[2].astype(np.float32)
	intensity_zyx = da.mean(bit_stack, axis=0).astype(np.float32)
	_profile_add(timing_stats, "assign_s", time.perf_counter() - t0)

	return assigned_da, euclidean_da, cosine_da, intensity_zyx


def _extract_spots_from_assignment(
	assigned_barcodes: np.ndarray | da.Array,
	min_size: int,
	min_intensity: float,
	euclidean_distance: np.ndarray | da.Array | None = None,
	cosine_distance: np.ndarray | da.Array | None = None,
	intensity_image: np.ndarray | da.Array | None = None,
	gene_names: list[str] | np.ndarray | None = None,
	gene_channels: list[int] | np.ndarray | None = None,
	gene_rounds: list[int] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
	"""Group connected assigned voxels into spots and summarize per-spot metrics."""
	assigned_arr = da.asarray(assigned_barcodes)
	if assigned_arr.ndim != 3:
		raise ValueError(f"Expected assigned_barcodes shape ZYX (ndim=3), got {assigned_arr.shape}")

	structure = ndi.generate_binary_structure(rank=3, connectivity=1)
	spots: list[dict[str, Any]] = []

	if gene_names is not None:
		gene_lookup = np.asarray(gene_names, dtype=object)
	else:
		gene_lookup = None

	if gene_channels is not None:
		channel_lookup = np.asarray(gene_channels, dtype=np.int32)
	else:
		channel_lookup = None

	if gene_rounds is not None:
		round_lookup = np.asarray(gene_rounds, dtype=np.int32)
	else:
		round_lookup = None

	def _gene_name_from_label(g: int) -> str:
		if gene_lookup is None:
			return str(g)
		if g < 0 or g >= int(gene_lookup.shape[0]):
			return str(g)
		return str(gene_lookup[g])

	def _channel_from_label(g: int) -> int | None:
		if channel_lookup is None:
			return None
		if g < 0 or g >= int(channel_lookup.shape[0]):
			return None
		channel = int(channel_lookup[g])
		return channel if channel >= 0 else None

	def _round_from_label(g: int) -> int | None:
		if round_lookup is None:
			return None
		if g < 0 or g >= int(round_lookup.shape[0]):
			return None
		round_value = int(round_lookup[g])
		return round_value if round_value >= 0 else None

	def _as_numpy_dask(arr_like: np.ndarray | da.Array, dtype: np.dtype | None = None) -> da.Array:
		arr_da = da.asarray(arr_like)
		target_dtype = np.dtype(dtype) if dtype is not None else np.dtype(arr_da.dtype)
		return arr_da.map_blocks(
			lambda block: cp.asnumpy(block) if (_HAS_CUCIM and cp is not None and isinstance(block, cp.ndarray)) else np.asarray(block),
			dtype=target_dtype,
			meta=np.empty((0,), dtype=target_dtype),
		)

	def _compute_numpy_many(items: list[tuple[np.ndarray | da.Array, np.dtype | None]]) -> list[np.ndarray]:
		lazy = [_as_numpy_dask(arr_like, dtype=dtype) for arr_like, dtype in items]
		with dask.config.set(scheduler="single-threaded"):
			computed = dask.compute(*lazy)
		return [
			np.asarray(value, dtype=np.dtype(dtype) if dtype is not None else np.asarray(value).dtype)
			for value, (_, dtype) in zip(computed, items, strict=True)
		]

	gpu_full_volume_failed = False

	assigned_dtype = np.dtype(np.int32) if assigned_arr.dtype.kind in ("i", "u", "f") else None
	compute_items: list[tuple[np.ndarray | da.Array, np.dtype | None]] = [(assigned_arr, assigned_dtype)]
	if euclidean_distance is not None:
		compute_items.append((euclidean_distance, np.dtype(np.float32)))
	if cosine_distance is not None:
		compute_items.append((cosine_distance, np.dtype(np.float32)))
	if intensity_image is not None:
		compute_items.append((intensity_image, np.dtype(np.float32)))

	computed_items = _compute_numpy_many(compute_items)
	item_idx = 0
	labels_arr = computed_items[item_idx]
	item_idx += 1
	if labels_arr.ndim != 3:
		raise ValueError(f"Expected assigned_barcodes shape ZYX (ndim=3), got {labels_arr.shape}")

	euclidean_np = None
	cosine_np = None
	intensity_np = None
	if euclidean_distance is not None:
		euclidean_np = computed_items[item_idx]
		item_idx += 1
	if cosine_distance is not None:
		cosine_np = computed_items[item_idx]
		item_idx += 1
	if intensity_image is not None:
		intensity_np = computed_items[item_idx]
		item_idx += 1

	if euclidean_np is not None and euclidean_np.shape != labels_arr.shape:
		raise ValueError("euclidean_distance must match assigned_barcodes shape")
	if cosine_np is not None and cosine_np.shape != labels_arr.shape:
		raise ValueError("cosine_distance must match assigned_barcodes shape")
	if intensity_np is not None and intensity_np.shape != labels_arr.shape:
		raise ValueError("intensity_image must match assigned_barcodes shape")

	is_integer_labels = labels_arr.dtype.kind in ("i", "u")
	if is_integer_labels:
		valid_mask = labels_arr >= 0
		if not np.any(valid_mask):
			return spots

		for gene_id in np.unique(labels_arr[valid_mask]).astype(np.int32, copy=False).tolist():
			gene_mask = labels_arr == int(gene_id)
			coords = np.where(gene_mask)
			if coords[0].size == 0:
				continue

			z0, z1 = int(coords[0].min()), int(coords[0].max()) + 1
			y0, y1 = int(coords[1].min()), int(coords[1].max()) + 1
			x0, x1 = int(coords[2].min()), int(coords[2].max()) + 1
			mask = gene_mask[z0:z1, y0:y1, x0:x1]

			can_label_gene_gpu = (
				_HAS_CUCIM
				and cp is not None
				and not gpu_full_volume_failed
				and _label_gpu is not None
				and _gpu_volume_fits(
					shape=tuple(int(v) for v in mask.shape),
					dtype=np.uint8,
					expansion_factor=10.0,
					vram_fraction=0.20,
				)
			)
			if can_label_gene_gpu:
				mask_gpu = comp_labels_gpu = None
				try:
					mask_gpu = cp.asarray(mask.astype(np.uint8, copy=False))
					comp_labels_gpu, n_components = _label_gpu(mask_gpu, connectivity=1, return_num=True)
					comp_labels = cp.asnumpy(comp_labels_gpu)
				except Exception:
					comp_labels, n_components = ndi.label(mask, structure=structure)
				finally:
					del mask_gpu, comp_labels_gpu
			else:
				comp_labels, n_components = ndi.label(mask, structure=structure)

			if n_components <= 0:
				continue

			component_ids = np.arange(1, int(n_components) + 1, dtype=np.int32)
			sizes = np.bincount(comp_labels.ravel(), minlength=int(n_components) + 1)[1:].astype(np.int64)
			keep_size_mask = sizes >= int(min_size)
			if not np.any(keep_size_mask):
				continue

			centers = np.asarray(
				ndi.center_of_mass(np.ones(mask.shape, dtype=np.uint8), labels=comp_labels, index=component_ids),
				dtype=np.float64,
			)
			centers[:, 0] += z0
			centers[:, 1] += y0
			centers[:, 2] += x0

			if intensity_np is not None:
				intensity_sub = intensity_np[z0:z1, y0:y1, x0:x1]
				mean_intensities = np.asarray(
					ndi.mean(intensity_sub, labels=comp_labels, index=component_ids), dtype=np.float64,
				)
			else:
				mean_intensities = sizes.astype(np.float64, copy=False)

			if euclidean_np is not None:
				euclidean_sub = euclidean_np[z0:z1, y0:y1, x0:x1]
				mean_euclidean = np.asarray(
					ndi.mean(euclidean_sub, labels=comp_labels, index=component_ids), dtype=np.float64,
				)
			else:
				mean_euclidean = np.full((int(n_components),), np.nan, dtype=np.float64)

			if cosine_np is not None:
				cosine_sub = cosine_np[z0:z1, y0:y1, x0:x1]
				mean_cosine = np.asarray(
					ndi.mean(cosine_sub, labels=comp_labels, index=component_ids), dtype=np.float64,
				)
			else:
				mean_cosine = np.full((int(n_components),), np.nan, dtype=np.float64)

			keep_mask = keep_size_mask & (mean_intensities >= float(min_intensity))
			for idx in np.where(keep_mask)[0]:
				zc, yc, xc = centers[idx]
				spots.append({
					"gene": _gene_name_from_label(int(gene_id)),
					"round": _round_from_label(int(gene_id)),
					"channel": _channel_from_label(int(gene_id)),
					"z": float(zc),
					"y": float(yc),
					"x": float(xc),
					"size": int(sizes[idx]),
					"euclidean_distance": float(mean_euclidean[idx]),
					"cosine_distance": float(mean_cosine[idx]),
					"mean_intensity": float(mean_intensities[idx]),
				})

		return spots

	labels_arr = labels_arr.astype(object, copy=False)
	unique_labels = [g for g in np.unique(labels_arr) if g != "None"]

	def _gene_name_from_object_label(g: Any) -> str:
		return str(g)

	for gene in unique_labels:
		coords = np.where(labels_arr == gene)
		if coords[0].size == 0:
			continue

		z0, z1 = int(coords[0].min()), int(coords[0].max()) + 1
		y0, y1 = int(coords[1].min()), int(coords[1].max()) + 1
		x0, x1 = int(coords[2].min()), int(coords[2].max()) + 1

		mask = (labels_arr[z0:z1, y0:y1, x0:x1] == gene)

		can_label_object_gpu = (
			_HAS_CUCIM
			and cp is not None
			and _label_gpu is not None
			and _gpu_volume_fits(
				shape=tuple(int(v) for v in mask.shape),
				dtype=np.uint8,
				expansion_factor=10.0,
				vram_fraction=0.20,
			)
		)
		if can_label_object_gpu:
			mask_gpu = comp_labels_gpu = None
			try:
				mask_gpu = cp.asarray(mask.astype(np.uint8, copy=False))
				comp_labels_gpu, n_components = _label_gpu(mask_gpu, connectivity=1, return_num=True)
				comp_labels = cp.asnumpy(comp_labels_gpu)
			except Exception:
				comp_labels, n_components = ndi.label(mask, structure=structure)
			finally:
				del mask_gpu, comp_labels_gpu
		else:
			comp_labels, n_components = ndi.label(mask, structure=structure)

		if n_components <= 0:
			continue

		component_ids = np.arange(1, int(n_components) + 1, dtype=np.int32)
		sizes = np.bincount(comp_labels.ravel(), minlength=int(n_components) + 1)[1:].astype(np.int64)
		keep_size_mask = sizes >= int(min_size)
		if not np.any(keep_size_mask):
			continue

		centers = np.asarray(
			ndi.center_of_mass(np.ones(mask.shape, dtype=np.uint8), labels=comp_labels, index=component_ids),
			dtype=np.float64,
		)
		centers[:, 0] += z0
		centers[:, 1] += y0
		centers[:, 2] += x0

		if intensity_np is not None:
			intensity_sub = intensity_np[z0:z1, y0:y1, x0:x1]
			mean_intensities = np.asarray(
				ndi.mean(intensity_sub, labels=comp_labels, index=component_ids), dtype=np.float64,
			)
		else:
			mean_intensities = sizes.astype(np.float64, copy=False)

		if euclidean_np is not None:
			euclidean_sub = euclidean_np[z0:z1, y0:y1, x0:x1]
			mean_euclidean = np.asarray(
				ndi.mean(euclidean_sub, labels=comp_labels, index=component_ids), dtype=np.float64,
			)
		else:
			mean_euclidean = np.full((int(n_components),), np.nan, dtype=np.float64)

		if cosine_np is not None:
			cosine_sub = cosine_np[z0:z1, y0:y1, x0:x1]
			mean_cosine = np.asarray(
				ndi.mean(cosine_sub, labels=comp_labels, index=component_ids), dtype=np.float64,
			)
		else:
			mean_cosine = np.full((int(n_components),), np.nan, dtype=np.float64)

		keep_mask = keep_size_mask & (mean_intensities >= float(min_intensity))
		keep_indices = np.where(keep_mask)[0]

		for idx in keep_indices:
			zc, yc, xc = centers[idx]
			spots.append({
				"gene": _gene_name_from_object_label(gene),
				"round": None,
				"channel": None,
				"z": float(zc),
				"y": float(yc),
				"x": float(xc),
				"size": int(sizes[idx]),
				"euclidean_distance": float(mean_euclidean[idx]),
				"cosine_distance": float(mean_cosine[idx]),
				"mean_intensity": float(mean_intensities[idx]),
			})

	return spots


def filter_spots_fdr(
	spots: list[dict[str, Any]],
	negative_control_genes: set[str] | list[str],
	fdr_threshold: float,
) -> list[dict[str, Any]]:
	"""Filter spots using blank-gene-based FDR on cosine distance."""
	if not spots:
		return []

	negatives = set(negative_control_genes)
	n_total = len(spots)
	n_blank = sum(1 for s in spots if s["gene"] in negatives)
	if n_total == 0 or n_blank == 0:
		return spots

	cosine_dist_vals = np.asarray([float(s.get("cosine_distance", np.nan)) for s in spots], dtype=float)
	cosine_dist_vals = cosine_dist_vals[~np.isnan(cosine_dist_vals)]
	if cosine_dist_vals.size == 0:
		return spots

	thresholds = np.unique(cosine_dist_vals)
	thresholds.sort()

	chosen_t = None
	for t in thresholds:
		n_total_lt = sum(1 for s in spots if float(s.get("cosine_distance", np.inf)) < t)
		if n_total_lt == 0:
			continue
		n_blank_lt = sum(
			1 for s in spots
			if s["gene"] in negatives and float(s.get("cosine_distance", np.inf)) < t
		)
		fdr_t = (n_blank_lt * (n_blank / n_total)) / n_total_lt
		if fdr_t <= fdr_threshold:
			chosen_t = float(t)

	if chosen_t is None:
		return []

	return [s for s in spots if float(s.get("cosine_distance", np.inf)) < chosen_t]


def merge_spots_spatial(
	spots: list[dict[str, Any]],
	radius_um: float = 0.75,
	z_scale: float = 1.0,
	across_rounds_only: bool = False,  # kept for API compatibility; ignored
) -> list[dict[str, Any]]:
	"""Suppress overlapping spot calls in physical coordinates.

	Nearby spots with different gene calls are merged within ``radius_um`` by
	keeping the higher-intensity call. Exact coordinate duplicates are still
	collapsed regardless of gene, matching the previous de-duplication behavior.
	"""
	_ = across_rounds_only
	if not spots:
		return list(spots)

	radius = max(0.0, float(radius_um))
	coords = np.asarray(
		[
			(
				float(s["x"]),
				float(s["y"]),
				float(s["z"]) * float(z_scale),
			)
			for s in spots
		],
		dtype=np.float64,
	)
	finite_mask = np.isfinite(coords).all(axis=1)
	finite_indices = np.where(finite_mask)[0]

	def _priority_key(idx: int) -> tuple[float, float, int]:
		mean_intensity = float(np.nan_to_num(float(spots[idx].get("mean_intensity", 0.0)), nan=-np.inf))
		cosine_distance = float(np.nan_to_num(float(spots[idx].get("cosine_distance", np.inf)), nan=np.inf))
		return mean_intensity, -cosine_distance, -int(idx)

	def _same_coord(i: int, j: int) -> bool:
		return bool(np.array_equal(coords[i], coords[j]))

	def _different_gene(i: int, j: int) -> bool:
		return str(spots[i].get("gene", "")) != str(spots[j].get("gene", ""))

	neighbor_map: dict[int, list[int]] = {i: [] for i in range(len(spots))}
	if finite_indices.size > 0:
		finite_coords = coords[finite_indices]
		from scipy.spatial import cKDTree

		tree = cKDTree(finite_coords)
		neighbor_rows = tree.query_ball_point(finite_coords, r=radius)

		for local_i, local_neighbors in enumerate(neighbor_rows):
			idx = int(finite_indices[local_i])
			neighbor_map[idx] = [
				int(finite_indices[int(local_j)])
				for local_j in local_neighbors
			]

	keep_indices: list[int] = []
	suppressed: set[int] = set()
	for idx in sorted(range(len(spots)), key=_priority_key, reverse=True):
		if idx in suppressed:
			continue

		keep_indices.append(idx)
		for neighbor_idx in neighbor_map.get(idx, []):
			if neighbor_idx == idx or neighbor_idx in suppressed:
				continue
			if _same_coord(idx, neighbor_idx) or _different_gene(idx, neighbor_idx):
				suppressed.add(int(neighbor_idx))

	keep_set = set(keep_indices)
	return [
		spots[i]
		for i in range(len(spots))
		if i in keep_set
	]


def filter_spots_intensity(
	spots: list[dict[str, Any]],
	output_plot_path: str | Path | None = None,
	s: float = 1.0,
) -> tuple[list[dict[str, Any]], float]:
	"""Filter spots by intensity using a rank-intensity inflection threshold."""
	if not spots:
		return [], float("nan")

	intensities = np.asarray([float(s.get("mean_intensity", 0.0)) for s in spots], dtype=float)
	threshold = _infer_intensity_cutoff(intensities, s=float(s))

	filtered = [s for s in spots if float(s.get("mean_intensity", 0.0)) >= threshold]

	if output_plot_path is not None:
		_write_intensity_rank_plot(
			intensities=intensities,
			threshold=threshold,
			output_plot_path=output_plot_path,
		)

	return filtered, threshold


def filter_spots_intensity_per_channel(
	spots: list[dict[str, Any]],
	output_plot_path: str | Path | None = None,
	s: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[int, float]]:
	"""Filter spots by intensity with separate knee thresholds per channel."""
	if not spots:
		return [], {}

	channels = np.asarray(
		[
			int(s.get("channel")) if s.get("channel") is not None else -1
			for s in spots
		],
		dtype=np.int32,
	)
	intensities = np.asarray([float(s.get("mean_intensity", 0.0)) for s in spots], dtype=float)

	keep_mask = np.zeros((len(spots),), dtype=bool)
	thresholds: dict[int, float] = {}

	for channel in np.unique(channels):
		idx = np.where(channels == channel)[0]
		if idx.size == 0:
			continue
		channel_intensities = intensities[idx]
		threshold = _infer_intensity_cutoff(channel_intensities, s=s)
		thresholds[int(channel)] = float(threshold)
		keep_mask[idx] = channel_intensities >= threshold

		if output_plot_path is not None:
			out_base = Path(output_plot_path)
			if len(np.unique(channels)) > 1:
				channel_label = f"ch{int(channel)}" if int(channel) >= 0 else "ch_unknown"
				out_path = out_base.with_name(f"{out_base.stem}_{channel_label}{out_base.suffix}")
			else:
				out_path = out_base
			_write_intensity_rank_plot(
				intensities=channel_intensities,
				threshold=threshold,
				output_plot_path=out_path,
			)

	filtered = [spots[idx] for idx in np.where(keep_mask)[0]]
	return filtered, thresholds


def _write_intensity_rank_plot(
	intensities: np.ndarray,
	threshold: float,
	output_plot_path: str | Path,
) -> None:
	"""Write rank-intensity plot with an intensity threshold line."""
	import matplotlib.pyplot as plt

	out = Path(output_plot_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	sorted_i = np.sort(np.asarray(intensities, dtype=float))[::-1]

	plt.figure(figsize=(6, 4))
	plt.plot(np.arange(1, sorted_i.size + 1), sorted_i)
	plt.axhline(float(threshold), color="red", linestyle="--")
	plt.xlabel("Rank")
	plt.ylabel("Mean Intensity")
	plt.tight_layout()
	plt.savefig(out, dpi=200)
	plt.close()


def _infer_intensity_cutoff(intensities: np.ndarray, s: float = 1.0) -> float:
	"""Infer intensity cutoff from rank-ordered intensity curvature."""
	if intensities.size == 0:
		return float("nan")
	sorted_i = np.sort(intensities)[::-1]
	if sorted_i.size < 3:
		return float(np.min(sorted_i))

	from kneed import KneeLocator

	ranks = np.arange(1, sorted_i.size + 1, dtype=float)
	locator = KneeLocator(ranks, sorted_i, curve="concave", direction="decreasing", S=float(s))
	if locator.knee is not None:
		knee_idx = int(max(0, min(sorted_i.size - 1, int(locator.knee) - 1)))
		return float(sorted_i[knee_idx])

	# Fallback if no knee is detected for near-linear rank curves.
	second_derivative = np.gradient(np.gradient(sorted_i))
	inflection_idx = int(np.argmin(second_derivative))
	return float(sorted_i[inflection_idx])


def save_spot_qc_pdf(spots: list[dict[str, Any]], output_pdf_path: str | Path) -> Path:
	"""Save QC plots for spot intensity and size distributions as a multi-page PDF."""
	from matplotlib.backends.backend_pdf import PdfPages
	import matplotlib.pyplot as plt
	plt.rcParams["font.family"] = "sans-serif"
	plt.rcParams["font.sans-serif"] = ["Helvetica"] + plt.rcParams["font.sans-serif"]

	out = Path(output_pdf_path)
	out.parent.mkdir(parents=True, exist_ok=True)

	intensities = np.asarray([float(s.get("mean_intensity", 0.0)) for s in spots], dtype=float)
	sizes = np.asarray([int(s.get("size", 0)) for s in spots], dtype=float)
	threshold = _infer_intensity_cutoff(intensities)
	sorted_i = np.sort(intensities)[::-1] if intensities.size > 0 else np.asarray([], dtype=float)

	with PdfPages(out) as pdf:
		fig1 = plt.figure(figsize=(7, 5))
		if intensities.size > 0:
			plt.hist(intensities, bins=100)
			if np.isfinite(threshold):
				plt.axvline(threshold, color="red", linestyle="--", label=f"cutoff={threshold:.2f}")
				plt.legend(loc="best")
		else:
			plt.text(0.5, 0.5, "No spots", ha="center", va="center")
		plt.xlabel("Mean Intensity")
		plt.ylabel("Count")
		plt.title("Spot Intensity Histogram")
		plt.tight_layout()
		pdf.savefig(fig1)
		plt.close(fig1)

		fig2 = plt.figure(figsize=(7, 5))
		if sizes.size > 0:
			bins = min(100, max(10, int(np.nanmax(sizes))))
			plt.hist(sizes, bins=bins)
		else:
			plt.text(0.5, 0.5, "No spots", ha="center", va="center")
		plt.xlabel("Spot Size (voxels)")
		plt.ylabel("Count")
		plt.title("Spot Size Histogram")
		plt.tight_layout()
		pdf.savefig(fig2)
		plt.close(fig2)

		fig3 = plt.figure(figsize=(7, 5))
		if sorted_i.size > 0:
			plt.plot(np.arange(1, sorted_i.size + 1), sorted_i)
			if np.isfinite(threshold):
				plt.axhline(threshold, color="red", linestyle="--", label=f"cutoff={threshold:.2f}")
				plt.legend(loc="best")
		else:
			plt.text(0.5, 0.5, "No spots", ha="center", va="center")
		plt.xlabel("Rank")
		plt.ylabel("Mean Intensity")
		plt.title("Rank Plot of Spot Intensities")
		plt.tight_layout()
		pdf.savefig(fig3)
		plt.close(fig3)

	return out


def save_spots(
	spots: list[dict[str, Any]],
	output_path: str | Path,
	pixel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
	"""Save spot list to CSV with origin rebased to lower-left at z=0 (µm)."""
	out = Path(output_path)
	out.parent.mkdir(parents=True, exist_ok=True)

	z_um, y_um, x_um = pixel_size_um
	if not spots:
		return _write_records_csv([], columns=["gene", "x", "y", "z"], output_path=out)

	# Rebase output coordinates so the lower-left corner on the z=0 plane
	# is the origin (0, 0, 0). If z=0 is absent, use the minimum-z plane.
	z_vals = np.asarray([float(s["z"]) * z_um for s in spots], dtype=np.float64)
	plane_z = 0.0 if np.any(np.isclose(z_vals, 0.0, atol=1e-6)) else float(np.min(z_vals))
	plane_mask = np.isclose(z_vals, plane_z, atol=1e-6)

	x_vals = np.asarray([float(s["x"]) * x_um for s in spots], dtype=np.float64)
	y_vals = np.asarray([float(s["y"]) * y_um for s in spots], dtype=np.float64)
	origin_x = float(np.min(x_vals[plane_mask])) if np.any(plane_mask) else float(np.min(x_vals))
	origin_y = float(np.min(y_vals[plane_mask])) if np.any(plane_mask) else float(np.min(y_vals))
	origin_z = plane_z

	records = []
	for idx, s in enumerate(spots):
		records.append({
			"gene": s["gene"],
			"x": float(x_vals[idx] - origin_x),
			"y": float(y_vals[idx] - origin_y),
			"z": float(z_vals[idx] - origin_z),
		})

	return _write_records_csv(records, columns=["gene", "x", "y", "z"], output_path=out)


def decode_images(
	image_folder: str | Path,
	codebook_path: str | Path,
	output_folder: str | Path,
	min_size: int,
	min_intensity: float,
	experiment_json: str | Path | list[dict[str, Any]],
	one_bit_threshold: float = 0.5176,
	fdr_threshold: float = 0.05,
	negative_control_genes: set[str] | list[str] | None = None,
	overlap_radius_um: float = 0.5,
	overlap_z_scale: float = 1.0,
	timing: bool = False,
	intensity_knee_s: float = 1.0,
	intensity_threshold_by_channel: bool = True,
	tile_shape_zyx: tuple[int, int, int] | None = None,
	tile_overlap_zyx: tuple[int, int, int] | None = None,
	batch_size_xyz: tuple[int, int, int] | None = None,
) -> list[dict[str, Any]]:
	"""Decode OME-Zarr image groups and save spot calls with spatial overlap merging."""
	t_pipeline = time.perf_counter()
	timing_stats: dict[str, float] | None = {} if timing else None
	image_root = Path(image_folder)
	out_root = Path(output_folder)
	out_root.mkdir(parents=True, exist_ok=True)
	batch_shape_zyx = _batch_size_xyz_to_zyx(batch_size_xyz)
	if batch_shape_zyx is not None and tile_shape_zyx is not None:
		raise ValueError("Specify either batch_size_xyz=(x, y, z) or tile_shape_zyx=(z, y, x), not both.")
	decode_tile_shape_zyx = batch_shape_zyx or tile_shape_zyx

	codebook = _read_codebook_csv(codebook_path)
	if negative_control_genes is None:
		gene_col = "name" if "name" in codebook.columns else "gene"
		negative_control_genes = _infer_negative_controls(codebook[gene_col].astype(str).tolist())

	if image_root.exists() and image_root.suffix == ".zarr":
		image_paths = [image_root]
	else:
		image_paths = sorted(image_root.glob("*.zarr"))
	if not image_paths:
		raise ValueError(f"No .zarr images found in: {image_root}")

	mapping = map_codebook_to_experiment(codebook, experiment_json)
	readouts = [m["readout"] for m in mapping]
	for readout in readouts:
		if readout not in codebook.columns:
			raise ValueError(f"Readout {readout} not found in codebook columns")

	code_vectors = codebook[readouts].to_numpy(dtype=np.float32)
	code_norms = np.linalg.norm(code_vectors, axis=1, keepdims=True)
	code_norms = np.maximum(code_norms, 1e-8)
	code_vectors = code_vectors / code_norms

	gene_column = "name" if "name" in codebook.columns else "gene"
	if gene_column not in codebook.columns:
		raise ValueError("Codebook must include a 'name' or 'gene' column for gene labels.")

	assignment_cache = {
		"mapping": mapping,
		"code_vectors": code_vectors,
		"gene_arr": codebook[gene_column].astype(str).to_numpy(dtype=object),
	}
	gene_names = assignment_cache["gene_arr"].tolist()
	gene_channels = infer_gene_dominant_channels(codebook=codebook, mapping=mapping)
	gene_rounds = infer_gene_dominant_rounds(codebook=codebook, mapping=mapping)

	tile_records: list[dict[str, Any]] = []
	for image_path in image_paths:
		t_read = time.perf_counter()
		group_transforms = _load_group_global_transforms(image_path)
		image_groups = _iter_ome_zarr_tczyx(image_path)
		_profile_add(timing_stats, "read_s", time.perf_counter() - t_read)
		for image_group, image in image_groups:
			tile_records.append({
				"image_path": image_path,
				"group_transforms": group_transforms,
				"image_group": image_group,
				"image": image,
				"tile_index": len(tile_records),
			})

	t_norm_global = time.perf_counter()
	first_image = tile_records[0]["image"]
	first_t_dim = int(first_image.shape[0])
	first_c_dim = int(first_image.shape[1])
	for rec in tile_records:
		image = rec["image"]
		if int(image.shape[0]) != first_t_dim or int(image.shape[1]) != first_c_dim:
			raise ValueError(
				"All image groups must have the same T/C shape for shared experiment-channel normalization "
				f"(expected T={first_t_dim}, C={first_c_dim}; found T={int(image.shape[0])}, "
				f"C={int(image.shape[1])} in {rec['image_path']}::{rec['image_group']})."
			)
	experiment_bit_indices = _experiment_bit_indices_from_mapping(
		mapping=mapping,
		t_dim=first_t_dim,
		c_dim=first_c_dim,
	)
	global_backgrounds, global_scales = compute_global_normalization_factors(
		images=[rec["image"] for rec in tile_records],
		bit_indices=experiment_bit_indices,
		chunk_shape_zyx=batch_shape_zyx,
	)
	_profile_add(timing_stats, "normalize_s", time.perf_counter() - t_norm_global)

	all_spots: list[dict[str, Any]] = []
	for rec in tile_records:
		image_path = rec["image_path"]
		group_transforms = rec["group_transforms"]
		image_group = rec["image_group"]
		image = rec["image"]

		transform = group_transforms.get(image_group, group_transforms.get("/", {
			"stage_x": 0.0,
			"stage_y": 0.0,
			"stage_z": 0.0,
			"pps_x": 1.0,
			"pps_y": 1.0,
			"pps_z": 1.0,
		}))
		spatial_shape = (int(image.shape[2]), int(image.shape[3]), int(image.shape[4]))
		spatial_tiles = _iter_decode_spatial_tiles(
			shape_zyx=spatial_shape,
			tile_shape_zyx=decode_tile_shape_zyx,
			tile_overlap_zyx=tile_overlap_zyx,
		)

		for decode_tile_index, tile in enumerate(spatial_tiles):
			read_start = tile["read_start"]
			read_stop = tile["read_stop"]
			core_start = tile["core_start"]
			core_stop = tile["core_stop"]
			z0, y0, x0 = read_start
			z1, y1, x1 = read_stop

			image_tile = image[:, :, z0:z1, y0:y1, x0:x1]
			normalized = normalize_pixels_with_factors(
				image=image_tile,
				backgrounds=global_backgrounds,
				scale_factors=global_scales,
				bit_indices=experiment_bit_indices,
				chunk_shape_zyx=batch_shape_zyx,
			)

			t_assign = time.perf_counter()
			assigned, euclidean_distance_map, cosine_distance_map, intensity_map = assign_barcodes(
				normalized_image=normalized,
				codebook=codebook,
				experiment_json=experiment_json,
				one_bit_threshold=one_bit_threshold,
				precomputed_assignment=assignment_cache,
				timing_stats=timing_stats,
			)
			_profile_add(timing_stats, "assign_s", time.perf_counter() - t_assign)

			t_combine = time.perf_counter()
			spots = _extract_spots_from_assignment(
				assigned_barcodes=assigned,
				min_size=min_size,
				min_intensity=min_intensity,
				euclidean_distance=euclidean_distance_map,
				cosine_distance=cosine_distance_map,
				intensity_image=intensity_map,
				gene_names=gene_names,
				gene_channels=gene_channels,
				gene_rounds=gene_rounds,
			)
			_profile_add(timing_stats, "combine_s", time.perf_counter() - t_combine)

			for s in spots:
				z_local = float(s["z"]) + float(read_start[0])
				y_local = float(s["y"]) + float(read_start[1])
				x_local = float(s["x"]) + float(read_start[2])
				if not (
					float(core_start[0]) <= z_local < float(core_stop[0])
					and float(core_start[1]) <= y_local < float(core_stop[1])
					and float(core_start[2]) <= x_local < float(core_stop[2])
				):
					continue

				s["x_local_px"] = x_local
				s["y_local_px"] = y_local
				s["z_local_px"] = z_local

				s["x"] = x_local * transform["pps_x"] + transform["stage_x"]
				s["y"] = y_local * transform["pps_y"] + transform["stage_y"]
				s["z"] = z_local * transform["pps_z"] + transform["stage_z"]
				s["tile_index"] = int(rec["tile_index"])
				s["tile_path"] = str(image_path)
				s["tile_group"] = str(image_group)
				s["decode_tile_index"] = int(decode_tile_index)
				all_spots.append(s)

	t_post = time.perf_counter()
	negative_controls = set(negative_control_genes)
	if len(negative_controls) == 0:
		filtered = all_spots
	else:
		filtered = filter_spots_fdr(all_spots, negative_control_genes=negative_controls, fdr_threshold=fdr_threshold)

	if intensity_threshold_by_channel:
		filtered, _ = filter_spots_intensity_per_channel(filtered, s=float(intensity_knee_s))
	else:
		filtered, _ = filter_spots_intensity(filtered, s=float(intensity_knee_s))

	resolved = merge_spots_spatial(
		filtered,
		radius_um=overlap_radius_um,
		z_scale=overlap_z_scale,
	)

	save_spots(resolved, out_root / "decoded_spots.csv")
	save_spot_qc_pdf(resolved, out_root / "decoded_spots_qc.pdf")
	_profile_add(timing_stats, "merge_filter_save_s", time.perf_counter() - t_post)
	_profile_add(timing_stats, "total_s", time.perf_counter() - t_pipeline)
	if timing and timing_stats is not None:
		_print_profile_summary(timing_stats)

	return resolved


try:
	from .tracing import instrument_module_functions
except ImportError:  # pragma: no cover - supports direct module execution
	from tracing import instrument_module_functions

instrument_module_functions(globals())
