"""Run slice-wise Cellpose-SAM plus uSegment3D on stitched OME-NGFF images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dask.array as da
import geopandas as gpd
import numpy as np
import zarr
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from skimage import io, measure, segmentation

try:
	from kvikio.zarr import GDSStore
	_HAS_KVIKIO = True
except Exception:
	GDSStore = None
	_HAS_KVIKIO = False


@dataclass(frozen=True)
class _YxTile:
	core_y0: int
	core_y1: int
	core_x0: int
	core_x1: int
	read_y0: int
	read_y1: int
	read_x0: int
	read_x1: int

	@property
	def core_local_y0(self) -> int:
		return self.core_y0 - self.read_y0

	@property
	def core_local_y1(self) -> int:
		return self.core_y1 - self.read_y0

	@property
	def core_local_x0(self) -> int:
		return self.core_x0 - self.read_x0

	@property
	def core_local_x1(self) -> int:
		return self.core_x1 - self.read_x0


def _get_multiscales(group: zarr.Group) -> list | None:
	ms = group.attrs.get("multiscales")
	if isinstance(ms, list) and ms:
		return ms
	ome = group.attrs.get("ome")
	if isinstance(ome, dict):
		ms = ome.get("multiscales")
		if isinstance(ms, list) and ms:
			return ms
	return None


def _open_root(path: str | Path) -> zarr.Group:
	path_str = str(path)
	if _HAS_KVIKIO and GDSStore is not None:
		try:
			return zarr.open_group(store=GDSStore(path_str), mode="r")
		except Exception:
			pass
	return zarr.open_group(path_str, mode="r")


def _enumerate_series_groups(in_root: zarr.Group) -> list[tuple[zarr.Group, str]]:
	results: list[tuple[zarr.Group, str]] = []

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

	if _get_multiscales(in_root) is not None:
		return [(in_root, "/")]

	for key in sorted(in_root.keys()):
		candidate = in_root[key]
		if isinstance(candidate, zarr.Group) and _get_multiscales(candidate) is not None:
			results.append((candidate, key))

	return results


def _read_image_data_from_group(group: zarr.Group) -> da.Array:
	multiscales = _get_multiscales(group)
	if not multiscales:
		raise ValueError(f"Group has no multiscales metadata: {group.name}")

	datasets = multiscales[0].get("datasets", [])
	if not datasets:
		raise ValueError(f"No datasets listed in multiscales for group: {group.name}")

	dataset_path = datasets[0].get("path", "0") if isinstance(datasets[0], dict) else "0"
	if dataset_path not in group:
		raise ValueError(f"Dataset path '{dataset_path}' not found in group: {group.name}")

	return da.from_zarr(group[dataset_path])


def _extract_scale_transform(transformations: Any) -> list[float] | None:
	if not isinstance(transformations, list):
		return None
	for transform in transformations:
		if not isinstance(transform, dict):
			continue
		if str(transform.get("type", "")).lower() != "scale":
			continue
		scale = transform.get("scale")
		if not isinstance(scale, (list, tuple)):
			continue
		values: list[float] = []
		valid = True
		for item in scale:
			try:
				values.append(float(item))
			except (TypeError, ValueError):
				valid = False
				break
		if valid and values:
			return values
	return None


def _extract_axis_names(multiscales_entry: dict[str, Any]) -> list[str]:
	raw_axes = multiscales_entry.get("axes", [])
	axis_names: list[str] = []
	if isinstance(raw_axes, list):
		for axis in raw_axes:
			if isinstance(axis, dict) and isinstance(axis.get("name"), str):
				axis_names.append(axis["name"].lower())
			elif isinstance(axis, str):
				axis_names.append(axis.lower())
	return axis_names


def _extract_voxel_sizes_zyx_from_group(group: zarr.Group) -> dict[str, float]:
	"""Read Z/Y/X voxel sizes from OME-NGFF multiscales metadata."""
	multiscales = _get_multiscales(group)
	default_sizes = {"z": 1.0, "y": 1.0, "x": 1.0}
	if not multiscales:
		return default_sizes

	ms_entry = multiscales[0] if isinstance(multiscales[0], dict) else {}
	axis_names = _extract_axis_names(ms_entry)
	datasets = ms_entry.get("datasets", []) if isinstance(ms_entry, dict) else []
	dataset0 = datasets[0] if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict) else {}

	scales = _extract_scale_transform(dataset0.get("coordinateTransformations"))
	if scales is None:
		scales = _extract_scale_transform(ms_entry.get("coordinateTransformations"))
	if scales is None:
		return default_sizes

	if axis_names and len(axis_names) == len(scales):
		axis_to_scale = dict(zip(axis_names, scales, strict=False))
		z_size = float(axis_to_scale.get("z", default_sizes["z"]))
		y_size = float(axis_to_scale.get("y", default_sizes["y"]))
		x_size = float(axis_to_scale.get("x", default_sizes["x"]))
		if z_size > 0 and y_size > 0 and x_size > 0:
			return {"z": z_size, "y": y_size, "x": x_size}

	# Common fallback for TCZYX datasets when axes are missing.
	if len(scales) >= 5:
		z_size = float(scales[-3])
		y_size = float(scales[-2])
		x_size = float(scales[-1])
		if z_size > 0 and y_size > 0 and x_size > 0:
			return {"z": z_size, "y": y_size, "x": x_size}

	return default_sizes


def _extract_group_global_transform(meta: dict[str, Any] | None) -> dict[str, float]:
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


def _extract_zyx_volume(arr: da.Array, t_index: int, c_index: int) -> np.ndarray:
	if arr.ndim != 5:
		raise ValueError(f"Expected TCZYX array, got shape {arr.shape}")
	if t_index < 0 or t_index >= int(arr.shape[0]):
		raise ValueError(f"time index {t_index} out of bounds for T={arr.shape[0]}")
	if c_index < 0 or c_index >= int(arr.shape[1]):
		raise ValueError(f"channel index {c_index} out of bounds for C={arr.shape[1]}")
	return np.asarray(arr[t_index, c_index].compute(), dtype=np.float32)


def _validate_tczyx_indices(arr: da.Array, t_index: int, c_index: int) -> None:
	if arr.ndim != 5:
		raise ValueError(f"Expected TCZYX array, got shape {arr.shape}")
	if t_index < 0 or t_index >= int(arr.shape[0]):
		raise ValueError(f"time index {t_index} out of bounds for T={arr.shape[0]}")
	if c_index < 0 or c_index >= int(arr.shape[1]):
		raise ValueError(f"channel index {c_index} out of bounds for C={arr.shape[1]}")


def _normalize_yx_pair(value: int | tuple[int, int] | list[int], name: str, *, allow_zero: bool = False) -> tuple[int, int]:
	if isinstance(value, int):
		y_value = x_value = value
	elif isinstance(value, (tuple, list)) and len(value) == 2:
		y_value = int(value[0])
		x_value = int(value[1])
	else:
		raise TypeError(f"{name} must be an int or a 2-item YX tuple/list")

	min_value = 0 if allow_zero else 1
	if y_value < min_value or x_value < min_value:
		comparator = "non-negative" if allow_zero else "positive"
		raise ValueError(f"{name} values must be {comparator}; got {(y_value, x_value)}")
	return y_value, x_value


def _iter_yx_tiles(
	shape_yx: tuple[int, int],
	tile_size_yx: int | tuple[int, int] | list[int],
	tile_overlap_yx: int | tuple[int, int] | list[int],
) -> list[_YxTile]:
	y_dim, x_dim = int(shape_yx[0]), int(shape_yx[1])
	if y_dim <= 0 or x_dim <= 0:
		raise ValueError(f"Invalid YX image shape: {shape_yx}")

	tile_y, tile_x = _normalize_yx_pair(tile_size_yx, "tile_size_yx")
	overlap_y, overlap_x = _normalize_yx_pair(tile_overlap_yx, "tile_overlap_yx", allow_zero=True)
	if overlap_y >= tile_y or overlap_x >= tile_x:
		raise ValueError(
			"tile_overlap_yx must be smaller than tile_size_yx on both axes; "
			f"got overlap={(overlap_y, overlap_x)}, tile_size={(tile_y, tile_x)}"
		)

	tiles: list[_YxTile] = []
	for core_y0 in range(0, y_dim, tile_y):
		core_y1 = min(core_y0 + tile_y, y_dim)
		for core_x0 in range(0, x_dim, tile_x):
			core_x1 = min(core_x0 + tile_x, x_dim)
			tiles.append(
				_YxTile(
					core_y0=core_y0,
					core_y1=core_y1,
					core_x0=core_x0,
					core_x1=core_x1,
					read_y0=max(0, core_y0 - overlap_y),
					read_y1=min(y_dim, core_y1 + overlap_y),
					read_x0=max(0, core_x0 - overlap_x),
					read_x1=min(x_dim, core_x1 + overlap_x),
				)
			)
	return tiles


def _extract_zyx_tile(arr: da.Array, t_index: int, c_index: int, tile: _YxTile) -> np.ndarray:
	_validate_tczyx_indices(arr, t_index=t_index, c_index=c_index)
	return np.asarray(
		arr[
			t_index,
			c_index,
			:,
			tile.read_y0:tile.read_y1,
			tile.read_x0:tile.read_x1,
		].compute(),
		dtype=np.float32,
	)


def _tile_transform(base_transform: dict[str, float], tile: _YxTile) -> dict[str, float]:
	transform = dict(base_transform)
	transform["stage_x"] = float(transform["stage_x"]) + float(tile.read_x0) * float(transform["pps_x"])
	transform["stage_y"] = float(transform["stage_y"]) + float(tile.read_y0) * float(transform["pps_y"])
	return transform


def _mask_centroid_in_tile_core(mask_zyx: np.ndarray, tile: _YxTile) -> bool:
	coords = np.argwhere(mask_zyx)
	if coords.size == 0:
		return False
	centroid_y = float(coords[:, 1].mean())
	centroid_x = float(coords[:, 2].mean())
	return (
		tile.core_local_y0 <= centroid_y < tile.core_local_y1
		and tile.core_local_x0 <= centroid_x < tile.core_local_x1
	)


def _extract_cell_probability_2d(flows: Any, fallback_shape: tuple[int, int]) -> np.ndarray:
	if isinstance(flows, (tuple, list)):
		if len(flows) >= 3:
			arr = np.asarray(flows[2])
			if arr.shape == fallback_shape:
				return arr.astype(np.float32, copy=False)
			squeezed = np.squeeze(arr)
			if squeezed.shape == fallback_shape:
				return squeezed.astype(np.float32, copy=False)
		for item in flows:
			arr = np.asarray(item)
			if arr.shape == fallback_shape:
				return arr.astype(np.float32, copy=False)
	return np.zeros(fallback_shape, dtype=np.float32)


def _normalize_slice_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
	arr = np.asarray(slice_2d, dtype=np.float32)
	if arr.size == 0:
		return np.zeros(arr.shape, dtype=np.uint8)

	lo = float(np.percentile(arr, 1.0))
	hi = float(np.percentile(arr, 99.0))
	if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
		lo = float(np.min(arr))
		hi = float(np.max(arr))
	if hi <= lo:
		return np.zeros(arr.shape, dtype=np.uint8)

	scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
	return np.asarray(scaled * 255.0, dtype=np.uint8)


def _build_outline_overlay(slice_2d: np.ndarray, labels_2d: np.ndarray) -> np.ndarray:
	base = _normalize_slice_to_uint8(slice_2d)
	overlay = np.stack([base, base, base], axis=-1)
	boundaries = segmentation.find_boundaries(labels_2d, mode="outer")
	overlay[boundaries, 0] = 255
	overlay[boundaries, 1] = 40
	overlay[boundaries, 2] = 40
	return overlay


def _save_overlay_pngs(
	volume_zyx: np.ndarray,
	masks_zyx: np.ndarray,
	group_path: str,
	output_dir: Path,
) -> list[Path]:
	if volume_zyx.shape != masks_zyx.shape:
		raise ValueError(
			f"Overlay input shape mismatch: volume={volume_zyx.shape}, masks={masks_zyx.shape}"
		)

	group_label = "root" if group_path.strip("/") == "" else group_path.strip("/").replace("/", "_")
	group_dir = output_dir / f"group_{group_label}"
	group_dir.mkdir(parents=True, exist_ok=True)

	written_paths: list[Path] = []
	for z_idx in range(int(volume_zyx.shape[0])):
		overlay = _build_outline_overlay(volume_zyx[z_idx], masks_zyx[z_idx])
		out_path = group_dir / f"z_{z_idx:04d}_nuclei_outline.png"
		io.imsave(str(out_path), overlay, check_contrast=False)
		written_paths.append(out_path)

	return written_paths


def _link_slice_instances_across_z(
	labels_zyx: np.ndarray,
	min_overlap_px: int,
	min_iou: float,
) -> np.ndarray:
	"""Link 2D instance masks across adjacent z-slices into 3D object IDs."""
	if labels_zyx.ndim != 3:
		raise ValueError(f"Expected ZYX labels, got shape {labels_zyx.shape}")

	z_dim = int(labels_zyx.shape[0])
	node_parent: list[int] = [0]
	next_node_id = 1
	slice_node_maps: list[dict[int, int]] = []

	def _make_node() -> int:
		nonlocal next_node_id
		node_id = next_node_id
		next_node_id += 1
		node_parent.append(node_id)
		return node_id

	def _find(node: int) -> int:
		while node_parent[node] != node:
			node_parent[node] = node_parent[node_parent[node]]
			node = node_parent[node]
		return node

	def _union(a: int, b: int) -> None:
		ra = _find(a)
		rb = _find(b)
		if ra != rb:
			node_parent[rb] = ra

	for z_idx in range(z_dim):
		plane = labels_zyx[z_idx]
		local_ids = np.unique(plane)
		local_ids = local_ids[local_ids > 0]
		curr_map: dict[int, int] = {}
		for local_id in local_ids:
			curr_map[int(local_id)] = _make_node()
		slice_node_maps.append(curr_map)

		if z_idx == 0:
			continue

		prev_plane = labels_zyx[z_idx - 1]
		prev_map = slice_node_maps[z_idx - 1]
		if not prev_map or not curr_map:
			continue

		prev_areas = np.bincount(prev_plane.ravel())
		curr_areas = np.bincount(plane.ravel())

		for curr_local, curr_node in curr_map.items():
			curr_mask = plane == curr_local
			overlaps = prev_plane[curr_mask]
			if overlaps.size == 0:
				continue

			overlap_ids, overlap_counts = np.unique(overlaps[overlaps > 0], return_counts=True)
			if overlap_ids.size == 0:
				continue

			curr_area = int(curr_areas[curr_local]) if curr_local < curr_areas.size else int(curr_mask.sum())
			for prev_local, inter in zip(overlap_ids, overlap_counts):
				inter_px = int(inter)
				if inter_px < int(min_overlap_px):
					continue

				prev_local_int = int(prev_local)
				if prev_local_int not in prev_map:
					continue
				prev_area = int(prev_areas[prev_local_int]) if prev_local_int < prev_areas.size else int((prev_plane == prev_local_int).sum())
				union_px = curr_area + prev_area - inter_px
				if union_px <= 0:
					continue
				iou = float(inter_px) / float(union_px)
				if iou >= float(min_iou):
					_union(curr_node, prev_map[prev_local_int])

	if next_node_id == 1:
		return np.zeros_like(labels_zyx, dtype=np.int32)

	root_to_final: dict[int, int] = {}
	next_final_id = 1
	merged = np.zeros_like(labels_zyx, dtype=np.int32)

	for z_idx in range(z_dim):
		plane = labels_zyx[z_idx]
		curr_map = slice_node_maps[z_idx]
		if not curr_map:
			continue
		for local_id, node in curr_map.items():
			root = _find(node)
			final_id = root_to_final.get(root)
			if final_id is None:
				final_id = next_final_id
				root_to_final[root] = final_id
				next_final_id += 1
			merged[z_idx][plane == local_id] = final_id

	return merged


def _run_cellpose_sam_z_slices(
	model: Any,
	volume_zyx: np.ndarray,
	diameter: float | None,
	flow_threshold: float,
	cellprob_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
	"""Run Cellpose-SAM independently on every Z plane of a ZYX volume."""
	if volume_zyx.ndim != 3:
		raise ValueError(f"Expected ZYX volume, got shape {volume_zyx.shape}")

	labels_zyx = np.zeros(volume_zyx.shape, dtype=np.int32)
	cellprob_zyx = np.zeros(volume_zyx.shape, dtype=np.float32)

	for z_idx in range(int(volume_zyx.shape[0])):
		slice_yx = np.asarray(volume_zyx[z_idx], dtype=np.float32)
		masks_yx, flows, _styles = model.eval(
			slice_yx,
			do_3D=False,
			channel_axis=None,
			z_axis=None,
			channels=[0, 0],
			diameter=diameter,
			flow_threshold=flow_threshold,
			cellprob_threshold=cellprob_threshold,
		)
		masks_yx = np.asarray(masks_yx, dtype=np.int32)
		if masks_yx.shape != slice_yx.shape:
			raise ValueError(
				f"Cellpose-SAM mask shape {masks_yx.shape} does not match z={z_idx} input {slice_yx.shape}"
			)
		labels_zyx[z_idx] = masks_yx
		cellprob_zyx[z_idx] = _extract_cell_probability_2d(flows, fallback_shape=slice_yx.shape)

	return labels_zyx, cellprob_zyx


def _run_usegment3d_from_xy_labels(
	labels_zyx: np.ndarray,
	min_mask_voxels: int,
	dtform_method: str,
	n_cpu: int | None,
) -> np.ndarray:
	"""Aggregate ZYX slice labels into 3D objects with uSegment3D indirect mode.

	Coordinate convention: the input and output are ZYX label volumes. The only
	orthoview supplied to uSegment3D is the XY view, i.e. one 2D Cellpose-SAM
	mask image for each z-level. XZ/YZ orthoviews are passed as empty inputs.
	"""
	if labels_zyx.ndim != 3:
		raise ValueError(f"Expected ZYX labels, got shape {labels_zyx.shape}")
	if not np.any(labels_zyx > 0):
		return np.zeros_like(labels_zyx, dtype=np.int32)

	import segment3D.parameters as usegment3d_params
	import segment3D.usegment3d as usegment3d

	params = usegment3d_params.get_2D_to_3D_aggregation_params()
	params["indirect_method"]["dtform_method"] = str(dtform_method)
	params["indirect_method"]["n_cpu"] = n_cpu
	params["postprocess_binary"]["remove_small_objects"] = max(0, int(min_mask_voxels))
	params["connected_component"]["min_area"] = max(1, min(5, int(min_mask_voxels)))

	masks_zyx, _intermediates = usegment3d.aggregate_2D_to_3D_segmentation_indirect_method(
		segmentations=[np.asarray(labels_zyx, dtype=np.int32), [], []],
		params=params,
		img_xy_shape=labels_zyx.shape,
		precomputed_binary=None,
		savefolder=None,
		basename=None,
	)
	masks_zyx = np.asarray(masks_zyx, dtype=np.int32)
	if masks_zyx.shape != labels_zyx.shape:
		raise ValueError(
			f"uSegment3D mask shape {masks_zyx.shape} does not match input label shape {labels_zyx.shape}"
		)
	return masks_zyx


def _mask_to_footprint(mask_zyx: np.ndarray, transform: dict[str, float]) -> Polygon | MultiPolygon | None:
	xy_mask = np.any(mask_zyx, axis=0)
	contours = measure.find_contours(xy_mask.astype(np.uint8), level=0.5)
	if not contours:
		return None

	polygons: list[Polygon] = []
	for contour in contours:
		if contour.shape[0] < 3:
			continue
		coords = []
		for row, col in contour:
			x_um = float(col) * transform["pps_x"] + transform["stage_x"]
			y_um = float(row) * transform["pps_y"] + transform["stage_y"]
			coords.append((x_um, y_um))
		poly = Polygon(coords)
		if poly.is_empty:
			continue
		if not poly.is_valid:
			poly = poly.buffer(0)
		if poly.is_empty:
			continue
		polygons.append(poly)

	if not polygons:
		return None
	return unary_union(polygons)


def _rebase_origin(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
	if gdf.empty:
		return gdf

	plane_mask = (gdf["z_min_um"] <= 0.0) & (gdf["z_max_um"] >= 0.0)
	if plane_mask.any():
		origin_z = 0.0
		subset = gdf.loc[plane_mask]
	else:
		origin_z = float(gdf["z_min_um"].min())
		subset = gdf.loc[np.isclose(gdf["z_min_um"], origin_z)]

	min_bounds = subset.geometry.bounds[["minx", "miny"]]
	origin_x = float(min_bounds["minx"].min())
	origin_y = float(min_bounds["miny"].min())

	shifted = gdf.copy()
	shifted["geometry"] = shifted["geometry"].map(lambda geom: affinity.translate(geom, xoff=-origin_x, yoff=-origin_y))
	shifted["z_min_um"] = shifted["z_min_um"] - origin_z
	shifted["z_max_um"] = shifted["z_max_um"] - origin_z
	return shifted


def _write_geodataframe_parquet(gdf: gpd.GeoDataFrame, output_path: str | Path) -> Path:
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	if output.exists():
		output.unlink()

	gdf.to_parquet(output)
	return output


def run_cellpose_on_ngff(
	ngff_path: str | Path,
	output_parquet_path: str | Path,
	channel_index: int,
	time_index: int,
	model_type: str = "cpsam",
	diameter: float | None = None,
	flow_threshold: float = 0.4,
	cellprob_threshold: float = 0.0,
	use_gpu: bool = True,
	min_mask_voxels: int = 20,
	overlay_output_dir: str | Path | None = None,
	usegment3d_dtform_method: str = "cellpose_improve",
	usegment3d_n_cpu: int | None = None,
	tile_size_yx: int | tuple[int, int] | list[int] | None = 2048,
	tile_overlap_yx: int | tuple[int, int] | list[int] = 128,
	write_overlays: bool = False,
) -> gpd.GeoDataFrame:
	"""Run tiled Cellpose-SAM on each z-level, then aggregate masks with uSegment3D.

	Input image convention is TCZYX in OME-NGFF and ZYX after extraction.
	Cellpose-SAM is run independently on each YX z-slice within each XY tile.
	uSegment3D receives each tile's ZYX label stack as the XY orthoview and
	returns a ZYX 3D instance-label volume. Tile coordinates are converted back
	to global micrometer coordinates, then rebased so the lower-left corner at
	z=0 is (0, 0, 0).
	"""
	from cellpose import models

	root = _open_root(ngff_path)
	series = _enumerate_series_groups(root)
	if not series:
		raise ValueError(f"No image series found in OME-NGFF path: {ngff_path}")

	model = models.CellposeModel(gpu=bool(use_gpu), pretrained_model=model_type)
	transforms = _load_group_global_transforms(ngff_path)
	records: list[dict[str, Any]] = []
	mask_counter = 1
	save_overlays = bool(write_overlays or overlay_output_dir is not None)
	overlay_dir: Path | None = None
	if save_overlays:
		overlay_dir = (
			Path(overlay_output_dir)
			if overlay_output_dir is not None
			else Path(output_parquet_path).parent / f"{Path(output_parquet_path).stem}_overlays"
		)
		overlay_dir.mkdir(parents=True, exist_ok=True)

	group, group_path = series[0]
	arr = _read_image_data_from_group(group)
	_validate_tczyx_indices(arr, t_index=time_index, c_index=channel_index)
	y_dim, x_dim = (int(arr.shape[-2]), int(arr.shape[-1]))
	if tile_size_yx is None:
		tiles = [
			_YxTile(
				core_y0=0,
				core_y1=y_dim,
				core_x0=0,
				core_x1=x_dim,
				read_y0=0,
				read_y1=y_dim,
				read_x0=0,
				read_x1=x_dim,
			)
		]
	else:
		tiles = _iter_yx_tiles(
			shape_yx=(y_dim, x_dim),
			tile_size_yx=tile_size_yx,
			tile_overlap_yx=tile_overlap_yx,
		)

	transform = transforms.get(group_path.strip("/") or "/", {
		"stage_x": 0.0,
		"stage_y": 0.0,
		"stage_z": 0.0,
		"pps_x": 1.0,
		"pps_y": 1.0,
		"pps_z": 1.0,
	})

	for tile_idx, tile in enumerate(tiles):
		volume_zyx = _extract_zyx_tile(arr, t_index=time_index, c_index=channel_index, tile=tile)
		tile_transform = _tile_transform(transform, tile)
		labels_zyx, cellprob = _run_cellpose_sam_z_slices(
			model=model,
			volume_zyx=volume_zyx,
			diameter=diameter,
			flow_threshold=flow_threshold,
			cellprob_threshold=cellprob_threshold,
		)
		masks = _run_usegment3d_from_xy_labels(
			labels_zyx=labels_zyx,
			min_mask_voxels=min_mask_voxels,
			dtform_method=usegment3d_dtform_method,
			n_cpu=usegment3d_n_cpu,
		)

		masks = np.asarray(masks, dtype=np.int32)
		if masks.shape != volume_zyx.shape:
			raise ValueError(
				f"uSegment3D mask shape {masks.shape} does not match tile input shape {volume_zyx.shape} "
				f"for {group_path} tile {tile_idx}"
			)
		if overlay_dir is not None:
			tile_overlay_group = (
				f"{group_path.strip('/') or 'root'}/"
				f"tile_y{tile.core_y0}_{tile.core_y1}_x{tile.core_x0}_{tile.core_x1}"
			)
			_save_overlay_pngs(volume_zyx, masks, group_path=tile_overlay_group, output_dir=overlay_dir)

		for label_id in np.unique(masks):
			if label_id == 0:
				continue
			obj_mask = masks == label_id
			voxel_count = int(obj_mask.sum())
			if voxel_count < int(min_mask_voxels):
				continue
			if not _mask_centroid_in_tile_core(obj_mask, tile):
				continue

			footprint = _mask_to_footprint(obj_mask, tile_transform)
			if footprint is None or footprint.is_empty:
				continue

			z_coords = np.where(np.any(obj_mask, axis=(1, 2)))[0]
			z_min_um = float(z_coords.min()) * tile_transform["pps_z"] + tile_transform["stage_z"]
			z_max_um = float(z_coords.max()) * tile_transform["pps_z"] + tile_transform["stage_z"]
			# Score is the mean per-slice Cellpose-SAM cell probability over the
			# final uSegment3D object footprint.
			score = float(np.mean(cellprob[obj_mask])) if voxel_count > 0 else float("-inf")

			records.append({
				"mask_id": mask_counter,
				"score": score,
				"voxel_count": voxel_count,
				"z_min_um": z_min_um,
				"z_max_um": z_max_um,
				"tile_group": f"tile_{tile_idx:05d}",
				"tile_core_y0": tile.core_y0,
				"tile_core_y1": tile.core_y1,
				"tile_core_x0": tile.core_x0,
				"tile_core_x1": tile.core_x1,
				"geometry": footprint,
			})
			mask_counter += 1

	if records:
		gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=None)
	else:
		# Keep a valid empty GeoDataFrame schema so downstream calls do not fail.
		gdf = gpd.GeoDataFrame(
			{
				"mask_id": [],
				"score": [],
				"voxel_count": [],
				"z_min_um": [],
				"z_max_um": [],
				"tile_group": [],
				"tile_core_y0": [],
				"tile_core_y1": [],
				"tile_core_x0": [],
				"tile_core_x1": [],
				"geometry": [],
			},
			geometry="geometry",
			crs=None,
		)
	gdf = _rebase_origin(gdf)
	_write_geodataframe_parquet(gdf, output_parquet_path)
	return gdf


try:
	from .tracing import instrument_module_functions
except ImportError:  # pragma: no cover - supports direct module execution
	from tracing import instrument_module_functions

instrument_module_functions(globals())
