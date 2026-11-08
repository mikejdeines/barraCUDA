"""Convert microscope output data to OME-NGFF (OME-Zarr)."""

from __future__ import annotations

import shutil
import re
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import tifffile
import zarr
from bioio import BioImage
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

try:
    from .array_utils import (
        cast_array_for_storage,
        storage_dtype_from_env,
    )
    from .kvikio_zarr import write_dask_array_to_zarr
except ImportError:  # pragma: no cover - supports direct module execution
    from array_utils import (
        cast_array_for_storage,
        storage_dtype_from_env,
    )
    from kvikio_zarr import write_dask_array_to_zarr

try:
    from kvikio.zarr import GDSStore
    _HAS_KVIKIO = True
except Exception:
    GDSStore = None
    _HAS_KVIKIO = False


# ---------------------------------------------------------------------------
# Axis-spec helpers
# ---------------------------------------------------------------------------

_AXIS_TYPES: dict[str, str] = {
    "t": "time",
    "c": "channel",
    "z": "space",
    "y": "space",
    "x": "space",
    "m": "space",
}

_FALLBACK_CONVERSION_CHUNK_SHAPE: tuple[int, int, int] = (18, 2048, 2048)
_STORAGE_DTYPE = storage_dtype_from_env()


def _axes_dicts(axes: str) -> list[dict[str, str]]:
    """Convert an axis-order string to the list-of-dicts form required by ome-zarr.

    e.g. "tczyx" -> [{"name": "t", "type": "time"}, {"name": "c", "type": "channel"}, ...]

    Passing bare characters to write_image works in old ome-zarr releases but
    newer versions require dicts so that Reader can interpret the multiscales
    metadata correctly.
    """
    return [{"name": ax, "type": _AXIS_TYPES.get(ax, "space")} for ax in axes.lower()]


def _bounded_spatial_chunk_shape(
    shape_zyx: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Return a Z/Y/X chunk shape capped by the array bounds."""
    return (
        max(1, min(int(shape_zyx[0]), int(_FALLBACK_CONVERSION_CHUNK_SHAPE[0]))),
        max(1, min(int(shape_zyx[1]), int(_FALLBACK_CONVERSION_CHUNK_SHAPE[1]))),
        max(1, min(int(shape_zyx[2]), int(_FALLBACK_CONVERSION_CHUNK_SHAPE[2]))),
    )


# ---------------------------------------------------------------------------
# Path / store helpers
# ---------------------------------------------------------------------------


def _prepare_output_path(output_path: str | Path) -> Path:
    """Normalize output path and ensure OME-Zarr-compatible suffix."""
    output = Path(output_path)
    if output.suffix != ".zarr":
        output = output.with_suffix(".zarr")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _reset_existing_output(output_path: str | Path) -> Path:
    """Remove an existing OME-Zarr output path so writes start from a clean store."""
    output = _prepare_output_path(output_path)
    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    return output


def _open_group_with_optional_kvikio(path: str | Path, mode: str) -> zarr.Group:
    """Open a zarr group using KvikIO GDSStore when available, else parse_url."""
    path_str = str(path)
    if _HAS_KVIKIO and GDSStore is not None:
        try:
            return zarr.open_group(store=GDSStore(path_str), mode=mode)
        except Exception:
            pass

    loc = parse_url(path_str, mode=mode)
    if loc is None:
        raise ValueError(f"Could not open OME-Zarr path: {path_str}")
    return zarr.open_group(store=loc.store, mode=mode)


# ---------------------------------------------------------------------------
# Core writer
# ---------------------------------------------------------------------------


def _write_ngff_array(
    image: np.ndarray | da.Array,
    output_path: str | Path,
    axes: str,
    group_path: str | None = None,
    first_write: bool = True,
    storage_dtype: np.dtype | None = None,
) -> None:
    """Write an array to an OME-Zarr group using ome-zarr with Dask arrays."""
    output = _prepare_output_path(output_path)
    root_group = _open_group_with_optional_kvikio(
        output,
        mode="w" if first_write else "a",
    )

    group = root_group
    if group_path:
        group = root_group.require_group(group_path)

    target_dtype = np.dtype(storage_dtype if storage_dtype is not None else _STORAGE_DTYPE)
    image_da = cast_array_for_storage(da.asarray(image), target_dtype)
    if image_da.ndim == 5:
        chunk_z, chunk_y, chunk_x = _bounded_spatial_chunk_shape(
            (int(image_da.shape[2]), int(image_da.shape[3]), int(image_da.shape[4])),
        )
        image_da = image_da.rechunk((1, 1, chunk_z, chunk_y, chunk_x))
    elif image_da.ndim == 3:
        chunk_z, chunk_y, chunk_x = _bounded_spatial_chunk_shape(
            (int(image_da.shape[0]), int(image_da.shape[1]), int(image_da.shape[2])),
        )
        image_da = image_da.rechunk((chunk_z, chunk_y, chunk_x))
    elif image_da.ndim == 2:
        image_da = image_da.rechunk((image_da.shape[0], image_da.shape[1]))

    component = "0" if not group_path else f"{str(group_path).strip('/')}/0"
    if write_dask_array_to_zarr(
        image_da,
        root_path=output,
        component=component,
        overwrite=True,
        num_gpus_per_task=0.0,
    ):
        group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "axes": _axes_dicts(axes),
                "datasets": [{"path": "0"}],
            }
        ]
        return

    # BUG FIX: pass axes as list-of-dicts, not bare characters.
    write_image(
        image=image_da,
        group=group,
        axes=_axes_dicts(axes),
    )


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _extract_scene_tile_info(
    bio_image: BioImage,
    scene_index: int,
    scene_name: str,
) -> dict[str, Any]:
    """Collect best-effort tile metadata (stage position + scene info)."""
    tile_info: dict[str, Any] = {
        "scene_index": scene_index,
        "scene_name": scene_name,
    }

    try:
        pps = bio_image.physical_pixel_sizes
        if pps is not None:
            tile_info["physical_pixel_size_z"] = pps.Z
            tile_info["physical_pixel_size_y"] = pps.Y
            tile_info["physical_pixel_size_x"] = pps.X
    except Exception:
        pass

    try:
        ome_meta = bio_image.ome_metadata
        if ome_meta and len(ome_meta.images) > scene_index:
            pixels = ome_meta.images[scene_index].pixels
            if pixels.planes:
                plane = pixels.planes[0]
                if plane.position_x is not None:
                    tile_info["stage_x"] = float(plane.position_x)
                if plane.position_y is not None:
                    tile_info["stage_y"] = float(plane.position_y)
                if plane.position_z is not None:
                    tile_info["stage_z"] = float(plane.position_z)
    except Exception:
        pass

    return tile_info


def _coerce_tile_position_yx(pos: Any) -> tuple[float, float] | None:
    """Convert a mosaic tile position object into a (y_px, x_px) tuple."""
    if isinstance(pos, dict):
        y_val = pos.get("y", pos.get("Y"))
        x_val = pos.get("x", pos.get("X"))
        if y_val is None or x_val is None:
            return None
        return float(y_val), float(x_val)

    if isinstance(pos, (list, tuple)):
        if len(pos) >= 2:
            return float(pos[0]), float(pos[1])
        return None

    y_val = getattr(pos, "y", getattr(pos, "Y", None))
    x_val = getattr(pos, "x", getattr(pos, "X", None))
    if y_val is None or x_val is None:
        return None
    return float(y_val), float(x_val)


def _extract_mosaic_tile_positions_yx(bio_image: BioImage, m_dim: int) -> dict[int, tuple[float, float]]:
    """Return per-M-index mosaic tile positions in pixel space as (y_px, x_px)."""
    positions_by_index: dict[int, tuple[float, float]] = {}

    # Preferred API in bioio/aicsimageio: all positions at once.
    try:
        all_positions = bio_image.get_mosaic_tile_positions()
        if isinstance(all_positions, (list, tuple)):
            for idx, pos in enumerate(all_positions):
                yx = _coerce_tile_position_yx(pos)
                if yx is not None:
                    positions_by_index[idx] = yx
    except Exception:
        pass

    # Fallback API in some versions: one position per tile index.
    if len(positions_by_index) < m_dim:
        for idx in range(int(m_dim)):
            if idx in positions_by_index:
                continue
            try:
                pos = bio_image.get_mosaic_tile_position(idx)
            except Exception:
                continue
            yx = _coerce_tile_position_yx(pos)
            if yx is not None:
                positions_by_index[idx] = yx

    return positions_by_index


def _apply_per_m_tile_stage_um(
    base_tile_info: dict[str, Any],
    bio_image: BioImage,
    m_index: int,
    m_dim: int,
    tile_shape_yx: tuple[int, int],
    positions_by_index: dict[int, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Attach per-M tile stage origin (and center) in microns to tile metadata."""
    tile_info = dict(base_tile_info)
    pps_x = float(tile_info.get("physical_pixel_size_x", 1.0) or 1.0)
    pps_y = float(tile_info.get("physical_pixel_size_y", 1.0) or 1.0)
    pps_z = float(tile_info.get("physical_pixel_size_z", 1.0) or 1.0)

    base_stage_x = float(tile_info.get("stage_x", 0.0) or 0.0)
    base_stage_y = float(tile_info.get("stage_y", 0.0) or 0.0)
    base_stage_z = float(tile_info.get("stage_z", 0.0) or 0.0)

    pos_map = positions_by_index if positions_by_index is not None else _extract_mosaic_tile_positions_yx(bio_image, m_dim)
    if pos_map:
        if 0 in pos_map:
            y0_px, x0_px = pos_map[0]
        else:
            first_idx = min(pos_map.keys())
            y0_px, x0_px = pos_map[first_idx]

        yx_px = pos_map.get(int(m_index))
        if yx_px is not None:
            y_px, x_px = yx_px
            stage_x = base_stage_x + (x_px - x0_px) * pps_x
            stage_y = base_stage_y + (y_px - y0_px) * pps_y
            tile_info["stage_x"] = float(stage_x)
            tile_info["stage_y"] = float(stage_y)
            tile_info["stage_z"] = float(base_stage_z)

    # Store tile-center coordinates for diagnostics / downstream stitching.
    size_y_px, size_x_px = int(tile_shape_yx[0]), int(tile_shape_yx[1])
    stage_x_for_center = float(tile_info.get("stage_x", base_stage_x))
    stage_y_for_center = float(tile_info.get("stage_y", base_stage_y))
    tile_info["tile_center_x"] = float(stage_x_for_center + 0.5 * size_x_px * pps_x)
    tile_info["tile_center_y"] = float(stage_y_for_center + 0.5 * size_y_px * pps_y)
    tile_info["tile_center_z"] = float(base_stage_z + 0.5 * pps_z)

    return tile_info


def _get_scene_image_with_optional_m(bio_image: BioImage) -> tuple[da.Array, bool]:
    """Return scene data as MTCZYX (if available) or TCZYX."""

    def _coerce_axes(data: np.ndarray, order: str, target_order: str) -> np.ndarray:
        arr = np.asarray(data)
        order_local = order
        if arr.ndim != len(order_local):
            raise ValueError(
                f"Array ndim ({arr.ndim}) does not match order '{order_local}' ({len(order_local)})."
            )
        for axis_name in target_order:
            if axis_name not in order_local:
                arr = np.expand_dims(arr, axis=0)
                order_local = axis_name + order_local
        perm = [order_local.index(axis_name) for axis_name in target_order]
        return np.transpose(arr, axes=perm)

    native_order = ""
    try:
        native_order = str(getattr(bio_image.dims, "order", ""))
    except Exception:
        native_order = ""

    if native_order and "M" in native_order:
        try:
            raw = np.asarray(bio_image.get_image_data(native_order))
            image_m = da.asarray(_coerce_axes(raw, native_order, "MTCZYX"))
            if image_m.ndim == 6:
                return image_m, True
        except Exception:
            pass

    for order_try in ("MTCZYX", "MCZYX", "MTZYX", "MZYX"):
        try:
            raw = np.asarray(bio_image.get_image_data(order_try))
            image_m = da.asarray(_coerce_axes(raw, order_try, "MTCZYX"))
            if image_m.ndim == 6:
                return image_m, True
        except Exception:
            continue

    if native_order:
        try:
            raw = np.asarray(bio_image.get_image_data(native_order))
            image = da.asarray(_coerce_axes(raw, native_order, "TCZYX"))
            if image.ndim == 5:
                return image, False
        except Exception:
            pass

    image = da.asarray(np.asarray(bio_image.get_image_data("TCZYX")))
    if image.ndim != 5:
        raise ValueError(f"Expected TCZYX image shape, found {image.shape}.")
    return image, False


def _write_image_collection_metadata(root_group: zarr.Group, image_group_paths: list[str]) -> None:
    """Write transitional metadata for multi-image collections.

    Follows the bioformats2raw layout (v3):
    - root .zattrs  → {"bioformats2raw.layout": 3}   (flat, NOT nested under "ome")
    - OME/.zattrs   → {"series": ["0", "1", ...]}     (flat, NOT nested under "ome")

    The original code nested both dicts under an extra "ome" key, which meant
    ome_zarr.reader.Reader could not locate `bioformats2raw.layout` or `series`
    and therefore could not enumerate the image sub-groups.
    """
    if not image_group_paths:
        return

    # Root-level: flag this as a bioformats2raw v3 layout.
    root_attrs = dict(root_group.attrs)
    root_attrs["bioformats2raw.layout"] = 3
    root_group.attrs.update(root_attrs)

    # OME sub-group: list of image group paths so readers can discover each image.
    ome_group = root_group.require_group("OME")
    ome_attrs = dict(ome_group.attrs)
    ome_attrs["series"] = image_group_paths
    ome_group.attrs.update(ome_attrs)


def _get_multiscales(group: zarr.Group) -> list | None:
    """Return multiscales metadata from common locations."""
    multiscales = group.attrs.get("multiscales")
    if isinstance(multiscales, list) and multiscales:
        return multiscales

    ome_attrs = group.attrs.get("ome")
    if isinstance(ome_attrs, dict):
        multiscales = ome_attrs.get("multiscales")
        if isinstance(multiscales, list) and multiscales:
            return multiscales

    return None


def _enumerate_ngff_image_groups(root_group: zarr.Group) -> list[tuple[zarr.Group, str]]:
    """List image groups that contain multiscales metadata.

    Supports root-level images, transitional OME/series layouts, and
    direct child groups (for multi-tile writes).
    """
    image_groups: list[tuple[zarr.Group, str]] = []

    if "OME" in root_group:
        ome_group = root_group["OME"]
        if isinstance(ome_group, zarr.Group):
            series_paths = ome_group.attrs.get("series", [])
            if isinstance(series_paths, list) and series_paths:
                for series_path in series_paths:
                    series_path_clean = str(series_path).strip("/")
                    if series_path_clean in root_group:
                        candidate = root_group[series_path_clean]
                        if isinstance(candidate, zarr.Group) and _get_multiscales(candidate):
                            image_groups.append((candidate, series_path_clean))
                if image_groups:
                    return image_groups

    if _get_multiscales(root_group):
        return [(root_group, "/")]

    for key in sorted(root_group.keys()):
        candidate = root_group[key]
        if isinstance(candidate, zarr.Group) and _get_multiscales(candidate):
            image_groups.append((candidate, str(key)))

    return image_groups


def _get_ngff_primary_dataset(group: zarr.Group) -> zarr.Array:
    """Return the highest-resolution dataset in a multiscales group."""
    multiscales = _get_multiscales(group)
    if not multiscales:
        raise ValueError(f"Group has no multiscales metadata: {group.name}")

    datasets = multiscales[0].get("datasets", [])
    if not datasets:
        raise ValueError(f"No datasets listed in multiscales for group: {group.name}")

    dataset_path = datasets[0].get("path", "0") if isinstance(datasets[0], dict) else "0"
    if dataset_path not in group:
        raise ValueError(f"Dataset path '{dataset_path}' not found in group: {group.name}")

    dataset = group[dataset_path]
    if not isinstance(dataset, zarr.Array):
        raise ValueError(f"Expected zarr.Array at '{group.name}/{dataset_path}'.")
    return dataset


# ---------------------------------------------------------------------------
# Public converters
# ---------------------------------------------------------------------------


def convert_psf(psf_path: str | Path, output_path: str | Path) -> Path:
    """Load a Deconwolf PSF TIFF and save it as an OME-NGFF Zarr array."""
    psf = np.asarray(tifffile.imread(str(psf_path)), dtype=np.float32)
    psf = np.squeeze(psf)

    if psf.ndim == 2:
        axes = "yx"
    elif psf.ndim == 3:
        axes = "zyx"
    else:
        raise ValueError(
            f"Expected a 2D or 3D PSF TIFF, but found shape {psf.shape} (ndim={psf.ndim})."
        )

    output = _reset_existing_output(output_path)
    _write_ngff_array(da.asarray(psf), output, axes=axes, first_write=True, storage_dtype=np.float32)
    return output


def convert_czi_to_ngff(czi_path: str | Path, output_path: str | Path) -> Path:
    """Convert a CZI file to OME-NGFF and include per-tile scene metadata."""
    output = _reset_existing_output(output_path)
    bio_image = BioImage(str(czi_path), reconstruct_mosaic=False, use_aicspylibczi=True)

    scenes = list(bio_image.scenes)
    if not scenes:
        raise ValueError(f"No scenes found in CZI file: {czi_path}")

    tile_metadata: list[dict[str, Any]] = []
    tile_index = 0
    image_group_paths: list[str] = []

    for scene_index, scene_name in enumerate(scenes):
        bio_image.set_scene(scene_name)
        image, has_m = _get_scene_image_with_optional_m(bio_image)
        scene_info_base = _extract_scene_tile_info(bio_image, scene_index, str(scene_name))

        if has_m:
            m_dim = int(image.shape[0])
            positions_by_index = _extract_mosaic_tile_positions_yx(bio_image, m_dim)
            for m_idx in range(m_dim):
                tile_img = image[m_idx]
                group_path = (
                    None
                    if len(scenes) == 1 and m_dim == 1 and tile_index == 0
                    else str(tile_index)
                )
                # BUG FIX: only the very first write initialises the store with "w".
                _write_ngff_array(
                    tile_img, output, axes="tczyx",
                    group_path=group_path,
                    first_write=(tile_index == 0),
                )
                if group_path is not None:
                    image_group_paths.append(group_path)

                scene_info = _apply_per_m_tile_stage_um(
                    base_tile_info=scene_info_base,
                    bio_image=bio_image,
                    m_index=m_idx,
                    m_dim=m_dim,
                    tile_shape_yx=(int(tile_img.shape[-2]), int(tile_img.shape[-1])),
                    positions_by_index=positions_by_index,
                )
                scene_info["m_index"] = m_idx
                scene_info["ngff_group"] = group_path if group_path is not None else "/"
                tile_metadata.append(scene_info)
                tile_index += 1
        else:
            group_path = None if len(scenes) == 1 and tile_index == 0 else str(tile_index)
            # BUG FIX: only the very first write initialises the store with "w".
            _write_ngff_array(
                image, output, axes="tczyx",
                group_path=group_path,
                first_write=(tile_index == 0),
            )
            if group_path is not None:
                image_group_paths.append(group_path)

            scene_info = _apply_per_m_tile_stage_um(
                base_tile_info=scene_info_base,
                bio_image=bio_image,
                m_index=0,
                m_dim=1,
                tile_shape_yx=(int(image.shape[-2]), int(image.shape[-1])),
            )
            scene_info["ngff_group"] = group_path if group_path is not None else "/"
            tile_metadata.append(scene_info)
            tile_index += 1

    # Re-open store in append mode to attach collection-level metadata.
    zarr_group = _open_group_with_optional_kvikio(output, mode="a")
    zarr_group.attrs["tile_metadata"] = tile_metadata
    _write_image_collection_metadata(zarr_group, image_group_paths)
    print(f"Converted {len(tile_metadata)} tile(s) from {czi_path}.", flush=True)

    return output


def extract_dapi_channel(
    input_file: str | Path,
    output_folder: str | Path,
    dapi_channel: int,
    dapi_timepoint: int,
) -> list[Path]:
    """Extract one channel/timepoint volume from each tile in an OME-Zarr store.

    Expects each image tile to follow TCZYX ordering and writes one TIFF per tile.
    """
    if dapi_channel < 0:
        raise ValueError("dapi_channel must be >= 0")
    if dapi_timepoint < 0:
        raise ValueError("dapi_timepoint must be >= 0")

    input_path = Path(input_file)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    root_group = _open_group_with_optional_kvikio(input_path, mode="r")
    image_groups = _enumerate_ngff_image_groups(root_group)
    if not image_groups:
        raise ValueError(f"No image groups found in OME-Zarr input: {input_file}")

    written_paths: list[Path] = []
    for tile_index, (image_group, group_path) in enumerate(image_groups):
        arr = da.from_zarr(_get_ngff_primary_dataset(image_group))
        if arr.ndim != 5:
            raise ValueError(
                f"Expected TCZYX array for group '{group_path}', found shape {arr.shape}."
            )

        t_dim = int(arr.shape[0])
        c_dim = int(arr.shape[1])
        if dapi_timepoint >= t_dim:
            raise ValueError(
                f"dapi_timepoint {dapi_timepoint} out of bounds for group '{group_path}' (T={t_dim})."
            )
        if dapi_channel >= c_dim:
            raise ValueError(
                f"dapi_channel {dapi_channel} out of bounds for group '{group_path}' (C={c_dim})."
            )

        # Extract ZYX volume for the requested (T, C) location.
        dapi_zyx = np.asarray(arr[dapi_timepoint, dapi_channel].compute(), dtype=np.float32)

        group_label = "root" if group_path == "/" else group_path.replace("/", "_")
        out_path = output_dir / (
            f"tile_{tile_index:04d}_group_{group_label}_t{dapi_timepoint}_c{dapi_channel}_DAPI.tiff"
        )
        tifffile.imwrite(str(out_path), dapi_zyx)
        written_paths.append(out_path)

    print(
        f"Extracted DAPI channel C={dapi_channel}, T={dapi_timepoint} "
        f"for {len(written_paths)} tile(s) into {output_dir}.",
        flush=True,
    )
    return written_paths


def return_dapi(input_folder: str | Path, input_zarr: str | Path) -> list[Path]:
    """Replace each tile DAPI plane in OME-Zarr using matching *_DAPI.tif(f) files.

    Expects files named like:
    shbcl_tile_0000_group_fovs_fov_0000_t0_c2_DAPI.tiff
    """
    folder = Path(input_folder)
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Input folder not found: {folder}")

    ngff_path = Path(input_zarr)
    if not ngff_path.exists() or not ngff_path.is_dir():
        raise NotADirectoryError(f"Input OME-Zarr path not found: {ngff_path}")

    pattern = re.compile(
        r"^tile_(?P<tile>\d+)_group_(?P<group>.+?)_t(?P<t>\d+)_c(?P<c>\d+)_DAPI\.tiff?$",
        flags=re.IGNORECASE,
    )

    dapi_files: list[Path] = []
    for file_path in sorted(folder.iterdir()):
        if file_path.is_file() and pattern.match(file_path.name):
            dapi_files.append(file_path)

    if not dapi_files:
        raise FileNotFoundError(
            f"No files matching shbcl_tile_*_group_*_t*_c*_DAPI.tiff found in {folder}."
        )

    root_group = _open_group_with_optional_kvikio(ngff_path, mode="r+")
    image_groups = _enumerate_ngff_image_groups(root_group)
    if not image_groups:
        raise ValueError(f"No image groups found in OME-Zarr input: {ngff_path}")

    group_lookup: dict[str, zarr.Group] = {}
    for image_group, group_path in image_groups:
        label = "root" if group_path == "/" else group_path.replace("/", "_")
        group_lookup[label] = image_group

    updated_files: list[Path] = []
    for file_path in dapi_files:
        match = pattern.match(file_path.name)
        if match is None:
            continue

        tile_idx = int(match.group("tile"))
        group_label = match.group("group")
        t_idx = int(match.group("t"))
        c_idx = int(match.group("c"))

        if group_label not in group_lookup:
            raise ValueError(
                f"Could not find OME-Zarr group for label '{group_label}' from file {file_path.name}."
            )

        image_group = group_lookup[group_label]
        dataset = _get_ngff_primary_dataset(image_group)
        if dataset.ndim != 5:
            raise ValueError(
                f"Expected TCZYX dataset for group '{group_label}', found shape {dataset.shape}."
            )

        t_dim = int(dataset.shape[0])
        c_dim = int(dataset.shape[1])
        if t_idx >= t_dim:
            raise ValueError(
                f"Time index {t_idx} out of bounds for group '{group_label}' (T={t_dim})."
            )
        if c_idx >= c_dim:
            raise ValueError(
                f"Channel index {c_idx} out of bounds for group '{group_label}' (C={c_dim})."
            )

        dapi_dw = np.asarray(tifffile.imread(str(file_path)), dtype=dataset.dtype)
        expected_shape = tuple(int(v) for v in dataset.shape[2:])
        if dapi_dw.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {file_path.name}: got {dapi_dw.shape}, expected {expected_shape} "
                f"for tile {tile_idx}, group '{group_label}', t={t_idx}, c={c_idx}."
            )

        dataset[t_idx, c_idx, :, :, :] = dapi_dw
        updated_files.append(file_path)

    print(
        f"Replaced DAPI channel for {len(updated_files)} tile(s) in {ngff_path} "
        f"using files from {folder} (input_zarr modified in place).",
        flush=True,
    )
    return updated_files


try:
    from .tracing import instrument_module_functions
except ImportError:  # pragma: no cover - supports direct module execution
    from tracing import instrument_module_functions

instrument_module_functions(globals())
