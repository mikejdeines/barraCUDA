import shutil
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import pandas as pd
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel, TableModel
import numpy as np
from scipy.sparse import csr_matrix
import shapely
import shapely.vectorized
from anndata import AnnData


def _string_object_index(values, name: str | None = None) -> pd.Index:
	"""Return an object-backed string index that AnnData can write portably."""
	return pd.Index([str(value) for value in values], dtype=object, name=name)


def load_nuclei_shapes(nuclei_parquet: Path):
	gdf = gpd.read_parquet(nuclei_parquet)
	if not isinstance(gdf, gpd.GeoDataFrame):
		raise ValueError(f"Expected a GeoParquet file at {nuclei_parquet}, got {type(gdf)}")
	if gdf.geometry.name not in gdf.columns:
		raise ValueError(f"No geometry column found in {nuclei_parquet}")

	if isinstance(gdf, gpd.GeoSeries):
		gdf = gpd.GeoDataFrame(geometry=gdf, crs=gdf.crs)
	elif not isinstance(gdf, gpd.GeoDataFrame):
		raise TypeError(f"Expected GeoDataFrame/GeoSeries, got {type(gdf)}")

	return ShapesModel.parse(gdf)


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
	lower_to_original = {c.lower(): c for c in columns}
	for candidate in candidates:
		if candidate.lower() in lower_to_original:
			return lower_to_original[candidate.lower()]
	return None


def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Drop duplicate CSV columns created by alias normalization.

	Pandas returns a DataFrame, not a Series, when selecting a duplicated column
	name. That breaks later scalar checks such as ``df["feature_name"].isna().any()``.
	Keep the first occurrence because load_transcripts deliberately prefers the
	canonical column before aliases such as ``gene``.
	"""
	if not df.columns.has_duplicates:
		return df
	return df.loc[:, ~df.columns.duplicated()].copy()


def _apply_feature_names(df: pd.DataFrame, feature_names: Sequence[str] | None, transcripts_csv: Path) -> pd.DataFrame:
	if feature_names is None:
		return df

	names = [str(name) for name in feature_names]
	if not names:
		raise ValueError("feature_names must contain at least one value when provided")

	if "channel_index" in df.columns:
		channel_indices = sorted(df["channel_index"].dropna().unique().tolist())
		if len(names) != len(channel_indices):
			raise ValueError(
				f"Expected {len(channel_indices)} feature names for channel_index values "
				f"{channel_indices} in {transcripts_csv}, got {len(names)}"
			)
		channel_to_feature = dict(zip(channel_indices, names, strict=True))
		df = df.copy()
		df["feature_name"] = df["channel_index"].map(channel_to_feature)
		if df["feature_name"].isna().any():
			raise ValueError(f"Could not assign feature_name for every transcript in {transcripts_csv}")
		return df

	if len(names) == 1:
		df = df.copy()
		df["feature_name"] = names[0]
		return df

	if len(names) == len(df):
		df = df.copy()
		df["feature_name"] = names
		return df

	raise ValueError(
		f"Could not apply {len(names)} feature names to {transcripts_csv}; expected one "
		f"name, one name per row ({len(df)}), or a channel_index column for channel mapping."
	)


def _set_points_feature_key(points, feature_key: str):
	points.attrs.setdefault("spatialdata_attrs", {})["feature_key"] = feature_key
	return points


def load_transcripts(transcripts_csv: Path, feature_names: Sequence[str] | None = None):
	df = pd.read_csv(transcripts_csv)
	if df.empty:
		raise ValueError(f"Transcripts CSV is empty: {transcripts_csv}")

	x_col = _find_column(df.columns.tolist(), ["x", "x_location", "global_x", "xcoord", "xc"])
	y_col = _find_column(df.columns.tolist(), ["y", "y_location", "global_y", "ycoord", "yc"])
	z_col = _find_column(df.columns.tolist(), ["z", "z_location", "global_z", "zcoord", "zc"])
	gene_col = _find_column(df.columns.tolist(), ["feature_name", "gene", "target", "target_name", "gene_name"])

	if x_col is None or y_col is None:
		raise ValueError(
			f"Could not infer transcript x/y columns from {transcripts_csv}. "
			f"Columns found: {', '.join(df.columns.astype(str))}"
		)

	if gene_col is not None and gene_col != "feature_name":
		df = df.rename(columns={gene_col: "feature_name"})
		df = _deduplicate_columns(df)
	df = _apply_feature_names(df, feature_names, transcripts_csv)
	if "feature_name" not in df.columns:
		raise ValueError(
			f"Could not infer transcript feature/gene column from {transcripts_csv}. "
			f"Columns found: {', '.join(df.columns.astype(str))}"
		)

	df = df.rename(columns={x_col: "x", y_col: "y", **({z_col: "z"} if z_col is not None else {})})

	coordinates = {"x": "x", "y": "y"}
	if "z" in df.columns:
		coordinates["z"] = "z"

	# Ground-truth transcripts are spot observations, not circle/polygon shapes.
	# Passing feature_key currently triggers a SpatialData/Dask assignment error in
	# some environments, while parsing without it preserves feature_name as a column.
	try:
		points = PointsModel.parse(df, coordinates=coordinates)
		return _set_points_feature_key(points, "feature_name")
	except Exception as exc:
		raise RuntimeError(
			f"Could not parse transcript spots from {transcripts_csv} as SpatialData points"
		) from exc

def create_table(sdata):
	points_df = sdata.points["transcripts"].compute()
	nuclei_gdf = sdata.shapes["nuclei"].copy()
	unique_genes = _string_object_index(points_df["feature_name"].dropna().unique(), name="feature_name")
	gene_to_idx = {gene: i for i, gene in enumerate(unique_genes)}
	row_idx: list[int] = []
	col_idx: list[int] = []
	data_vals: list[int] = []
	if "z_min" not in nuclei_gdf.columns:
		def get_z_bounds(poly):
			bounds = shapely.get_coordinates(poly)
			if bounds.shape[1] < 3: return (0.0, 0.0)
			return (bounds[:, 2].min(), bounds[:, 2].max())
		z_bounds = nuclei_gdf.geometry.apply(get_z_bounds)
		nuclei_gdf["z_min"] = z_bounds.apply(lambda x: x[0])
		nuclei_gdf["z_max"] = z_bounds.apply(lambda x: x[1])
	for i, (idx, row) in enumerate(nuclei_gdf.iterrows()):
		z_mask = (points_df["z"] >= row["z_min"] - 0.1) & (points_df["z"] <= row["z_max"] + 0.1)
		slice_points = points_df[z_mask]
		if slice_points.empty:
			continue

		is_inside = shapely.vectorized.contains(row.geometry, slice_points.x.values, slice_points.y.values)
		if not np.any(is_inside):
			continue

		genes_found = slice_points["feature_name"].values[is_inside]
		gene_counts = pd.Series(genes_found).value_counts()
		for g, c in gene_counts.items():
			g = str(g)
			row_idx.append(i)
			col_idx.append(gene_to_idx[g])
			data_vals.append(int(c))

	counts_csr = csr_matrix(
		(data_vals, (row_idx, col_idx)),
		shape=(len(nuclei_gdf), len(unique_genes)),
		dtype=np.int32,
	)
	table =  AnnData(
		X=counts_csr,
		obs=pd.DataFrame(index=_string_object_index(nuclei_gdf.index, name="instance")),
		var=pd.DataFrame(index=unique_genes),
	)
	table.obs["region"] = "nuclei"
	table.obs["region"] = table.obs["region"].astype("category")
	table.obs["instance_id"] = np.arange(len(table.obs))
	table = TableModel.parse(table,
						  region="nuclei",
						  region_key="region",
						  instance_key="instance_id")
	sdata.tables["table"] = table
	return sdata

def build_spatialdata(
	nuclei_parquet: Path,
	transcripts_csv: Path,
	feature_names: Sequence[str] | None = None,
):
	for required_path in (nuclei_parquet, transcripts_csv):
		if not required_path.exists():
			raise FileNotFoundError(f"Required input not found: {required_path}")

	nuclei = load_nuclei_shapes(nuclei_parquet)
	transcripts = load_transcripts(transcripts_csv, feature_names=feature_names)

	shapes = {"nuclei": nuclei}
	points = {"transcripts": transcripts}

	return SpatialData(shapes=shapes, points=points)


def save_spatialdata(sdata: SpatialData, output_path: Path):
	if output_path.exists():
		shutil.rmtree(output_path)

	try:
		sdata.write(output_path)
	except AttributeError:
		sdata.write_zarr(output_path)


def spatialdata_from_stores(
	nuclei_parquet: str | Path,
	transcripts_csv: str | Path,
	output_path: str | Path,
	feature_names: Sequence[str] | None = None,
) -> SpatialData:
	"""Create and save SpatialData from nuclei/transcripts stores.

	Parameters
	----------
	nuclei_parquet
		Path to nuclei GeoParquet input.
	transcripts_csv
		Path to transcripts CSV input.
	output_path
		Path where SpatialData Zarr will be written.
	feature_names
		Optional feature names to write into the transcript feature_name column.
		If channel_index is present, names are assigned by sorted channel_index.

	Returns
	-------
	SpatialData
		The created SpatialData object that was written to disk.
	"""
	nuclei_path = Path(nuclei_parquet)
	transcripts_path = Path(transcripts_csv)
	out_path = Path(output_path)

	sdata = build_spatialdata(
		nuclei_parquet=nuclei_path,
		transcripts_csv=transcripts_path,
		feature_names=feature_names,
	)
	sdata = create_table(sdata)
	save_spatialdata(sdata, out_path)
	return sdata


try:
	from .tracing import instrument_module_functions
except ImportError:  # pragma: no cover - supports direct module execution
	from tracing import instrument_module_functions

instrument_module_functions(globals())
