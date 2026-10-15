"""Trace the flood path downstream of a point on the 30 m D8 grid and list the settlements along it."""

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from pysheds import sview
from pysheds.grid import Grid
from rasterio.transform import rowcol, xy
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from reach_stage import D8_OFFSETS

SNAP_RADIUS_PX = 10
CHANNEL_MIN_LOG_ACC = 2.5
MAX_TRACE_STEPS = 200_000
MAX_SETTLEMENT_DISTANCE_M = 3000


def snap_to_channel(acc_log: np.ndarray, row: int, col: int) -> tuple[int, int]:
    """Move (row, col) to the highest-accumulation cell within SNAP_RADIUS_PX if that cell is a channel."""
    row_start, row_stop = max(0, row - SNAP_RADIUS_PX), min(acc_log.shape[0], row + SNAP_RADIUS_PX + 1)
    col_start, col_stop = max(0, col - SNAP_RADIUS_PX), min(acc_log.shape[1], col + SNAP_RADIUS_PX + 1)
    window = acc_log[row_start:row_stop, col_start:col_stop]
    window_row, window_col = np.unravel_index(np.nanargmax(window), window.shape)
    if window[window_row, window_col] >= CHANNEL_MIN_LOG_ACC:
        return int(row_start + window_row), int(col_start + window_col)
    return row, col


def trace(fdir: np.ndarray, row: int, col: int) -> np.ndarray:
    """(row, col) of the cells along the D8 path from (row, col) to a sink, the grid edge or a loop."""
    path = [(row, col)]
    visited = {(row, col)}
    for _ in range(MAX_TRACE_STEPS):
        code = int(fdir[row, col])
        if code not in D8_OFFSETS:
            break
        d_row, d_col = D8_OFFSETS[code]
        row, col = row + d_row, col + d_col
        if not (0 <= row < fdir.shape[0] and 0 <= col < fdir.shape[1]) or (row, col) in visited:
            break
        path.append((row, col))
        visited.add((row, col))
    return np.array(path)


def height_above_path(dem_tif: str, fdir_tif: str, path_rc: np.ndarray) -> np.ndarray:
    """HAND with the traced path as the only channel."""
    grid = Grid.from_raster(dem_tif)
    dem = grid.read_raster(dem_tif)
    fdir_raster = grid.read_raster(fdir_tif)
    view = sview.ViewFinder(affine=fdir_raster.affine, shape=fdir_raster.shape, crs=fdir_raster.crs, nodata=0)
    fdir = sview.Raster(np.nan_to_num(np.asarray(fdir_raster), nan=0).astype('int64'), viewfinder=view)
    channel = np.zeros(dem.shape, bool)
    channel[path_rc[:, 0], path_rc[:, 1]] = True
    channel_view = sview.ViewFinder(
        affine=fdir_raster.affine, shape=fdir_raster.shape, crs=fdir_raster.crs, nodata=False
    )
    hand = grid.compute_hand(fdir, dem, sview.Raster(channel, viewfinder=channel_view), dirmap=tuple(D8_OFFSETS))
    return np.asarray(hand, dtype='float32')


def settlements_along(
    places: gpd.GeoDataFrame,
    crs: str,
    transform: object,
    height: np.ndarray,
    path_xy: np.ndarray,
    dist_m: np.ndarray,
    tiers: tuple[float, float],
) -> gpd.GeoDataFrame:
    """Places at most tiers[1] m above the path and MAX_SETTLEMENT_DISTANCE_M from it, ordered downstream."""
    settlements = places.to_crs(crs).copy()
    centroid_x = settlements.geometry.centroid.x.values
    centroid_y = settlements.geometry.centroid.y.values
    rows, cols = rowcol(transform, centroid_x, centroid_y)
    rows = np.clip(np.asarray(rows), 0, height.shape[0] - 1)
    cols = np.clip(np.asarray(cols), 0, height.shape[1] - 1)
    settlements['h_above_path_m'] = height[rows, cols]
    distance_m, nearest = cKDTree(path_xy).query(np.c_[centroid_x, centroid_y])
    settlements['dist_to_path_m'] = distance_m
    settlements['km_downstream'] = dist_m[nearest] / 1000
    height_m = settlements['h_above_path_m']
    settlements['tier'] = np.where(height_m <= tiers[0], 'high', np.where(height_m <= tiers[1], 'medium', 'low'))
    near_path = (height_m <= tiers[1]) & (settlements['dist_to_path_m'] <= MAX_SETTLEMENT_DISTANCE_M)
    return settlements[near_path].sort_values('km_downstream')


def trace_flood_path(
    lon: float,
    lat: float,
    dem_tif: str,
    fdir_tif: str,
    acc_tif: str,
    *,
    places: gpd.GeoDataFrame | None,
    tiers: tuple[float, float],
    crs: str,
) -> dict[str, object]:
    """Return dict(path, hand: height above the path, settlements when `places` is not empty)."""
    with rasterio.open(fdir_tif) as src:
        fdir = np.nan_to_num(src.read(1), nan=0).astype('int64')
        transform = src.transform
    with rasterio.open(acc_tif) as src:
        acc_log = src.read(1)
    with rasterio.open(dem_tif) as src:
        dem = src.read(1)
    x, y = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(lon, lat)
    path_rc = trace(fdir, *snap_to_channel(acc_log, *rowcol(transform, x, y)))
    xs, ys = xy(transform, path_rc[:, 0], path_rc[:, 1])
    elevation = dem[path_rc[:, 0], path_rc[:, 1]]
    dist_m = np.cumsum(np.r_[0, np.hypot(np.diff(xs), np.diff(ys))])
    height = height_above_path(dem_tif, fdir_tif, path_rc)
    path = gpd.GeoDataFrame(
        dict(length_km=[dist_m[-1] / 1000], drop_m=[float(elevation[0] - elevation[-1])]),
        geometry=[LineString(list(zip(xs, ys, strict=True)))],
        crs=crs,
    )
    result: dict[str, object] = dict(path=path, hand=height)
    if places is not None and len(places):
        result['settlements'] = settlements_along(places, crs, transform, height, np.c_[xs, ys], dist_m, tiers)
    return result
