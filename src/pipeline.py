"""Map a flood and calculate infrastructure exposure and settlement access for an area and date."""

import argparse
import collections
import copy
import datetime as dt
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from typing import Any

import geopandas as gpd
import numpy as np
import odc.stac
import planetary_computer
import pystac
import pystac_client
import rasterio
from affine import Affine
from pyproj import Transformer
from pysheds.grid import Grid
from rasterio.warp import Resampling, reproject
from rasterio.warp import transform as warp_transform
from scipy import ndimage as ndi
from shapely.geometry import Polygon

import flowpath
import fusion
import impact
import osm_fetch
import reach_stage
import report
import s1_change_rule
import s1_change_test
import s1_geometry
import s1_grd
import s2_calibration
import store

# blob-storage reads time out now and then
os.environ.setdefault('GDAL_HTTP_MAX_RETRY', '6')
os.environ.setdefault('GDAL_HTTP_RETRY_DELAY', '3')
os.environ.setdefault('GDAL_HTTP_TIMEOUT', '120')
os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif,.TIF,.tiff')

STAC_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1'
ON_PROJECT_VM = os.path.isdir('/content/tri')
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.environ.get('TRI_ROOT') or ('/content/tri/events' if ON_PROJECT_VM else os.path.abspath('runs'))
MODELS_DIR = os.environ.get('TRI_MODELS') or (
    '/content/tri/ckpt' if ON_PROJECT_VM else os.path.join(REPO_DIR, 'models')
)  # <run>/{best.pt,config.json}
RADAR_AI_RUNS = ['r34_scratch_v2', 'r50_scratch_v2', 'r34_s1f11rl_v1']  # radar water model ensemble, equal weights
OPTICAL_AI_RUN = 's2ai_r34_v1'
GRID_RES_M = 10.0
TRANSIENT_IO_ERRORS = (OSError, RuntimeError)


def log(*parts):
    print(*parts, flush=True)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, value, indent=None, default=None):
    with open(path, 'w') as f:
        json.dump(value, f, indent=indent, default=default)


def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1)


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def parse_event_time(text):
    """Return the UTC event time and whether a time was supplied; an omitted offset means UTC."""
    text = text.strip()
    event_time = dt.datetime.fromisoformat(text)
    if event_time.tzinfo is not None:
        event_time = event_time.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return event_time, len(text) > 10


def stac_catalog():
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def retry_io(load, label, delay_s, attempts=3):
    """Call load(), retrying on I/O errors."""
    for _ in range(attempts - 1):
        try:
            return load()
        except TRANSIENT_IO_ERRORS as error:
            log(label, 'retry', repr(error)[:150])
            time.sleep(delay_s)
    return load()


def _grid_from_bbox(bbox, crs):
    """10 m grid in `crs` covering the bbox; its edges are densified because they curve in UTM."""
    west, south, east, north = bbox
    s = np.linspace(0, 1, 41)
    lon = np.r_[west + s * (east - west), np.full(41, east), west + s * (east - west), np.full(41, west)]
    lat = np.r_[np.full(41, south), south + s * (north - south), np.full(41, north), south + s * (north - south)]
    xs, ys = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(lon, lat)
    res = GRID_RES_M
    x0, x1 = np.floor(min(xs) / res) * res, np.ceil(max(xs) / res) * res
    y0, y1 = np.floor(min(ys) / res) * res, np.ceil(max(ys) / res) * res
    transform = Affine(res, 0, x0, 0, -res, y1)
    return dict(
        transform=list(transform)[:6],
        height=round(float((y1 - y0) / res)),
        width=round(float((x1 - x0) / res)),
        crs=crs,
    )


class RunContext:
    """Inputs, output folder and summary of one event run."""

    def __init__(self, event, bbox, date_given, root, should_push, flowpath_source=None):
        self.event = event
        self.bbox = bbox
        self.should_push = should_push
        self.flowpath_source = flowpath_source
        self.date_given = date_given
        self.event_time, self.has_time = parse_event_time(date_given)
        self.event_utc = self.event_time.strftime('%Y-%m-%dT%H:%M:%S')
        self.event_date = self.event_utc[:10]
        self.out_dir = os.path.join(root, event)
        os.makedirs(self.out_dir, exist_ok=True)
        self.crs = f'EPSG:{utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)}'
        self.summary: dict[str, Any] = {}

    def warn(self, message):
        """Add a plain-language warning to the summary; the report shows it."""
        warnings = self.summary.setdefault('warnings', [])
        if message not in warnings:
            warnings.append(message)
        log('WARNING:', message)

    def clear_warnings(self, prefix):
        if self.summary.get('warnings'):
            self.summary['warnings'] = [w for w in self.summary['warnings'] if not w.startswith(prefix)]

    def acquisition_role(self, time_utc):
        """'pre', 'post' or None for an acquisition time 'YYYY-MM-DDTHH:MM:SS' (UTC)."""
        if self.has_time:
            return 'post' if time_utc[:19] >= self.event_utc[:19] else 'pre'
        # with a date only, an acquisition on the event day may precede the flood
        day = time_utc[:10]
        if day == self.event_date:
            return None
        return 'pre' if day < self.event_date else 'post'

    def path(self, *parts):
        """Path under the event folder; creates its parent folder."""
        path = os.path.join(self.out_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def upload(self, path):
        """Push a file of the event folder to the project store when the run pushes."""
        if not self.should_push:
            return
        remote = f'events/{self.event}/' + os.path.relpath(path, self.out_dir).replace(os.sep, '/')
        try:
            store.put(path, remote)
        except (OSError, RuntimeError) as error:  # HTTP errors or a mismatched store copy; the local file is complete
            log('push failed', path, repr(error)[:150])

    def grid(self):
        """Raster profile of the 10 m event grid, stored in grid.json on first use."""
        grid_path = self.path('grid.json')
        if not os.path.exists(grid_path):
            write_json(grid_path, _grid_from_bbox(self.bbox, self.crs))
            self.upload(grid_path)
        stored = read_json(grid_path)
        return dict(
            driver='GTiff',
            height=stored['height'],
            width=stored['width'],
            count=1,
            dtype='float32',
            crs=self.crs,
            transform=Affine(*stored['transform']),
            nodata=np.nan,
        )

    def write_raster(self, relative_path, array, dtype, nodata=None):
        """Write a single-band GeoTIFF on the event grid and push it; float rasters get NaN as nodata by default."""
        is_float = dtype.startswith('float')
        profile = dict(
            self.grid(),
            count=1,
            dtype=dtype,
            compress='deflate',
            tiled=True,
            blockxsize=512,
            blockysize=512,
            predictor=3 if is_float else 2,
        )
        if nodata is not None:
            profile['nodata'] = nodata
        elif is_float:
            profile['nodata'] = np.nan
        else:
            profile.pop('nodata', None)
        path = self.path(relative_path)
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(array.astype(dtype), 1)
        self.upload(path)


def read_optional(ctx, *parts):
    path = ctx.path(*parts)
    return read_band(path) if os.path.exists(path) else None


def to_grid(ctx, array, src_transform, resampling):
    """Resample a layer in the event CRS onto the 10 m event grid."""
    grid = ctx.grid()
    out = np.full((grid['height'], grid['width']), np.nan, 'float32')
    reproject(
        array,
        out,
        src_transform=src_transform,
        src_crs=ctx.crs,
        dst_transform=grid['transform'],
        dst_crs=ctx.crs,
        resampling=resampling,
    )
    return out


S1_SEARCH_PRE_DAYS = 60
S1_SEARCH_POST_DAYS = 24
S1_PRE_DATES = 4
S1_POST_DATES = 2
S1_MIN_POST_COVER = 0.3
GEOCODING_DIAGNOSTICS = ('gcp_line_err', 'gcp_pix_err', 'gcp_pix_err_hybrid', 'lut_line_err', 'lut_pix_err', 'cover')

# grid and ellipsoidal DEM of the S1 geocoding workers
_GRID_STATE: dict[str, Any] = {}


def _set_grid_state(state):
    """Pool initializer: the forkserver workers receive the grid and the DEM instead of inheriting them."""
    _GRID_STATE.update(state)


def _search_s1(ctx):
    """IW VV+VH Sentinel-1 GRD items around the event date."""
    event_date = dt.date.fromisoformat(ctx.event_date)
    window = (
        f'{event_date - dt.timedelta(days=S1_SEARCH_PRE_DAYS)}/{event_date + dt.timedelta(days=S1_SEARCH_POST_DAYS)}'
    )
    found = list(
        stac_catalog().search(collections=['sentinel-1-grd'], bbox=ctx.bbox, datetime=window).item_collection()
    )
    items = [
        item
        for item in found
        if item.properties.get('sar:instrument_mode') == 'IW'
        and {'VV', 'VH'} <= set(item.properties.get('sar:polarizations', []))
    ]
    if found and not items:  # polar and some ocean areas are acquired in HH/HV or EW mode only
        modes = sorted(
            {
                '+'.join(item.properties.get('sar:polarizations', []))
                + ' '
                + str(item.properties.get('sar:instrument_mode'))
                for item in found
            }
        )
        ctx.warn(
            f'Sentinel-1 scenes exist for this area and date ({len(found)}: {", ".join(modes)[:80]}) but none is IW '
            'VV+VH, the only mode this system processes (high-latitude areas are often HH/HV only): no radar evidence.'
        )
    return items


def _select_s1_acquisitions(ctx, items):
    """Select pre-event and post-event acquisitions per track, grouped by track and date."""
    items_by_key: dict[tuple[str, str], list] = {}
    start_times: dict[tuple[str, str], str] = {}
    for item in items:
        properties = item.properties
        track = properties['sat:orbit_state'][:3] + f'{properties["sat:relative_orbit"]:03d}'
        key = (track, properties['datetime'][:10])
        items_by_key.setdefault(key, []).append(item)
        start = properties.get('start_datetime', properties['datetime'])[:19]
        start_times[key] = min(start_times.get(key, '9'), start)
    days_by_track: dict[str, dict[str, list]] = {}
    for (track, day), day_items in items_by_key.items():
        days_by_track.setdefault(track, {})[day] = day_items
    jobs = []
    ambiguous = []
    for track, days in sorted(days_by_track.items()):
        roles = {day: ctx.acquisition_role(start_times[(track, day)]) for day in days}
        ambiguous += [f'{track}_{day}' for day, role in roles.items() if role is None]
        pre = sorted(day for day in days if roles[day] == 'pre')[-S1_PRE_DATES:]
        post = sorted(day for day in days if roles[day] == 'post')[:S1_POST_DATES]
        if len(pre) < 2 or not post:
            log('skip track', track, pre, post)
            continue
        jobs += [(track, day, days[day], roles[day]) for day in pre + post]
    if ambiguous:
        log('same-day acquisitions not used (date-only event, could precede the flood):', ambiguous)
        ctx.summary['s1_ambiguous_same_day'] = ambiguous
    return jobs, start_times


def _geocode_slice(item, state):
    """Calibrate and geocode one GRD slice; return intensities and diagnostics, or None after failed reads."""
    for _ in range(3):
        try:
            return s1_grd.geocode_item(
                item,
                state['transform'],
                state['crs'],
                state['height'],
                state['width'],
                state['dem_ell'],
                log=lambda *_args: None,
            )
        except TRANSIENT_IO_ERRORS as error:
            log('s1 grd retry', item.id, repr(error)[:150])
            time.sleep(15)
    return None


def _write_s1_mosaic(path, sums, counts, state):
    profile = dict(
        driver='GTiff',
        height=state['height'],
        width=state['width'],
        count=2,
        dtype='float32',
        crs=state['crs'],
        transform=state['transform'],
        nodata=np.nan,
        compress='deflate',
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    with rasterio.open(path + '.tmp.tif', 'w', **profile) as dst:
        for band, pol in ((1, 'vv'), (2, 'vh')):
            dst.write(np.where(counts > 0, sums[pol] / np.maximum(counts, 1), np.nan).astype('float32'), band)
    os.replace(path + '.tmp.tif', path)


def _geocode_track_day(task):
    """Geocode all GRD slices of one (track, day) and write their mean mosaic. Returns (diagnostics, written)."""
    _track, _day, item_dicts, _role, path = task
    if os.path.exists(path):
        return [], True
    state = _GRID_STATE
    sums = None
    counts = None
    diagnostics = []
    for item_dict in item_dicts:
        geocoded = _geocode_slice(planetary_computer.sign(pystac.Item.from_dict(item_dict)), state)
        if geocoded is None:
            continue
        sigma0, slice_diagnostics = geocoded
        diagnostics.append({k: round(slice_diagnostics[k], 3) for k in GEOCODING_DIAGNOSTICS if k in slice_diagnostics})
        if sigma0['vv'] is None:
            continue
        if sums is None:
            sums = {pol: np.zeros(sigma0[pol].shape, 'float32') for pol in sigma0}
            counts = np.zeros(sigma0['vv'].shape, 'uint8')
        valid = np.isfinite(sigma0['vv']) & np.isfinite(sigma0['vh'])
        for pol in sigma0:
            sums[pol][valid] += sigma0[pol][valid]
        counts += valid
        del geocoded, sigma0  # free the slice before the next one is geocoded
    if sums is None:
        return diagnostics, False
    _write_s1_mosaic(path, sums, counts, state)
    return diagnostics, True


def _geocode_all(ctx, tasks):
    """Geocode the acquisition groups in Linux worker processes or in the calling process on other systems."""
    if not sys.platform.startswith('linux'):
        return [_geocode_track_day(task) for task in tasks]
    workers = max(1, min(len(tasks), (os.cpu_count() or 2) - 1, 6))
    results = []
    # forkserver: forking a process whose DEM-step threads are alive crashes the pool
    try:
        with ProcessPoolExecutor(
            workers,
            mp_context=mp.get_context('forkserver'),
            initializer=_set_grid_state,
            initargs=(dict(_GRID_STATE),),
        ) as pool:
            for result in pool.map(_geocode_track_day, tasks):
                results.append(result)
    except BrokenProcessPool:
        ctx.warn('S1 worker pool crashed (BrokenProcessPool); remaining acquisitions processed in-process (slower)')
        results += [_geocode_track_day(task) for task in tasks[len(results) :]]
    return results


def _discard_skipped_s1(ctx, index_path, skipped_path):
    """Remove the empty radar outputs of a failed acquisition step before retrying it."""
    evidence = [
        ctx.path('evidence', f'{tag}_{name}.tif')
        for tag in ('s1', 's1j')
        for name in ('ntracks', 'observed', 'geom_uncertain')
    ]
    for path in [index_path, skipped_path, *evidence]:
        if os.path.exists(path):
            os.remove(path)
    log('s1: retrying after an earlier failure (the run had continued without radar)')


def step_s1(ctx):
    """Sentinel-1 GRD stacks on the event grid: own calibration and terrain correction on the Copernicus DEM."""
    index_path = ctx.path('s1', 'index.json')
    skipped_path = ctx.path('s1', 'skipped.json')
    if os.path.exists(skipped_path):
        _discard_skipped_s1(ctx, index_path, skipped_path)
    if os.path.exists(index_path):
        return
    jobs, start_times = _select_s1_acquisitions(ctx, _search_s1(ctx))
    if not jobs:
        log('s1: no usable pre/post track pair')
        write_json(index_path, [])
        ctx.upload(index_path)
        return
    grid = ctx.grid()
    dem_ellipsoidal, geoid_info = s1_grd.dem_ellipsoid(
        read_band(ctx.path('dem', 'dem10.tif')), grid['transform'], ctx.crs
    )
    ctx.summary['s1_processing'] = dict(
        source='sentinel-1-grd (ESA L1 GRD)',
        calibration='sigma0, thermal noise removed',
        terrain_correction='Range-Doppler on Copernicus DEM GLO-30',
        **geoid_info,
    )
    log('geoid', geoid_info)
    tasks = [
        (track, day, [item.to_dict() for item in day_items], role, ctx.path('s1', f'{track}_{day}.tif'))
        for track, day, day_items, role in jobs
    ]
    _GRID_STATE.update(
        transform=grid['transform'], crs=ctx.crs, height=grid['height'], width=grid['width'], dem_ell=dem_ellipsoidal
    )
    try:
        results = _geocode_all(ctx, tasks)
    finally:
        _GRID_STATE.clear()
    index = []
    for (track, day, item_dicts, role, path), (diagnostics, written) in zip(tasks, results, strict=True):
        if not written:
            log('s1 grd failed', track, day)
            continue
        ctx.upload(path)
        with rasterio.open(path) as src:
            cover = float(np.isfinite(src.read(1, out_shape=(1, 200, 200))).mean())
        log('s1', track, day, role, f'cover {cover:.2f}', diagnostics)
        index.append(
            dict(
                track=track,
                day=day,
                time=start_times[(track, day)],
                role=role,
                file=path,
                cover=cover,
                ids=[item_dict['id'] for item_dict in item_dicts],
                radiometry='sigma0_gtc',
                geocoding=diagnostics,
            )
        )
    kept = {r['track'] for r in index if r['role'] == 'post' and r['cover'] >= S1_MIN_POST_COVER}
    kept = {track for track in kept if sum(r['track'] == track and r['role'] == 'pre' for r in index) >= 2}
    write_json(index_path, [r for r in index if r['track'] in kept], indent=1)
    ctx.upload(index_path)


DEM_BUFFER_DEG = 0.15
D8_DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
CHANNEL_MIN_CELLS = 2000  # 30 m cells draining into a channel, the drainage that HAND is measured from


def _load_dem_30m(ctx):
    """Write the buffered 30 m DEM and return its elevations and raster profile."""
    west, south, east, north = ctx.bbox
    buffered = [west - DEM_BUFFER_DEG, south - DEM_BUFFER_DEG, east + DEM_BUFFER_DEG, north + DEM_BUFFER_DEG]
    items = stac_catalog().search(collections=['cop-dem-glo-30'], bbox=buffered).item_collection()
    ds = odc.stac.load(
        items,
        bands=['data'],
        bbox=buffered,
        crs=ctx.crs,
        resolution=30,
        resampling='bilinear',
        dtype='float32',
        nodata=np.nan,
    )
    dem = ds.data.values[0]
    profile = dict(
        driver='GTiff',
        height=dem.shape[0],
        width=dem.shape[1],
        count=1,
        dtype='float32',
        crs=ctx.crs,
        transform=ds.odc.geobox.transform,
        nodata=np.nan,
        compress='deflate',
        tiled=True,
    )
    with rasterio.open(ctx.path('dem', 'dem30_buf.tif'), 'w', **profile) as dst:
        dst.write(dem, 1)
    return dem, profile


def _terrain_layers_30m(dem_path, dem):
    """Elevation, HAND, slope, log10 flow accumulation, D8 flow direction and the conditioned DEM at 30 m."""
    pysheds_grid = Grid.from_raster(dem_path)
    raster = pysheds_grid.read_raster(dem_path)
    conditioned = pysheds_grid.resolve_flats(pysheds_grid.fill_depressions(pysheds_grid.fill_pits(raster)))
    flow_direction = pysheds_grid.flowdir(conditioned, dirmap=D8_DIRMAP)
    accumulation = pysheds_grid.accumulation(flow_direction, dirmap=D8_DIRMAP)
    hand = np.asarray(
        pysheds_grid.compute_hand(flow_direction, conditioned, accumulation > CHANNEL_MIN_CELLS, dirmap=D8_DIRMAP),
        dtype='float32',
    )
    gradient_y, gradient_x = np.gradient(dem, 30.0)
    slope_deg = np.degrees(np.arctan(np.hypot(gradient_x, gradient_y))).astype('float32')
    return dict(
        dem30=dem,
        hand30=hand,
        slope30=slope_deg,
        acc30=np.log10(np.asarray(accumulation, 'float64') + 1).astype('float32'),
        fdir30=np.asarray(flow_direction).astype('float32'),
        demfill30=np.asarray(conditioned, 'float32'),
    )


def step_dem(ctx):
    """Calculate the 30 m drainage grid and resample elevation, HAND, slope and accumulation to the event grid."""
    if os.path.exists(ctx.path('dem', 'hand10.tif')):
        return
    dem, profile_30m = _load_dem_30m(ctx)
    layers_30m = _terrain_layers_30m(ctx.path('dem', 'dem30_buf.tif'), dem)
    for name, layer in layers_30m.items():
        path = ctx.path('dem', f'{name}.tif')
        with rasterio.open(path, 'w', **profile_30m) as dst:
            dst.write(layer.astype('float32'), 1)
        ctx.upload(path)
    for name, resampling in [
        ('dem30', Resampling.bilinear),
        ('hand30', Resampling.bilinear),
        ('slope30', Resampling.bilinear),
        ('acc30', Resampling.max),
    ]:
        resampled = to_grid(ctx, layers_30m[name], profile_30m['transform'], resampling)
        ctx.write_raster(f'dem/{name.replace("30", "10")}.tif', resampled, 'float32')


def _lon_lat_grid(shape, transform, crs):
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    lon, lat = warp_transform(crs, 'EPSG:4326', xs, ys)
    return np.array(lon).reshape(shape), np.array(lat).reshape(shape)


def step_geometry(ctx):
    """Write incidence angle, layover and shadow for each radar track."""
    index = read_json(ctx.path('s1', 'index.json'))
    with rasterio.open(ctx.path('dem', 'dem30_buf.tif')) as src:
        dem = src.read(1)
        transform_30m = src.transform
    lon, lat = _lon_lat_grid(dem.shape, transform_30m, ctx.crs)
    catalog = stac_catalog()
    info = {}
    for track in sorted({r['track'] for r in index}):
        if os.path.exists(ctx.path('geom', f'{track}_layover10.tif')):
            continue
        item_id = next(r for r in index if r['track'] == track and r['role'] == 'post')['ids'][0]
        annotation = s1_geometry.fetch_annotation(catalog.get_collection('sentinel-1-grd').get_item(item_id))
        incidence = s1_geometry.incidence_on_grid(annotation, (lon, lat))
        layover, shadow = s1_geometry.layover_shadow(dem, 30.0, annotation['heading'], incidence)
        info[track] = dict(
            heading=annotation['heading'], inc_min=float(np.nanmin(incidence)), inc_max=float(np.nanmax(incidence))
        )
        ctx.write_raster(
            f'geom/{track}_inc10.tif',
            to_grid(ctx, incidence.astype('float32'), transform_30m, Resampling.bilinear),
            'float32',
        )
        for name, mask in (('layover', layover), ('shadow', shadow)):
            resampled = to_grid(ctx, mask.astype('float32'), transform_30m, Resampling.max)
            ctx.write_raster(f'geom/{track}_{name}10.tif', (resampled > 0.5).astype('uint8'), 'uint8')
        log('geom', track, info[track])
    ctx.summary['geometry'] = info


def layover_or_shadow(ctx, track):
    """Layover or shadow mask of a track; None when the geometry step did not write it."""
    layover_path = ctx.path('geom', f'{track}_layover10.tif')
    shadow_path = ctx.path('geom', f'{track}_shadow10.tif')
    if not (os.path.exists(layover_path) and os.path.exists(shadow_path)):
        return None
    return (read_band(layover_path) > 0) | (read_band(shadow_path) > 0)


S1_RULE_FAR = 0.01
GEOM_MISSING = 'Radar layover/shadow masks are missing'


def _s1_boxcar_db(path):
    with rasterio.open(path) as src:
        return {'VV': s1_change_rule.boxcar_db(src.read(1)), 'VH': s1_change_rule.boxcar_db(src.read(2))}


def _s1_rule_track(ctx, tag, track, records, valley_mask, far):
    """Run the s1_change rule on one track and write its rasters. Returns (changed, observed, thresholds)."""
    pre = [_s1_boxcar_db(r['file']) for r in records if r['role'] == 'pre']
    post = [r for r in records if r['role'] == 'post']
    post1 = _s1_boxcar_db(post[0]['file'])
    post2 = _s1_boxcar_db(post[1]['file']) if len(post) > 1 else None
    pre_by_pol = {pol: [p[pol] for p in pre] for pol in ['VV', 'VH']}
    observed = np.isfinite(post1['VV']) & np.isfinite(pre_by_pol['VV'][-1])
    detection = s1_change_rule.detect_track(pre_by_pol, post1, valley_mask, post2=post2, far=far)
    decrease = detection['dec'] & observed
    increase = detection['inc'] & observed
    ctx.write_raster(f'evidence/{tag}_{track}_dec.tif', np.where(observed, decrease, 255), 'uint8', 255)
    ctx.write_raster(f'evidence/{tag}_{track}_inc.tif', np.where(observed, increase, 255), 'uint8', 255)
    z_max = np.maximum(np.abs(detection['VV_z1']), np.abs(detection['VH_z1']))
    ctx.write_raster(f'evidence/{tag}_{track}_zmax.tif', np.where(observed, z_max, np.nan), 'float32')
    thresholds = {k: v for k, v in detection.items() if k.endswith('_thr')}
    return decrease | increase, observed, thresholds


def step_s1_rules(ctx, joint=False):
    """Calculate radar change per track; the joint rule applies Bonferroni correction to four tests."""
    tag = 's1j' if joint else 's1'
    if os.path.exists(ctx.path('evidence', f'{tag}_ntracks.tif')):
        return
    far = S1_RULE_FAR / 4.0 if joint else S1_RULE_FAR
    index = read_json(ctx.path('s1', 'index.json'))
    grid = ctx.grid()
    hand = read_band(ctx.path('dem', 'hand10.tif'))
    slope = read_band(ctx.path('dem', 'slope10.tif'))
    valley_mask = (hand < 30) & (slope < 30)
    track_count = np.zeros((grid['height'], grid['width']), 'uint8')
    observed = np.zeros_like(track_count, bool)
    geometry_uncertain = np.ones_like(track_count, bool)
    thresholds = {}
    tracks_without_geometry = []
    ctx.clear_warnings(GEOM_MISSING)
    for track in sorted({r['track'] for r in index}):
        records = sorted([r for r in index if r['track'] == track], key=lambda r: r['day'])
        changed, track_observed, thresholds[track] = _s1_rule_track(ctx, tag, track, records, valley_mask, far)
        track_count += changed.astype('uint8')
        observed |= track_observed
        layover_shadow = layover_or_shadow(ctx, track)
        if layover_shadow is None:  # the geometry step failed: the map still builds without the flag
            layover_shadow = np.zeros(track_observed.shape, bool)
            tracks_without_geometry.append(track)
        geometry_uncertain &= ~track_observed | layover_shadow
        log('s1 rules', track, thresholds[track])
    ctx.write_raster(f'evidence/{tag}_ntracks.tif', track_count, 'uint8')
    ctx.write_raster('evidence/s1_observed.tif', observed.astype('uint8'), 'uint8')
    ctx.write_raster('evidence/s1_geom_uncertain.tif', (geometry_uncertain & observed).astype('uint8'), 'uint8')
    if tracks_without_geometry:
        ctx.warn(
            f'{GEOM_MISSING} for radar track(s) {", ".join(tracks_without_geometry)}: treated as having no '
            'layover/shadow (the radar geometric-uncertainty flag is incomplete there).'
        )
    ctx.summary[f'{tag}_thresholds'] = thresholds


def step_s1_joint(ctx):
    step_s1_rules(ctx, joint=True)


S1X_FAR = 0.01  # per track, the nominal rate of s1j
S1X_BOXCAR_PX = 5
S1X_STALE = 'The improved radar test (s1x) does not match the current radar images'


def s1x_signature(ctx):
    """Acquisitions and settings associated with the saved radar change mask."""
    index = read_json(ctx.path('s1', 'index.json'))
    acquisitions = sorted([r['track'], r['day'], r['role'], os.path.basename(r['file'])] for r in index)
    return dict(acquisitions=acquisitions, false_alarm_rate=S1X_FAR, window_px=S1X_BOXCAR_PX)


def _s1x_track(ctx, track, pre, post, slope, calibration_sample, threads):
    """Run s1x on one track and write its -log10 p raster. Returns (-log10 p, flags, track info)."""
    started = time.time()
    intensities = s1_change_test.read_boxcar_means(
        [ctx.path('s1', os.path.basename(r['file'])) for r in pre + post], S1X_BOXCAR_PX, threads
    )
    layover_shadow = layover_or_shadow(ctx, track)
    if layover_shadow is None:
        layover_shadow = np.zeros(slope.shape, bool)
    neg_log_p, observed, details = s1_change_test.s1x_track(
        intensities[: len(pre)],
        intensities[len(pre) :],
        slope,
        ndi.binary_dilation(layover_shadow, iterations=1),
        calibration_sample,
        threads=threads,
    )
    del intensities
    flags = (np.nan_to_num(neg_log_p, nan=0.0) >= -math.log10(S1X_FAR)) & observed
    ctx.write_raster(f'evidence/s1x_{track}_nlp.tif', neg_log_p, 'float32')
    info = dict(
        n_pre=len(pre),
        n_post=len(post),
        strata=len(details['strata']),
        secs=round(time.time() - started, 1),
        flagged_frac_observed=round(float(flags.sum() / max(observed.sum(), 1)), 5),
    )
    return neg_log_p, flags, info


def step_s1x(ctx):
    """Calibrate the radar change test on pre-event pairs; use the joint rule for tracks with fewer than three
    dates.
    """
    count_path = ctx.path('evidence', 's1x_ntracks.tif')
    meta_path = ctx.path('evidence', 's1x_meta.json')
    signature = s1x_signature(ctx)
    if os.path.exists(count_path) and os.path.exists(meta_path) and read_json(meta_path)['sig'] == signature:
        return
    evidence_dir = os.path.dirname(count_path)
    for name in os.listdir(evidence_dir):  # s1x outputs of other radar inputs
        if name.startswith('s1x_'):
            os.remove(os.path.join(evidence_dir, name))
    index = read_json(ctx.path('s1', 'index.json'))
    tracks = sorted({r['track'] for r in index})
    if not tracks:
        log('s1x: no radar track (fusion uses s1j)')
        ctx.summary['s1x'] = dict(tracks={})
        return
    grid = ctx.grid()
    hand = read_band(ctx.path('dem', 'hand10.tif'))
    slope = read_band(ctx.path('dem', 'slope10.tif'))
    calibration_sample = s1_change_test.calibration_sample(hand)
    del hand
    threads = min(16, s1_change_test.available_cpus())
    track_count = np.zeros((grid['height'], grid['width']), 'uint8')
    statistic = np.full(track_count.shape, np.nan, 'float32')
    info = {}
    for track in tracks:
        records = sorted([r for r in index if r['track'] == track], key=lambda r: r['day'])
        pre = [r for r in records if r['role'] == 'pre']
        post = [r for r in records if r['role'] == 'post'][:2]
        if len(pre) < 3:  # too few dates for the leave-one-out null
            decrease_path = ctx.path('evidence', f's1j_{track}_dec.tif')
            increase_path = ctx.path('evidence', f's1j_{track}_inc.tif')
            if os.path.exists(decrease_path) and os.path.exists(increase_path):
                s1j_flags = (read_band(decrease_path) == 1) | (read_band(increase_path) == 1)
                track_count += s1j_flags.astype('uint8')
            info[track] = dict(
                n_pre=len(pre), n_post=len(post), fallback='s1j' if os.path.exists(decrease_path) else 'none'
            )
            log('s1x', track, 'fewer than 3 pre dates: s1j flags kept for this track')
            continue
        neg_log_p, flags, info[track] = _s1x_track(ctx, track, pre, post, slope, calibration_sample, threads)
        track_count += flags.astype('uint8')
        statistic = np.fmax(statistic, neg_log_p)
        del neg_log_p, flags  # free before the next track's stacks are read
        log('s1x', track, info[track])
    ctx.write_raster('evidence/s1x_stat.tif', statistic, 'float32')
    ctx.write_raster('evidence/s1x_ntracks.tif', track_count, 'uint8')
    meta = dict(sig=signature, far_per_track=S1X_FAR, boxcar=S1X_BOXCAR_PX, threads=threads, tracks=info)
    write_json(meta_path, meta, indent=1)
    ctx.upload(meta_path)
    ctx.summary['s1x'] = dict(far_per_track=S1X_FAR, threads=threads, tracks=info)


S2_SEARCH_PRE_DAYS = 45
S2_SEARCH_POST_DAYS = 45
S2_PRE_DATES = 5
S2_POST_DATES = 4
S2_EARLY_DAYS = 12
S2_MIN_CLEAR_PRE = 0.15
S2_MIN_CLEAR_POST = 0.10
S2_CLEAR_CLASSES = (2, 4, 5, 6, 7)
S2_RES_M = 20
S2_BANDS = ('B02', 'B03', 'B04', 'B08', 'B11', 'B12')
S2_INDICES = ('ndvi', 'bsi', 'bright', 'mndwi', 'ndsi')


def _s2_clear_row(ctx, corridor_60m, day, items):
    """Clear fraction of the valley corridor and observed fraction of the area on one date (SCL at 60 m)."""
    ds = retry_io(
        lambda: odc.stac.load(
            items, bands=['SCL'], bbox=ctx.bbox, crs=ctx.crs, resolution=60, groupby='solar_day', resampling='nearest'
        ),
        f's2 scl {day}',
        delay_s=5,
    )
    scl = ds.SCL.values[0]
    height = min(scl.shape[0], corridor_60m.shape[0])
    width = min(scl.shape[1], corridor_60m.shape[1])
    scl, corridor = scl[:height, :width], corridor_60m[:height, :width]
    clear = np.isin(scl, S2_CLEAR_CLASSES)
    observed = scl > 0
    return dict(
        day=day, clear_corridor=float((clear & corridor).sum() / max(corridor.sum(), 1)), obs=float(observed.mean())
    )


def _s2_clear_table(ctx, items_by_day):
    corridor_60m = (read_band(ctx.path('dem', 'hand10.tif')) < 40)[::6, ::6]
    days = sorted(items_by_day)
    with ThreadPoolExecutor(8) as pool:  # I/O bound
        return list(pool.map(partial(_s2_clear_row, ctx, corridor_60m), days, [items_by_day[d] for d in days]))


def _select_s2_dates(ctx, table, acquisition_times):
    """Clearest pre-event and post-event dates, and the post-event dates within S2_EARLY_DAYS of the event."""
    event_date = dt.date.fromisoformat(ctx.event_date)

    def rows_with_role(role, min_clear):
        return [
            row
            for row in table
            if ctx.acquisition_role(acquisition_times[row['day']]) == role and row['clear_corridor'] > min_clear
        ]

    def clearest_first(rows):
        return [row['day'] for row in sorted(rows, key=lambda row: -row['clear_corridor'])]

    pre = clearest_first(rows_with_role('pre', S2_MIN_CLEAR_PRE))[:S2_PRE_DATES]
    post_rows = rows_with_role('post', S2_MIN_CLEAR_POST)
    post = clearest_first(post_rows)[:S2_POST_DATES]
    early_end = str(event_date + dt.timedelta(days=S2_EARLY_DAYS))
    early = sorted(row['day'] for row in post_rows if row['day'] <= early_end)
    return pre, post, early


def _load_s2_day(ctx, day, items):
    """Clear-sky reflectance of the bands, the snow mask and the transform of one date at S2_RES_M."""
    bands = [*S2_BANDS, 'SCL']
    ds = retry_io(
        lambda: odc.stac.load(
            items,
            bands=bands,
            bbox=ctx.bbox,
            crs=ctx.crs,
            resolution=S2_RES_M,
            groupby='solar_day',
            resampling={'SCL': 'nearest', '*': 'bilinear'},
            dtype='float32',
            nodata=0,
            pool=4,
        ),
        f's2 {day}',
        delay_s=10,
    )
    # processing baseline 04.00 and later adds 1000 to the digital numbers
    baseline = float(items[0].properties.get('s2:processing_baseline', '05.00'))
    offset = 1000.0 if baseline >= 4.0 else 0.0
    raw = {band: ds[band].values[0] for band in bands}
    clear = np.isin(raw['SCL'], S2_CLEAR_CLASSES)
    layers = {
        band: np.where(clear & (raw[band] > 0), (raw[band] - offset) / 1e4, np.nan).astype('float32')
        for band in S2_BANDS
    }
    layers['snow'] = raw['SCL'] == 11
    layers['transform'] = ds.odc.geobox.transform
    return layers


class S2DayLoader:
    """Loads Sentinel-2 dates for the composites; a date that several composites use is read once."""

    def __init__(self, ctx, items_by_day, plan):
        self._ctx = ctx
        self._items_by_day = items_by_day
        self._remaining_uses = collections.Counter(day for _, days, _ in plan for day in days)
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()

    def load(self, day):
        with self._lock:
            cached = self._cache.get(day)
            self._remaining_uses[day] -= 1
            if cached is not None and self._remaining_uses[day] <= 0:
                self._cache.pop(day, None)
        if cached is not None:
            return cached
        loaded = _load_s2_day(self._ctx, day, self._items_by_day[day])
        with self._lock:
            if self._remaining_uses[day] > 0:
                self._cache[day] = loaded
        return loaded


def _first_valid(layers):
    out = np.full_like(layers[0], np.nan)
    for layer in layers:
        out = np.where(np.isnan(out), layer, out)
    return out


def _s2_to_grid(ctx, layer, src_transform, is_mask):
    grid = ctx.grid()
    out = np.full((grid['height'], grid['width']), 0 if is_mask else np.nan, 'float32')
    reproject(
        layer.astype('float32'),
        out,
        src_transform=src_transform,
        src_crs=ctx.crs,
        dst_transform=grid['transform'],
        dst_crs=ctx.crs,
        resampling=Resampling.nearest if is_mask else Resampling.bilinear,
        src_nodata=None if is_mask else np.nan,
        dst_nodata=None if is_mask else np.nan,
    )
    return out > 0.5 if is_mask else out


def _s2_composite(ctx, loader, days, mode):
    """Combine clear pixels by their median or by the first clear observation in date order."""
    with ThreadPoolExecutor(4) as pool:
        loaded = list(pool.map(loader.load, days))
    log('s2 loaded', days)
    stacks = {band: [day_layers[band] for day_layers in loaded] for band in S2_BANDS}
    if mode == 'median':
        composite = {band: np.nanmedian(np.stack(layers), 0) for band, layers in stacks.items()}
    else:
        composite = {band: _first_valid(layers) for band, layers in stacks.items()}
    composite['snow'] = np.any(np.stack([day_layers['snow'] for day_layers in loaded]), 0)
    src_transform = loaded[0]['transform']
    return {name: _s2_to_grid(ctx, layer, src_transform, name == 'snow') for name, layer in composite.items()}


def s2_indices(bands):
    with np.errstate(divide='ignore', invalid='ignore'):
        return dict(
            ndvi=(bands['B08'] - bands['B04']) / (bands['B08'] + bands['B04']),
            mndwi=(bands['B03'] - bands['B11']) / (bands['B03'] + bands['B11']),
            bsi=((bands['B11'] + bands['B04']) - (bands['B08'] + bands['B02']))
            / ((bands['B11'] + bands['B04']) + (bands['B08'] + bands['B02'])),
            bright=np.sqrt((bands['B02'] ** 2 + bands['B03'] ** 2 + bands['B04'] ** 2) / 3),
            ndsi=(bands['B03'] - bands['B11']) / (bands['B03'] + bands['B11']),
        )


def _write_s2_composite(ctx, name, composite):
    """Indices, snow mask, int16 band stack (reflectance x 1e4, nodata -9999) and RGB quick look of a composite."""
    for index_name, values in s2_indices(composite).items():
        ctx.write_raster(f's2/{name}_{index_name}.tif', values.astype('float32'), 'float32')
    ctx.write_raster(f's2/{name}_snow.tif', composite['snow'].astype('uint8'), 'uint8')
    bands_path = ctx.path('s2', f'{name}_bands.tif')
    bands_profile = dict(
        ctx.grid(),
        count=len(S2_BANDS),
        dtype='int16',
        nodata=-9999,
        compress='deflate',
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    with rasterio.open(bands_path, 'w', **bands_profile) as dst:
        for band_number, band in enumerate(S2_BANDS, start=1):
            reflectance = composite[band]
            scaled = np.where(np.isfinite(reflectance), np.clip(reflectance * 1e4, -9998, 32767), -9999)
            dst.write(scaled.astype('int16'), band_number)
        dst.descriptions = S2_BANDS
    ctx.upload(bands_path)
    rgb_path = ctx.path('s2', f'{name}_rgb.tif')
    rgb = np.stack([composite['B04'], composite['B03'], composite['B02']])
    rgb_profile = dict(ctx.grid(), count=3, dtype='uint8', nodata=0, compress='deflate', tiled=True)
    with rasterio.open(rgb_path, 'w', **rgb_profile) as dst:
        dst.write(np.nan_to_num(np.clip(rgb / 0.25 * 254 + 1, 1, 255), nan=0).astype('uint8'))
    ctx.upload(rgb_path)


def step_s2(ctx):
    """Write pre-event, early and later optical composites, and two pre-event halves for change calibration."""
    if all(
        os.path.exists(ctx.path('s2', name))
        for name in ('post_ndvi.tif', 'pre_ndvi.tif', 'preB_bands.tif', 'clear_table.json')
    ):
        return
    event_date = dt.date.fromisoformat(ctx.event_date)
    window_end = min(event_date + dt.timedelta(days=S2_SEARCH_POST_DAYS), dt.date.today())
    window = f'{event_date - dt.timedelta(days=S2_SEARCH_PRE_DAYS)}/{window_end}'
    items = stac_catalog().search(collections=['sentinel-2-l2a'], bbox=ctx.bbox, datetime=window).item_collection()
    items_by_day: dict[str, list] = {}
    for item in items:
        items_by_day.setdefault(item.properties['datetime'][:10], []).append(item)
    table = _s2_clear_table(ctx, items_by_day)
    table_path = ctx.path('s2', 'clear_table.json')
    write_json(table_path, table, indent=1)
    ctx.upload(table_path)
    acquisition_times = {day: day_items[0].properties['datetime'][:19] for day, day_items in items_by_day.items()}
    pre, post, early = _select_s2_dates(ctx, table, acquisition_times)
    # the last-year keys stay empty: the seasonal composites are not built
    ctx.summary['s2_days'] = dict(pre=sorted(pre), post=sorted(post), early=early, preLY=[], postLY=[], earlyLY=[])
    log('s2 days', ctx.summary['s2_days'])
    pre_by_date = sorted(pre)
    half = len(pre_by_date) // 2
    halves = (
        [('preA', pre_by_date[:half], 'median'), ('preB', pre_by_date[half:], 'median')]
        if len(pre_by_date) >= 2
        else []
    )
    plan = [
        (name, days, mode)
        for name, days, mode in [('pre', pre, 'median'), ('post', post, 'median'), ('early', early, 'first'), *halves]
        if not os.path.exists(ctx.path('s2', f'{name}_rgb.tif'))
    ]
    loader = S2DayLoader(ctx, items_by_day, plan)
    for name, days, mode in plan:
        if days:
            _write_s2_composite(ctx, name, _s2_composite(ctx, loader, days, mode))


def _read_s2_indices(ctx, name):
    return {index_name: read_band(ctx.path('s2', f'{name}_{index_name}.tif')) for index_name in S2_INDICES}


def _write_s2_stable_water(ctx, pre_mndwi):
    """Mark water present before the event and in each clear image after it."""
    stable = np.isfinite(pre_mndwi) & (np.nan_to_num(pre_mndwi, nan=-1) > 0)
    seen_after = np.zeros(pre_mndwi.shape, bool)
    for name in ('post', 'early'):
        path = ctx.path('s2', f'{name}_mndwi.tif')
        if os.path.exists(path):
            mndwi = read_band(path)
            observed = np.isfinite(mndwi)
            stable &= np.where(observed, np.nan_to_num(mndwi, nan=-1) > 0, True)
            seen_after |= observed
    ctx.write_raster('evidence/s2_stable_water.tif', (stable & seen_after).astype('uint8'), 'uint8')


def _write_s2_rule_evidence(ctx, name, pre):
    """Write vegetation loss, bare-ground change and new water outside the snow mask."""
    post = _read_s2_indices(ctx, name)
    observed = np.isfinite(post['ndvi']) & np.isfinite(pre['ndvi'])
    with np.errstate(invalid='ignore'):
        debris = (post['ndvi'] - pre['ndvi'] < -0.15) & (
            (post['bsi'] - pre['bsi'] > 0) | (post['bright'] - pre['bright'] > 0.02)
        )
        new_water = (post['mndwi'] > 0) & (pre['mndwi'] < 0)
        # NDSI equals MNDWI; snow is also bright in the visible, water is dark
        snow = (
            ((pre['ndsi'] > 0.4) & (pre['bright'] > 0.25))
            | ((post['ndsi'] > 0.4) & (post['bright'] > 0.25))
            | (read_band(ctx.path('s2', f'{name}_snow.tif')) > 0)
            | (read_band(ctx.path('s2', 'pre_snow.tif')) > 0)
        )
        # vegetation loss means nothing where the ground was bare before
        bare_before = pre['ndvi'] < 0.2
    snow = ndi.binary_dilation(snow, iterations=3)
    change = ((debris & ~bare_before) | new_water) & ~snow
    ctx.write_raster(f'evidence/s2_{name}.tif', np.where(observed, change, 255), 'uint8', 255)
    ctx.write_raster(f'evidence/s2_{name}_wat.tif', np.where(observed, new_water & ~snow, 255), 'uint8', 255)


def step_s2_rules(ctx):
    """Fixed-threshold S2 change rules and the stable-water mask."""
    if not os.path.exists(ctx.path('s2', 'pre_ndvi.tif')):
        return
    pre = _read_s2_indices(ctx, 'pre')
    _write_s2_stable_water(ctx, pre['mndwi'])
    for name in ('post', 'early'):
        if os.path.exists(ctx.path('s2', f'{name}_ndvi.tif')):
            _write_s2_rule_evidence(ctx, name, pre)


S2_CAL_FAR = 0.01


def _read_s2_bands(ctx, name):
    return s2_calibration.read_bands(ctx.path('s2', f'{name}_bands.tif'))


def _s2_snow_mask(ctx, name, shape):
    """Snow or ice in the pre-event or `name` composite, dilated by 3 px."""
    mask = np.zeros(shape, bool)
    for composite in ('pre', name):
        bands = _read_s2_bands(ctx, composite)
        with np.errstate(invalid='ignore', divide='ignore'):
            ndsi = (bands['B03'] - bands['B11']) / (bands['B03'] + bands['B11'])
            brightness = np.sqrt((bands['B02'] ** 2 + bands['B03'] ** 2 + bands['B04'] ** 2) / 3)
        mask |= (ndsi > 0.4) & (brightness > 0.25)
        mask |= read_band(ctx.path('s2', f'{composite}_snow.tif')) > 0
    return ndi.binary_dilation(mask, iterations=3)


def step_s2_cal(ctx):
    """Calibrate optical change on the difference between the two pre-event composites."""
    if not os.path.exists(ctx.path('s2', 'preB_bands.tif')):
        log('s2_cal: no pre sub-composites -> skipped')
        return
    hand = read_band(ctx.path('dem', 'hand10.tif'))
    pre, pre_older, pre_newer = _read_s2_bands(ctx, 'pre'), _read_s2_bands(ctx, 'preA'), _read_s2_bands(ctx, 'preB')
    info = {}
    for name in ('post', 'early'):
        if not os.path.exists(ctx.path('s2', f'{name}_bands.tif')):
            continue
        change, info[name] = s2_calibration.detect(
            pre, _read_s2_bands(ctx, name), pre_older, pre_newer, hand, far=S2_CAL_FAR
        )
        change[(change == 1) & _s2_snow_mask(ctx, name, hand.shape)] = 0
        ctx.write_raster(f'evidence/s2cal_{name}.tif', change, 'uint8', 255)
    ctx.summary['s2_cal'] = info
    log('s2_cal', json.dumps(info)[:600])


def fetch_weights(run):
    """Folder of a model run; best.pt and config.json are fetched from the project store when missing."""
    folder = os.path.join(MODELS_DIR, run)
    os.makedirs(folder, exist_ok=True)
    for name in ('best.pt', 'config.json'):
        if not os.path.exists(os.path.join(folder, name)):
            store.get(f'models/{run}/{name}', os.path.join(folder, name))
    return folder


def _ai_device():
    import torch  # noqa: PLC0415 - optional dependency

    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _s1_linear(ctx, record):
    with rasterio.open(ctx.path('s1', os.path.basename(record['file']))) as src:
        return src.read(1), src.read(2)


def step_dl(ctx):
    """Predict radar flood probability per track, then average models and take the maximum across tracks."""
    if os.path.exists(ctx.path('evidence', 'dl_flood_prob.tif')) and os.path.exists(
        ctx.path('evidence', 'dl_config.json')
    ):
        return
    import radar_ai  # noqa: PLC0415 - radar_ai and terrain_channels import torch, which is optional
    import terrain_channels  # noqa: PLC0415

    device = _ai_device()
    tta = device == 'cuda'
    models = []
    for run in RADAR_AI_RUNS:
        folder = fetch_weights(run)
        model, config = radar_ai.load_model(
            os.path.join(folder, 'best.pt'), os.path.join(folder, 'config.json'), device=device
        )
        models.append((run, model, config))
    index = read_json(ctx.path('s1', 'index.json'))
    hand = read_band(ctx.path('dem', 'hand10.tif'))
    terrain_versions = sorted({int(config.get('terrain_v', 1)) for _, _, config in models if config.get('terrain')})
    terrain = {
        version: terrain_channels.channels_np(
            terrain_channels.fill_nodata(read_band(ctx.path('dem', 'dem10.tif'))), smooth=version >= 2
        )
        for version in terrain_versions
    }
    flood_max = dict.fromkeys(RADAR_AI_RUNS)
    permanent_max = dict.fromkeys(RADAR_AI_RUNS)
    for track in sorted({r['track'] for r in index}):
        records = sorted([r for r in index if r['track'] == track], key=lambda r: r['day'])
        pre = [r for r in records if r['role'] == 'pre']
        post = [r for r in records if r['role'] == 'post']
        stack = radar_ai.build_input(
            {'post': _s1_linear(ctx, post[0]), 'pre1': _s1_linear(ctx, pre[-1]), 'pre2': _s1_linear(ctx, pre[-2])},
            inc_deg=None,
            filt='lee7',
        )
        for run, model, config in models:
            prediction = radar_ai.predict(
                copy.deepcopy(model),  # AdaBN rewrites the batch-norm statistics of the model it is given
                stack,
                config['db_mean'],
                config['db_std'],
                device=device,
                tta=tta,
                diff=config.get('diff', False),
                adabn=True,
                adabn_mask=hand < 80,
                terrain=terrain.get(int(config.get('terrain_v', 1))) if config.get('terrain') else None,
                clip_db=radar_ai.CLIP_DB,
                adabn_valid=True,
            )
            flood, permanent = np.nan_to_num(prediction[2]), np.nan_to_num(prediction[1])
            flood_max[run] = flood if flood_max[run] is None else np.maximum(flood_max[run], flood)
            permanent_max[run] = permanent if permanent_max[run] is None else np.maximum(permanent_max[run], permanent)
        log('dl', track, 'done')
    if any(v is None for v in flood_max.values()):
        log('dl: no usable S1 track -> skipped')
        return
    flood = np.stack([flood_max[run] for run in RADAR_AI_RUNS])
    permanent = np.stack([permanent_max[run] for run in RADAR_AI_RUNS])
    ctx.write_raster('evidence/dl_flood_prob.tif', flood.mean(0).astype('float32'), 'float32')
    ctx.write_raster('evidence/dl_perm_prob.tif', permanent.mean(0).astype('float32'), 'float32')
    ctx.write_raster('evidence/dl_flood_disagreement.tif', (flood.max(0) - flood.min(0)).astype('float32'), 'float32')
    config_path = ctx.path('evidence', 'dl_config.json')
    write_json(
        config_path,
        dict(runs=RADAR_AI_RUNS, filter='lee7', adabn='v', fix=True, tta=tta, device=device, radiometry='sigma0'),
    )
    ctx.upload(config_path)
    ctx.summary['dl'] = dict(
        runs=RADAR_AI_RUNS, device=device, tta=tta, clip_db=list(radar_ai.CLIP_DB), adabn_valid=True
    )


def step_ais2(ctx):
    """Write optical water probability and the mask of water appearing after the event."""
    names = [name for name in ('pre', 'early', 'post') if os.path.exists(ctx.path('s2', f'{name}_bands.tif'))]
    if 'pre' not in names or len(names) < 2:
        log('ais2: no pre + post Sentinel-2 composites -> skipped')
        ctx.summary['ais2'] = dict(used=False, reason='no clear Sentinel-2 composite before and after the event')
        return
    band_paths = {name: ctx.path('s2', f'{name}_bands.tif') for name in names}
    signature = dict(
        run=OPTICAL_AI_RUN,
        inputs={name: [os.path.getsize(path), int(os.path.getmtime(path))] for name, path in band_paths.items()},
    )
    config_path = ctx.path('evidence', 'ais2_config.json')
    if (
        os.path.exists(config_path)
        and os.path.exists(ctx.path('evidence', 'ais2_new.tif'))
        and read_json(config_path)['sig'] == signature
    ):
        return
    import optical_ai  # noqa: PLC0415 - needs torch, which is optional

    device = _ai_device()
    tta = device == 'cuda'
    model = optical_ai.load_model(os.path.join(fetch_weights(OPTICAL_AI_RUN), 'best.pt'), device)
    hand = read_band(ctx.path('dem', 'hand10.tif'))
    valley_mask = ndi.binary_dilation(np.nan_to_num(hand, nan=1e6) < 80, iterations=5)  # valley floors and a margin
    del hand
    probability = {}
    for name, path in band_paths.items():
        bands, _ = optical_ai.read_bands(path)
        probability[name] = optical_ai.predict(model, bands, device=device, tta=tta, keep=valley_mask)
        del bands
        scaled = np.where(np.isfinite(probability[name]), np.clip(np.round(probability[name] * 250), 0, 250), 255)
        ctx.write_raster(f'evidence/ais2_{name}.tif', scaled.astype('uint8'), 'uint8', 255)
    first = 'early' if 'early' in probability else 'post'
    observed = np.isfinite(probability[first]) & np.isfinite(probability['pre'])
    new_water = (probability[first] > 0.5) & (probability['pre'] < 0.3) & observed
    ctx.write_raster('evidence/ais2_new.tif', np.where(observed, new_water, 255).astype('uint8'), 'uint8', 255)
    write_json(config_path, dict(sig=signature, device=device, tta=tta, first=first))
    ctx.upload(config_path)
    ctx.summary['ais2'] = dict(
        used=True,
        run=OPTICAL_AI_RUN,
        device=device,
        tta=tta,
        composites=names,
        first_image=first,
        new_water_km2=float(new_water.sum() / 1e4),
    )
    log('ais2', ctx.summary['ais2'])


def _joint_s1_evidence(ctx):
    """Return the statistical radar mask when it matches the current acquisitions, or the joint-rule mask."""
    ctx.clear_warnings(S1X_STALE)
    s1x_path = ctx.path('evidence', 's1x_ntracks.tif')
    if os.path.exists(s1x_path):
        meta_path = ctx.path('evidence', 's1x_meta.json')
        if os.path.exists(meta_path) and read_json(meta_path)['sig'] == s1x_signature(ctx):
            return read_band(s1x_path) >= 1, 's1x'
        ctx.warn(f'{S1X_STALE}; the previous radar rule (s1j) is used.')
    s1j_path = ctx.path('evidence', 's1j_ntracks.tif')
    if os.path.exists(s1j_path):
        return read_band(s1j_path) >= 1, 's1j'
    return None, 'none'


def _fusion_layers(ctx, s1_tracks, s1_joint):
    """Evidence layers for fusion.fuse; uint8 S2 layers use 255 for not observed."""
    blank = np.full(s1_tracks.shape, 255, 'uint8')

    def optional(name, default):
        layer = read_optional(ctx, 'evidence', name)
        return default if layer is None else layer

    dl = read_optional(ctx, 'evidence', 'dl_flood_prob.tif')
    layers = dict(
        hand=read_band(ctx.path('dem', 'hand10.tif')),
        acc=read_band(ctx.path('dem', 'acc10.tif')),
        slope=read_band(ctx.path('dem', 'slope10.tif')),
        s1=s1_tracks >= 1,
        s2p=optional('s2_post.tif', blank),
        s2e=optional('s2_early.tif', blank),
        dl=np.zeros(s1_tracks.shape, 'float32') if dl is None else np.nan_to_num(dl, nan=0),
    )
    if s1_joint is not None:
        layers['s1j'] = s1_joint
    for key, parts in (
        ('s2e_wat', ('evidence', 's2_early_wat.tif')),
        ('stable_wat', ('evidence', 's2_stable_water.tif')),
        ('pre_ndvi', ('s2', 'pre_ndvi.tif')),
    ):
        layer = read_optional(ctx, *parts)
        if layer is not None:
            layers[key] = layer
    calibrated_post = read_optional(ctx, 'evidence', 's2cal_post.tif')
    if calibrated_post is not None:
        layers['s2cal_p'] = calibrated_post
        layers['s2cal_e'] = optional('s2cal_early.tif', blank)
    return layers


def _add_optical_ai(ctx, layers):
    """Add optical new water to the early-water mask and the AI water mask used on plains."""
    ai_early_water_path = ctx.path('evidence', 's2_early_wat_ai.tif')
    if os.path.exists(ai_early_water_path):  # a copy from an earlier run must not reach the impact step
        os.remove(ai_early_water_path)
    ai_new = None
    new_water = read_optional(ctx, 'evidence', 'ais2_new.tif')
    if new_water is not None:
        early_water = layers.get('s2e_wat', np.full(new_water.shape, 255, 'uint8'))
        layers['s2e_wat'] = np.where(
            new_water == 1, 1, np.where((early_water == 255) & (new_water != 255), 0, early_water)
        ).astype('uint8')
        ai_new = new_water == 1
        ctx.write_raster('evidence/s2_early_wat_ai.tif', layers['s2e_wat'], 'uint8', 255)
    before = read_optional(ctx, 'evidence', 'ais2_pre.tif')
    if before is None:
        return ai_new
    any_new = np.zeros(before.shape, bool)
    for name in ('early', 'post'):
        after = read_optional(ctx, 'evidence', f'ais2_{name}.tif')
        if after is not None:
            # probabilities are stored x 250: p > 0.5 after, p < 0.3 before
            any_new |= (after != 255) & (before != 255) & (after > 125) & (before < 75)
    layers['ai_any'] = any_new
    return any_new if ai_new is None else (ai_new | any_new)


def _continuity_fill(ctx, layers, masks, s1_observed, likely, flat_deg):
    """Fill cloud-covered valley ground to the estimated flood level and write the level raster."""
    ctx.clear_warnings('River-continuity fill')
    try:
        with rasterio.open(ctx.path('dem', 'fdir30.tif')) as src:
            flow_direction_30m, transform_30m, crs_30m = src.read(1), src.transform, src.crs
        flat = ndi.binary_dilation(fusion.flat_mask(layers['slope'], flat_deg), iterations=3)
        fill, stage_m, stage_info = reach_stage.gap_fill(
            layers,
            masks,
            s1_observed,
            flow_direction_30m,
            read_band(ctx.path('dem', 'acc30.tif')),
            transform_30m,
            crs_30m,
            ctx.grid()['transform'],
            ctx.crs,
            likely,
            flat,
        )
        ctx.write_raster('hazard/stage_m.tif', stage_m, 'float32')
    except Exception as error:  # noqa: BLE001 - the map stands without the fill; the report shows the warning
        ctx.warn(
            f'River-continuity fill was not applied ({type(error).__name__}: {str(error)[:120]}); cloud-covered '
            'valley sections stay unmapped.'
        )
        return np.zeros(likely.shape, bool), None
    return fill, stage_info


def step_fusion(ctx):
    """Write the flood extent, observation mask and contributing sources."""
    params = dict(fusion.PARAMS)
    s1_tracks = read_band(ctx.path('evidence', 's1_ntracks.tif'))
    s1_observed = read_band(ctx.path('evidence', 's1_observed.tif')) == 1
    s1_joint, s1_source = _joint_s1_evidence(ctx)
    layers = _fusion_layers(ctx, s1_tracks, s1_joint)
    ai_new = _add_optical_ai(ctx, layers)
    likely, possible, sources, info = fusion.fuse(layers)
    masks = info.pop('_masks')
    fill, stage_info = _continuity_fill(ctx, layers, masks, s1_observed, likely, params['flat_deg'])
    likely = likely | fill
    possible = possible & ~likely
    sources = sources.astype('uint8')
    if ai_new is not None:
        sources = (sources | np.where(ai_new & likely, 16, 0)).astype('uint8')
    sources = (sources | np.where(fill, 32, 0)).astype('uint8')
    # observed: a radar view without layover/shadow on some track, or a clear optical image; likely is never hidden
    geometry_uncertain = read_band(ctx.path('evidence', 's1_geom_uncertain.tif')) == 1
    observed = (s1_observed & ~geometry_uncertain) | (layers['s2p'] != 255) | (layers['s2e'] != 255)
    hazard = np.where(likely, 2, np.where(possible, 1, 0)).astype('uint8')
    hazard[~observed & ~likely] = 255
    ctx.write_raster('hazard/hazard.tif', hazard, 'uint8', 255)
    ctx.write_raster('hazard/sources.tif', sources, 'uint8')
    likely_by_source = {name: float(((sources >> bit) & 1).sum() / 1e4) for bit, name in enumerate(['s1', 's2', 'dl'])}
    likely_by_source['ai_optical'] = float(((sources & 16) > 0).sum() / 1e4)
    likely_by_source['continuity'] = float(fill.sum() / 1e4)
    ctx.summary['hazard_km2'] = dict(
        likely=float(likely.sum() / 1e4),
        possible=float(possible.sum() / 1e4),
        unobserved=float((~observed).sum() / 1e4),
        likely_by_source=likely_by_source,
    )
    if stage_info is not None:
        ctx.summary['stage'] = stage_info
    ctx.summary['fusion_params'] = dict(
        params,
        dl_available=bool(os.path.exists(ctx.path('evidence', 'dl_flood_prob.tif'))),
        s1_source=s1_source,
        ai_optical=bool(ai_new is not None),
        stage_fill=bool(stage_info is not None),
        **info,
    )
    log('hazard', ctx.summary['hazard_km2'])


OSM_AREA_LAYERS = ['roads', 'buildings', 'bridges', 'places', 'health', 'waterways']
NET_BUFFER_KM = 15.0
NET_OSM_BUDGET_S = 600.0
OSM_SOURCE_ERRORS = (RuntimeError, OSError, ValueError)


def _fetch_osm_overpass(ctx, network_bbox, cutoff, has_area, has_network):
    """Missing area and network layers from Overpass/ohsome attic queries at the cutoff time."""
    osm_dir = ctx.path('osm')
    snapshot_utc = cutoff.strftime('%Y-%m-%dT%H:%M:%SZ')
    if not has_area:
        ctx.summary['osm_snapshot'] = f'overpass/ohsome {snapshot_utc}'
        osm_fetch.fetch_osm(
            tuple(ctx.bbox),
            snapshot_utc,
            osm_dir,
            layers=OSM_AREA_LAYERS,
            prefer='ohsome',
            overpass_timeout=180,
            overpass_rounds=1,
            budget_s=float(os.environ.get('TRI_OSM_BUDGET', '900')),
        )
    if has_network:
        return
    try:
        osm_fetch.fetch_osm(
            tuple(network_bbox),
            snapshot_utc,
            osm_dir,
            layers=['roads', 'places', 'health', 'admin'],
            prefix='net_',
            prefer='ohsome',
            overpass_timeout=180,
            overpass_rounds=1,
            budget_s=NET_OSM_BUDGET_S,
            optional=('places', 'health', 'admin'),
        )
    except OSM_SOURCE_ERRORS as error:
        log('network OSM failed', repr(error)[:300])
        ctx.warn(
            f'OSM: the road network around the area (+{NET_BUFFER_KM:g} km) could not be fetched; the cut-off '
            'analysis uses only the roads inside the area, so settlements near its edge may be wrongly shown as cut '
            'off.'
        )
        return
    write_json(
        ctx.path('osm', 'net_fetch_log.json'),
        dict(
            source='overpass/ohsome',
            snapshot_utc=snapshot_utc,
            extent=network_bbox,
            bbox=list(ctx.bbox),
            buffer_km=NET_BUFFER_KM,
        ),
        indent=1,
    )


def _fetch_osm(ctx, network_bbox, cutoff, has_area, has_network):
    """Missing area and network layers: the Geofabrik dated extract, else Overpass/ohsome."""
    fetch_log_path = ctx.path('osm', 'osm_fetch_log.json')
    # the network layers come from the dated extract of the area layers when possible
    extract_date = (
        read_json(fetch_log_path).get('extract_date') if has_area and os.path.exists(fetch_log_path) else None
    )
    try:
        _, fetch_log = osm_fetch.fetch_osm_geofabrik(
            ctx.bbox,
            ctx.event_date,
            ctx.path('osm'),
            cache_dir=os.path.join(os.path.dirname(ctx.out_dir), '_osm_cache'),
            net_bbox=None if has_network else network_bbox,
            write_aoi=not has_area,
            cutoff_utc=cutoff.strftime('%Y-%m-%dT%H:%M:%S'),
            want_date=extract_date,
        )
    except OSM_SOURCE_ERRORS as error:
        log('geofabrik failed', repr(error)[:300])
        _fetch_osm_overpass(ctx, network_bbox, cutoff, has_area, has_network)
        return
    if not has_area:
        ctx.summary['osm_snapshot'] = f'geofabrik {fetch_log["extract_date"]} ({fetch_log["source"]})'
    for warning in fetch_log.get('warnings', []):
        ctx.warn(warning)


def step_osm(ctx):
    """Fetch pre-event OSM for the area and the buffered road network, and record both snapshot times."""
    network_bbox = osm_fetch.net_bbox(ctx.bbox, NET_BUFFER_KM)
    has_area = all(os.path.exists(ctx.path('osm', f'{layer}.gpkg')) for layer in ['roads', 'buildings', 'places'])
    has_network = os.path.exists(ctx.path('osm', 'net_fetch_log.json'))
    if has_area and has_network and os.path.exists(ctx.path('osm', 'snapshot.json')):
        return
    ctx.clear_warnings('OSM:')
    # one day of margin: a date-only event may start early on its local day
    cutoff = ctx.event_time - dt.timedelta(days=1)
    if not (has_area and has_network):
        _fetch_osm(ctx, network_bbox, cutoff, has_area, has_network)
    osm_dir = ctx.path('osm')
    os.makedirs(osm_dir, exist_ok=True)
    record = osm_fetch.snapshot_record(osm_dir)
    record.update(
        event_utc=ctx.event_utc,
        cutoff_utc=cutoff.strftime('%Y-%m-%dT%H:%M:%S'),
        net_buffer_km=NET_BUFFER_KM,
        net_bbox=network_bbox,
    )
    write_json(ctx.path('osm', 'snapshot.json'), record, indent=1)
    network_record = record.get('net') or {}
    ctx.summary['osm_net'] = dict(
        buffer_km=NET_BUFFER_KM,
        bbox=network_bbox,
        snapshot=network_record.get('snapshot_utc'),
        source=network_record.get('source'),
    )
    for name in sorted(os.listdir(osm_dir)):
        if name.endswith(('.gpkg', '.json')):
            ctx.upload(os.path.join(osm_dir, name))


IMPACT_PARAMS = dict(
    dl_only_possible=False,
    bld_buffer_px=1,
    road_run_m=50.0,
    # cut-off analysis: snap distances to the motorable and foot networks, settlement catchment radius,
    # long-detour rule (after > factor x before + 1 km), number of ranked repairs
    snap_veh_m=500.0,
    snap_foot_m=500.0,
    unit_m=1500.0,
    detour_factor=1.5,
    n_repairs=5,
)
ROAD_SEGMENT_M = 50.0
OSM_MISSING = 'OSM layer missing (OSM step failed or not run)'
SETTLEMENT_COLUMNS = [
    'sid',
    'label',
    'name',
    'name:en',
    'name:ne',
    'place',
    'status',
    'tier',
    'access',
    'named',
    'near',
    'unit',
    'in_aoi',
    'cross_border',
    'dist_pre_km',
    'dist_post_km',
    'detour_km',
    'detour_h',
    'walk_km',
    'walk_h',
    'target',
    'exit_pre',
    'exit_post',
    'svc_post',
    'lost_exit_only',
    'trail_cut',
    'snap_dist_m',
    'adm_muni',
    'adm_ward',
    'via_roadhead',
    'unverified',
]


def _read_osm_layer(ctx, layer):
    path = ctx.path('osm', f'{layer}.gpkg')
    return gpd.read_file(path).to_crs(ctx.crs) if os.path.exists(path) else None


def _first_nonempty_osm_layer(ctx, *layers):
    for layer in layers:
        frame = _read_osm_layer(ctx, layer)
        if frame is not None and len(frame):
            return frame
    return None


def _lines_only(frame):
    return frame[frame.geom_type.isin(['LineString', 'MultiLineString'])]


def _bbox_polygon(bbox, to_utm):
    """The bbox as a UTM polygon, 41 points per edge."""
    west, south, east, north = bbox
    s = np.linspace(0, 1, 41)
    lon = np.r_[west + s * (east - west), np.full(41, east), east - s * (east - west), np.full(41, west)]
    lat = np.r_[np.full(41, south), south + s * (north - south), np.full(41, north), north - s * (north - south)]
    x, y = to_utm.transform(lon, lat)
    return Polygon(zip(x, y, strict=True))


def _assess_buildings(ctx, hazard_classes, observed, transform):
    """Return building exposure, counts and availability; absent building data give None counts."""
    buildings = _read_osm_layer(ctx, 'buildings')
    if buildings is None:
        return None, None, OSM_MISSING
    if not len(buildings):
        return None, None, 'no buildings mapped in OSM for this area'
    assessed = impact.assess_buildings(
        buildings, hazard_classes, observed, transform, buffer_px=IMPACT_PARAMS['bld_buffer_px']
    )
    path = ctx.path('impact', 'buildings_affected.gpkg')
    assessed[assessed['status'] != 'not_affected'].to_file(path, driver='GPKG')
    ctx.upload(path)
    counts = {str(k): int(v) for k, v in assessed['status'].value_counts().items()}
    return assessed, counts, 'ok'


def _strict_source_bits(ctx, sources):
    """Calculate road-break evidence from the radar and optical layers used in the flood map."""
    fusion_params = dict(fusion.PARAMS, **(ctx.summary.get('fusion_params') or {}))
    s1_tracks_file = (
        's1x_ntracks.tif'
        if fusion_params.get('s1_source') == 's1x' and os.path.exists(ctx.path('evidence', 's1x_ntracks.tif'))
        else 's1_ntracks.tif'
    )
    early_water_file = (
        's2_early_wat_ai.tif'
        if fusion_params.get('ai_optical') and os.path.exists(ctx.path('evidence', 's2_early_wat_ai.tif'))
        else 's2_early_wat.tif'
    )
    impact_params = ctx.summary['impact_params']
    impact_params['evidence_files'] = dict(s1_tracks=s1_tracks_file, early_water=early_water_file)
    strict_sources = impact.strict_source_bits(
        sources,
        fusion_params,
        s2p=read_optional(ctx, 'evidence', 's2_post.tif'),
        s2e=read_optional(ctx, 'evidence', 's2_early.tif'),
        s2cal_p=read_optional(ctx, 'evidence', 's2cal_post.tif'),
        s2cal_e=read_optional(ctx, 'evidence', 's2cal_early.tif'),
        pre_ndvi=read_optional(ctx, 's2', 'pre_ndvi.tif'),
        s2e_wat=read_optional(ctx, 'evidence', early_water_file),
        dl=read_optional(ctx, 'evidence', 'dl_flood_prob.tif'),
        slope=read_optional(ctx, 'dem', 'slope10.tif'),
        s1n=read_optional(ctx, 'evidence', s1_tracks_file),
    )
    impact_params['strong_rule'] = (
        'persistent S2, S1 on >= 2 tracks, or S1 and S2 (DL not counted: same S1 images)'
        if impact.ACCESS_RULES.get('strong_rule') == 'v2'
        else 'persistent S2 change or >= 2 sources; DL at the fusion thresholds'
    )
    return strict_sources


def _assess_area_roads(ctx, assess_roads):
    """Road segments of the area in the hazard footprint. Returns (segments or None, summary entries, availability)."""
    roads = _read_osm_layer(ctx, 'roads')
    if roads is None:
        return None, {}, OSM_MISSING
    if not len(_lines_only(roads)):
        return None, {}, 'no roads mapped in OSM for this area'
    segments = assess_roads(impact.explode_segments(_lines_only(roads), ROAD_SEGMENT_M))
    segments['km'] = segments.length / 1000
    path = ctx.path('impact', 'road_segments_affected.gpkg')
    segments[segments['status'] != 'open'].to_file(path, driver='GPKG')
    ctx.upload(path)
    return segments, impact.roads_summary(segments), 'ok'


def _write_cutoff_layers(ctx, result, settlements):
    layers = dict(
        settlements_vehicle=settlements,
        settlement_units=result['units'],
        access_routes=result['routes'],
        access_breaches=result['breaks'],
        break_runs=result['runs'],
        repairs=result['repairs'],
        health_status=result['health'],
        net_exits=result['exits'],
        buildings_access=result['bld_access'],
    )
    for name, frame in layers.items():
        path = ctx.path('impact', f'{name}.gpkg')
        if frame is not None and len(frame):
            frame.to_file(path, driver='GPKG')
            ctx.upload(path)
        elif os.path.exists(path):
            os.remove(path)  # an empty layer must not leave the file of an earlier run


def _add_lat_lon(summary, repairs, crs):
    to_lon_lat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    for key in ('priority', 'hardest_hit'):
        for row in summary.get(key, []):
            lon, lat = to_lon_lat.transform(row['x'], row['y'])
            row['lat'], row['lon'] = round(lat, 5), round(lon, 5)
    if repairs is not None:
        for row, point in zip(summary.get('repairs', []), repairs.geometry, strict=False):
            lon, lat = to_lon_lat.transform(point.x, point.y)
            row['lat'], row['lon'] = round(lat, 5), round(lon, 5)


def _labels_by_buildings_without_road(settlements, buildings_without_road):
    """Settlement labels, most buildings without road access first; at most 200."""
    counts = [buildings_without_road.get(sid, 0) for sid in settlements['sid']]
    return settlements.assign(_n=counts).sort_values('_n', ascending=False)['label'].astype(str).tolist()[:200]


def _cutoff_analysis(ctx, segments, buildings, assess_roads, hazard_classes, transform):
    """Calculate access on the buffered road network when available, and write routes, breaks and repairs."""
    to_utm = Transformer.from_crs('EPSG:4326', ctx.crs, always_xy=True)
    aoi_polygon = _bbox_polygon(ctx.bbox, to_utm)
    net_log_path = ctx.path('osm', 'net_fetch_log.json')
    net_log = read_json(net_log_path) if os.path.exists(net_log_path) else {}
    net_roads = _read_osm_layer(ctx, 'net_roads')
    if net_roads is not None and len(_lines_only(net_roads)) and net_log.get('extent'):
        net_segments = assess_roads(impact.explode_segments(_lines_only(net_roads), ROAD_SEGMENT_M))
        net_polygon = _bbox_polygon(net_log['extent'], to_utm)
        net_source = 'buffered'
    else:
        net_segments, net_polygon, net_source = segments, aoi_polygon, 'aoi'
    with rasterio.open(ctx.path('dem', 'dem30_buf.tif')) as dem:
        net_segments = impact.add_walk_time(net_segments, dem.read(1), dem.transform)
    is_buffered = net_source == 'buffered'
    places = _read_osm_layer(ctx, 'places')
    result = impact.road_access(
        net_segments,
        places,
        net_places=_first_nonempty_osm_layer(ctx, 'net_places', 'places') if is_buffered else places,
        health=_first_nonempty_osm_layer(ctx, 'net_health', 'health')
        if is_buffered
        else _read_osm_layer(ctx, 'health'),
        bld=buildings,
        hazard=hazard_classes,
        transform=transform,
        aoi_poly=aoi_polygon,
        net_poly=net_polygon,
        admin=_first_nonempty_osm_layer(ctx, 'net_admin', 'admin'),
        snap_veh=IMPACT_PARAMS['snap_veh_m'],
        snap_foot=IMPACT_PARAMS['snap_foot_m'],
        unit_m=IMPACT_PARAMS['unit_m'],
        detour_factor=IMPACT_PARAMS['detour_factor'],
        n_repairs=IMPACT_PARAMS['n_repairs'],
        net_source=net_source,
    )
    settlements = result['settlements'].copy()
    settlements['geometry'] = settlements.geometry.representative_point()
    settlements = settlements[[k for k in SETTLEMENT_COLUMNS if k in settlements] + ['geometry']]
    _write_cutoff_layers(ctx, result, settlements)
    summary = result['summary']
    _add_lat_lon(summary, result['repairs'], ctx.crs)
    buffer_km = net_log.get('buffer_km') or (ctx.summary.get('osm_net') or {}).get('buffer_km')
    summary['net']['buffer_km'] = buffer_km if is_buffered else 0
    summary['net']['snapshot'] = net_log.get('snapshot_utc') or net_log.get('replication_timestamp')
    for warning in summary.get('warnings', []):
        ctx.warn('Cut-off analysis: ' + warning)
    named = settlements[
        settlements['in_aoi'].astype(bool)
        & ~settlements['cross_border'].astype(bool)
        & settlements['named'].astype(bool)
    ]
    units = result['units']
    buildings_without_road = {}
    if units is not None:
        is_unit = units['uid'].str.startswith('U')
        buildings_without_road = dict(
            zip(units.loc[is_unit, 'uid'].str[1:], units.loc[is_unit, 'n_noroad'], strict=True)
        )
    possibly_cut_off = named['status'].isin(['possibly_cut_off', 'bridge_uncertain'])
    return dict(
        q3=summary,
        settlements_vehicle=summary['settlements']['by_status'],
        graph_vehicle=summary['net'],
        cut_off_vehicle=_labels_by_buildings_without_road(named[named['status'] == 'cut_off'], buildings_without_road),
        possibly_cut_off_vehicle=_labels_by_buildings_without_road(named[possibly_cut_off], buildings_without_road),
        access_routes=0 if result['routes'] is None else len(result['routes']),
    )


def step_impacts(ctx):
    """Calculate infrastructure exposure and road access; reject OSM snapshots containing post-event edits."""
    ctx.clear_warnings('OSM snapshot')
    ctx.clear_warnings('Cut-off analysis:')
    is_pre_event, message = osm_fetch.check_snapshot(ctx.path('osm'), ctx.event_utc)
    if message:
        ctx.warn(message)
    if not is_pre_event:
        raise RuntimeError(message)
    transform = ctx.grid()['transform']
    hazard = read_band(ctx.path('hazard', 'hazard.tif'))
    observed = hazard != 255
    hazard_classes = np.where(observed, hazard, 0).astype('uint8')
    ctx.summary['impact_params'] = dict(IMPACT_PARAMS)
    results = {}
    availability = {}
    buildings, building_counts, availability['buildings'] = _assess_buildings(ctx, hazard_classes, observed, transform)
    if building_counts is not None:
        results['buildings'] = building_counts
    sources = read_band(ctx.path('hazard', 'sources.tif'))
    strict_sources = _strict_source_bits(ctx, sources)
    ctx.summary['impact_params']['q3'] = dict(impact.ACCESS_RULES)

    def assess_roads(roads):
        return impact.assess_roads_grid(
            roads,
            hazard_classes,
            observed,
            transform,
            sources=sources,
            dl_only_possible=IMPACT_PARAMS['dl_only_possible'],
            run_m=IMPACT_PARAMS['road_run_m'],
            strict_sources=strict_sources,
        )

    segments, road_results, availability['roads'] = _assess_area_roads(ctx, assess_roads)
    results.update(road_results)
    if segments is None:
        availability['q3'] = availability['roads']
    else:
        try:
            results.update(_cutoff_analysis(ctx, segments, buildings, assess_roads, hazard_classes, transform))
            availability['q3'] = (
                'ok' if results['q3']['assessed'] else 'no town, hospital or road out of the area in the road network'
            )
        except Exception as error:  # noqa: BLE001 - the damage results stand without it; the report shows the reason
            traceback.print_exc()
            availability['q3'] = f'cut-off analysis failed ({type(error).__name__}: {str(error)[:150]})'
            ctx.warn('Cut-off analysis: ' + availability['q3'] + '; cut-off results are not available.')
    results['availability'] = availability
    ctx.summary['impacts'] = impact.jsonable(results)
    log('impacts', json.dumps(ctx.summary['impacts'], default=str)[:3000])


FLOWPATH_TIERS_M = (15.0, 40.0)


def step_flowpath(ctx):
    """Trace drainage below the supplied upstream point and list settlements along the path."""
    source = ctx.flowpath_source
    if not source:
        return
    places_path = ctx.path('osm', 'places.gpkg')
    places = gpd.read_file(places_path) if os.path.exists(places_path) else None
    flow_direction_path = ctx.path('dem', 'fdir30.tif')
    result = flowpath.trace_flood_path(
        source[0],
        source[1],
        ctx.path('dem', 'demfill30.tif'),
        flow_direction_path,
        ctx.path('dem', 'acc30.tif'),
        places=places,
        tiers=FLOWPATH_TIERS_M,
        crs=ctx.crs,
    )
    path_file = ctx.path('flowpath', 'path.geojson')
    result['path'].to_crs(4326).to_file(path_file, driver='GeoJSON')
    ctx.upload(path_file)
    with rasterio.open(flow_direction_path) as src:
        profile = dict(src.profile, dtype='float32', nodata=np.nan, compress='deflate')
    height_m = result['hand']
    height_m = np.where(height_m <= FLOWPATH_TIERS_M[1], height_m, np.nan).astype('float32')
    corridor_file = ctx.path('flowpath', 'corridor_height_above_path.tif')
    with rasterio.open(corridor_file, 'w', **profile) as dst:
        dst.write(height_m, 1)
    ctx.upload(corridor_file)
    settlements = result.get('settlements')
    if settlements is not None and len(settlements):
        columns = [
            k
            for k in ['name', 'place', 'km_downstream', 'h_above_path_m', 'dist_to_path_m', 'tier', 'geometry']
            if k in settlements
        ]
        settlements_file = ctx.path('flowpath', 'settlements_along_path.geojson')
        settlements[columns].to_crs(4326).to_file(settlements_file, driver='GeoJSON')
        ctx.upload(settlements_file)
    ctx.summary['flowpath'] = dict(
        source=source,
        length_km=float(result['path']['length_km'].iloc[0]),
        drop_m=float(result['path']['drop_m'].iloc[0]),
        n_settlements=int(0 if settlements is None else len(settlements)),
        settlements=[]
        if settlements is None
        else [
            dict(name=str(row.get('name')), km=round(float(row['km_downstream']), 1), tier=row['tier'])
            for _, row in settlements.head(60).iterrows()
        ],
    )
    log('flowpath', json.dumps(ctx.summary['flowpath'])[:1500])


def step_report(ctx):
    report.build(ctx.out_dir, ctx.event, ctx.should_push)


def check_data_availability(ctx):
    """Record available radar tracks and clear optical dates, and identify runs without observations."""
    index_path = ctx.path('s1', 'index.json')
    index = read_json(index_path) if os.path.exists(index_path) else []
    tracks = sorted({r['track'] for r in index if r['role'] == 'post'})
    s2_days = ctx.summary.get('s2_days') or {}
    s2_post_dates = len(s2_days.get('post') or []) + len(s2_days.get('early') or [])
    warnings = []
    if not tracks:
        warnings.append(
            'No Sentinel-1 radar pair (same track before and after) was found for this area and date: '
            'Sentinel-1 data start in October 2014, and a post-event pass may not exist yet for a very recent date.'
        )
    if not s2_post_dates:
        warnings.append(
            'No clear Sentinel-2 image after the event (clouds or no acquisition): the optical evidence is missing.'
        )
    if not (s2_days.get('pre') or []):
        warnings.append('No clear Sentinel-2 image before the event: optical change cannot be computed.')
    ctx.summary['data_availability'] = dict(
        s1_tracks=tracks,
        s2_post_dates=s2_post_dates,
        s2_pre_dates=len(s2_days.get('pre') or []),
        warnings=warnings,
    )
    ctx.summary['no_observation'] = bool(not tracks and not s2_post_dates)
    for warning in warnings:
        log('DATA WARNING:', warning)


CODE_FILES = (
    'pipeline.py',
    's1_grd.py',
    's1_change_rule.py',
    's1_change_test.py',
    's1_geometry.py',
    's2_calibration.py',
    'fusion.py',
    'reach_stage.py',
    'radar_ai.py',
    'optical_ai.py',
    'terrain_channels.py',
    'impact.py',
    'osm_fetch.py',
    'flowpath.py',
    'report.py',
    'dashboard_ui.py',
    'sitrep_nepali.py',
)
SUMMARY_META = ('event', 'bbox', 'date', 'event_time', 'date_given')
WARN_KM2 = 2500
MAX_KM2 = 12000
MIN_GRID_PX = 256
SMALL_BBOX = 'Small area'
STEPS = (
    ('dem', step_dem),
    ('s1', step_s1),
    ('geometry', step_geometry),
    ('s1_rules', step_s1_rules),
    ('s1_joint', step_s1_joint),
    ('s1x', step_s1x),
    ('s2', step_s2),
    ('s2_rules', step_s2_rules),
    ('s2_cal', step_s2_cal),
    ('dl', step_dl),
    ('ais2', step_ais2),
    ('fusion', step_fusion),
    ('osm', step_osm),
    ('impacts', step_impacts),
    ('flowpath', step_flowpath),
    ('report', step_report),
)
FATAL_STEPS = ('dem', 'fusion')  # without a grid or a hazard map nothing downstream is meaningful
STEP_FAILURE_NOTES = {
    'dem': 'Without the elevation model nothing can be computed; the run stopped.',
    's1': 'Radar (Sentinel-1) is not used: the map relies on Sentinel-2 optical evidence only, so cloud-covered ground '
    'is unobserved, and the AI model (which reads radar) is not used.',
    'geometry': 'Radar layover/shadow masks are not available; the flood map is still built.',
    's1_rules': 'Radar change evidence is missing.',
    's1_joint': 'The joint radar test is not used.',
    's1x': 'The improved radar change test (s1x) failed; the map uses the previous radar rule (s1j) for the radar '
    'evidence.',
    's2': 'Optical (Sentinel-2) evidence is missing or incomplete.',
    's2_rules': 'Optical change rules were not applied.',
    's2_cal': 'The calibrated optical detector was not used.',
    'dl': 'The AI flood model was not used: the map relies on the rule-based radar and optical evidence.',
    'ais2': 'The optical water model water model was not used: optical evidence comes from the change rules only.',
    'fusion': 'No hazard map could be built; the run stopped.',
    'osm': 'Pre-event OpenStreetMap data could not be fetched: damage and cut-off results are not available.',
    'impacts': 'Damage and cut-off results are not available or incomplete.',
    'flowpath': 'Flood-path tracing is not available.',
    'report': 'The dashboard / situation report may be missing or incomplete.',
}


def code_version():
    """Most recent modification time of the mapping code."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    modified = max(os.path.getmtime(os.path.join(src_dir, name)) for name in CODE_FILES)
    return dt.datetime.fromtimestamp(modified, dt.timezone.utc).isoformat(timespec='seconds')


def bbox_size_km(bbox):
    """(width km, height km, area km2) of a lon/lat bbox (W, S, E, N)."""
    width_km = (bbox[2] - bbox[0]) * 111.32 * math.cos(math.radians((bbox[1] + bbox[3]) / 2))
    height_km = (bbox[3] - bbox[1]) * 110.57
    return width_km, height_km, width_km * height_km


def check_inputs(args):
    """Problems with the event name, bbox (W S E N) and date; empty when they are valid."""
    problems = []
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}', args.event or ''):
        problems.append(
            f'--event {args.event!r}: use letters, digits, ".", "_" or "-" only (it names the output folder), '
            'e.g. judge_area_1'
        )
    west, south, east, north = args.bbox
    if not all(math.isfinite(v) for v in args.bbox):
        problems.append('--bbox contains a non-finite number')
    elif not -180 <= west < east <= 180:
        problems.append(
            f'--bbox W={west} E={east}: need -180 <= W < E <= 180 (order is W S E N; areas across the antimeridian '
            'are not supported)'
        )
    elif not -84 < south < north < 84:
        problems.append(f'--bbox S={south} N={north}: need -84 < S < N < 84 (order is W S E N)')
    elif min(east - west, north - south) <= 0.01:
        problems.append(
            f'--bbox is {east - west:.4f} x {north - south:.4f} deg: each side must exceed 0.01 deg (about 1 km)'
        )
    elif bbox_size_km(args.bbox)[2] > MAX_KM2 and not args.force_large:
        problems.append(
            f'--bbox covers ~{bbox_size_km(args.bbox)[2]:,.0f} km2 (> {MAX_KM2:,} km2): expect several hours and '
            'out-of-memory errors. Choose a smaller area, or add --force-large'
        )
    try:
        parse_event_time(args.date)
    except ValueError:
        problems.append(
            f'--date {args.date!r}: use YYYY-MM-DD, YYYY-MM-DDTHH:MM (UTC), or a time with an offset such as '
            '2026-08-26T03:15+05:45'
        )
    return problems


def run_key(ctx):
    return dict(
        event=ctx.event,
        bbox=[round(float(x), 6) for x in ctx.bbox],
        date_given=ctx.date_given,
        time_given=ctx.has_time,
        event_utc=ctx.event_utc,
    )


def key_differences(stored, current):
    differences = []
    if stored['bbox'] != current['bbox']:
        differences.append(f'bbox {stored["bbox"]} -> {current["bbox"]}')
    if stored['event_utc'] != current['event_utc']:
        differences.append(f'event start (UTC) {stored["event_utc"]} -> {current["event_utc"]}')
    if stored['time_given'] != current['time_given']:
        differences.append('date only <-> date with time (changes which same-day images are used)')
    return differences


def check_run_key(ctx):
    """Compare the stored area and event time; return the run state, differences and stored key."""
    key_path = os.path.join(ctx.out_dir, 'run_key.json')
    if not os.path.exists(key_path):
        return 'new', [], None
    stored = read_json(key_path)
    differences = key_differences(stored, run_key(ctx))
    return ('mismatch' if differences else 'resume'), differences, stored


def write_run_key(ctx, version):
    key_path = ctx.path('run_key.json')
    created_utc = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    write_json(key_path, dict(run_key(ctx), code_version=version, created_utc=created_utc), indent=1)
    ctx.upload(key_path)


def merge_previous_summary(ctx):
    """Keep the input metadata of the previous run of this folder (S2 dates, OSM snapshot, ...)."""
    summary_path = ctx.path('summary.json')
    if not os.path.exists(summary_path):
        return
    try:
        previous = read_json(summary_path)
    except json.JSONDecodeError:  # a run killed while writing it
        log('note: old summary.json not merged (unreadable)')
        return
    ctx.summary.update({k: v for k, v in previous.items() if k not in SUMMARY_META})


def write_summary(ctx):
    summary_path = ctx.path('summary.json')
    summary = dict(
        event=ctx.event,
        bbox=ctx.bbox,
        date=ctx.event_date,
        event_time=ctx.event_utc,
        date_given=ctx.date_given,
        **ctx.summary,
    )
    write_json(summary_path, summary, indent=1, default=str)
    ctx.upload(summary_path)


def continue_without_s1(ctx):
    """Record a failed radar step so the remaining steps can run and the next run can retry radar."""
    index_path = ctx.path('s1', 'index.json')
    if os.path.exists(index_path):
        return
    skipped = dict(
        utc=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        error=ctx.summary.get('errors', {}).get('s1'),
    )
    write_json(ctx.path('s1', 'skipped.json'), skipped)
    write_json(index_path, [])
    log('s1 skipped: continuing without radar evidence')


def warn_if_small_grid(ctx):
    """Report grids smaller than the model tiles, where change calibration also has fewer samples."""
    grid = ctx.grid()
    ctx.clear_warnings(SMALL_BBOX)
    if min(grid['width'], grid['height']) >= MIN_GRID_PX:
        ctx.summary.pop('small_bbox', None)
        return
    ctx.summary['small_bbox'] = dict(width_px=grid['width'], height_px=grid['height'], min_px=MIN_GRID_PX)
    ctx.warn(
        f'{SMALL_BBOX} ({grid["width"]} x {grid["height"]} px at 10 m, under 256 px on a side): the AI model works on '
        '512-px tiles and may fail or be less reliable here, and the radar/optical statistics rest on few pixels.'
    )


def run_step(ctx, name, step):
    """Run a step and record failures in the event summary; return whether the step ran."""
    started = time.time()
    ctx.clear_warnings(f'Step "{name}" failed')
    try:
        step(ctx)
    except Exception as error:  # noqa: BLE001 - the run continues without this step; the report shows the failure
        traceback.print_exc()
        ctx.summary.setdefault('errors', {})[name] = repr(error)[:500]
        ctx.warn(f'Step "{name}" failed ({type(error).__name__}: {str(error)[:200]}). {STEP_FAILURE_NOTES[name]}')
        if name == 's1':
            continue_without_s1(ctx)
        succeeded = False
    else:
        ctx.summary.setdefault('timing', {})[name] = round(time.time() - started)
        ctx.summary.get('errors', {}).pop(name, None)
        succeeded = True
    if 'errors' in ctx.summary and not ctx.summary['errors']:
        ctx.summary.pop('errors')
    log(f'== step {name} done in {time.time() - started:.0f}s')
    return succeeded


def run_steps(ctx, only):
    """Run the steps in order, writing summary.json after each. Returns the exit code."""
    started = time.time()
    for name, step in STEPS:
        if only and name not in only:
            continue
        if name in ('osm', 'impacts') and ctx.summary.get('no_observation'):
            log(f'== step {name} skipped: no satellite observation for this area/date')
            continue
        succeeded = run_step(ctx, name, step)
        if name == 's2_cal':
            check_data_availability(ctx)
        write_summary(ctx)
        if not succeeded and name in FATAL_STEPS:
            log(f'PIPELINE STOPPED: step {name} failed (fatal)', round(time.time() - started), 's')
            return 1
    log('PIPELINE DONE', round(time.time() - started), 's')
    return 0


def print_run_header(ctx):
    """Print how the inputs are read. Returns the bbox area in km2."""
    width_km, height_km, area_km2 = bbox_size_km(ctx.bbox)
    west, south, east, north = ctx.bbox
    log(
        f'== event {ctx.event}: bbox W {west} S {south} E {east} N {north} '
        f'({width_km:.1f} x {height_km:.1f} km, ~{area_km2:,.0f} km2)'
    )
    date_only_note = (
        '' if ctx.has_time else f'; date only: images acquired on UTC day {ctx.event_date} are ambiguous and not used'
    )
    log(f'== event start {ctx.event_utc} UTC (--date {ctx.date_given}){date_only_note}')
    log('== output', ctx.out_dir)
    if ctx.event_time > dt.datetime.now(dt.timezone.utc).replace(tzinfo=None):
        log('WARNING: the event start is in the future: no post-event images can exist yet')
    if area_km2 > WARN_KM2:
        log(
            f'WARNING: large area (~{area_km2:,.0f} km2 > {WARN_KM2:,}). Measured on 8 CPUs: ~950 km2 = 17 min, '
            '~1,600 km2 = 22 min; larger areas take at least proportionally longer and need more memory'
        )
    return area_km2


def run(args):
    """Run one event; return 0 on completion, 1 on fatal failure, 2 on invalid input or 3 on a cached-input
    mismatch.
    """
    step_names = [name for name, _ in STEPS]
    problems = check_inputs(args) + [
        f'--only {name!r} is not a step; steps: {" ".join(step_names)}' for name in args.only if name not in step_names
    ]
    if problems:
        log('INPUT ERROR (nothing was run):\n  - ' + '\n  - '.join(problems))
        return 2
    out_dir = os.path.join(args.root, args.event)
    existed = os.path.isdir(out_dir)
    if args.fresh and existed:
        if args.check:
            log('--fresh: would delete', out_dir)
        else:
            shutil.rmtree(out_dir)
            existed = False
            log('--fresh: deleted', out_dir, '(the store copy, if any, is not deleted)' if args.push else '')
    ctx = RunContext(args.event, args.bbox, args.date, args.root, args.push, args.source)
    area_km2 = print_run_header(ctx)
    status, differences, stored_key = ('new', [], None) if (args.fresh and args.check) else check_run_key(ctx)
    if status == 'mismatch':
        log(
            f'ERROR: RUN KEY MISMATCH. The folder {ctx.out_dir} was created for other inputs:\n  - '
            + '\n  - '.join(differences)
            + '\nIts cached inputs (grid, DEM, Sentinel-1/2 stacks, pre-event OpenStreetMap) belong to that area/date; '
            'reusing them would give wrong maps (and could use OpenStreetMap from after this event).\n'
            'Use a new --event name, or add --fresh to delete this folder and start over. Nothing was run.'
        )
        return 3
    version = code_version()
    if args.check:
        log(f'CHECK OK: run key {status}; code {version}. Nothing was written.')
        if not existed:
            os.rmdir(ctx.out_dir)
        return 0
    if status == 'resume':
        code_note = (
            f' (code changed since the first run: {stored_key["code_version"]} -> {version}; '
            'cached step outputs are reused)'
            if stored_key['code_version'] != version
            else ''
        )
        log('== resuming: inputs match run_key.json' + code_note)
    else:
        write_run_key(ctx, version)
    merge_previous_summary(ctx)
    ctx.summary.update(code_version=version, bbox_area_km2=round(area_km2, 1))
    warn_if_small_grid(ctx)
    return run_steps(ctx, args.only)


def main():
    parser = argparse.ArgumentParser(
        description='Flood extent, infrastructure exposure and settlement road access for an area and flood date.'
    )
    parser.add_argument('--event', help='output folder name <root>/<event> (letters, digits, . _ -)')
    parser.add_argument('--bbox', type=float, nargs=4, metavar=('W', 'S', 'E', 'N'), help='area, lon/lat degrees')
    parser.add_argument(
        '--date',
        help='flood start: YYYY-MM-DD, or YYYY-MM-DDTHH:MM in UTC, or with an offset (2026-08-26T03:15+05:45, ...Z)',
    )
    parser.add_argument(
        '--root',
        default=DEFAULT_ROOT,
        help='output root (default: $TRI_ROOT, else /content/tri/events on the project VMs, else ./runs)',
    )
    parser.add_argument('--push', action='store_true', help='push the outputs to the project store')
    parser.add_argument(
        '--only', nargs='*', default=[], help=f'run only these steps: {" ".join(name for name, _ in STEPS)}'
    )
    parser.add_argument('--fresh', action='store_true', help='delete <root>/<event> first and start from scratch')
    parser.add_argument('--force-large', action='store_true', help=f'allow areas above {MAX_KM2:,} km2')
    parser.add_argument(
        '--check',
        action='store_true',
        help='validate the inputs and the run key, print how they are read, write nothing',
    )
    parser.add_argument(
        '--preflight',
        action='store_true',
        help='check the environment (STAC, geoid grid, model weights, tools) and exit',
    )
    parser.add_argument(
        '--source', type=float, nargs=2, default=None, help='lon lat of an upstream point for flood-path tracing'
    )
    args = parser.parse_args()
    if args.preflight:
        import preflight  # noqa: PLC0415 - preflight imports this module

        return preflight.main(root=args.root)
    missing = [f'--{name}' for name in ('event', 'bbox', 'date') if not getattr(args, name)]
    if missing:
        parser.error('missing ' + ' '.join(missing))
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
