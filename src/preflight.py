"""Check dependencies, data access, model weights and available machine resources."""

import argparse
import datetime as dt
import os
import pickle
import shutil
import sys
import tempfile
import time
from importlib import import_module, metadata

# Checks import third-party and project modules by name, after the package check, so that a missing package
# is reported in the table instead of stopping the preflight at import time.

STAC_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1'
PROBE_BBOX = [85.30, 27.70, 85.33, 27.72]  # small Kathmandu-valley box for availability probes
PACKAGES = [
    ('numpy', 'numpy'),
    ('scipy', 'scipy'),
    ('rasterio', 'rasterio'),
    ('pyproj', 'pyproj'),
    ('shapely', 'shapely'),
    ('geopandas', 'geopandas'),
    ('pandas', 'pandas'),
    ('networkx', 'networkx'),
    ('requests', 'requests'),
    ('pystac', 'pystac'),
    ('pystac_client', 'pystac-client'),
    ('planetary_computer', 'planetary-computer'),
    ('odc.stac', 'odc-stac'),
    ('odc.geo', 'odc-geo'),
    ('pysheds', 'pysheds'),
    ('folium', 'folium'),
    ('PIL', 'Pillow'),
    ('torch', 'torch'),
    ('segmentation_models_pytorch', 'segmentation-models-pytorch'),
]
TOOLS = (
    ('osmium', 'clipping large Geofabrik extracts (without it: slow GDAL read)'),
    ('aria2c', 'fast Geofabrik downloads (without it: plain HTTP download)'),
)
OSM_SOURCES = (
    ('Geofabrik', 'https://download.geofabrik.de/index-v1-nogeom.json'),
    ('ohsome', 'https://api.ohsome.org/v1/metadata'),
    ('Overpass', 'https://overpass-api.de/api/status'),
)
WEIGHT_FILES = ('best.pt', 'config.json')
MIN_FREE_DISK_GB = 10
MIN_MEMORY_GB = 16


def _error_summary(error):
    """Exception type and HTTP status, without URLs or tokens."""
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    return type(error).__name__ + (f' (HTTP {status})' if status else '')


def _is_writable(path):
    """True when the nearest existing ancestor of path is writable."""
    path = os.path.abspath(path)
    while not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    return os.access(path, os.W_OK)


def check_packages():
    versions, missing = [], []
    for module_name, distribution in PACKAGES:
        try:
            module = import_module(module_name)
        except (ImportError, OSError) as error:
            missing.append(f'{distribution} ({type(error).__name__})')
            continue
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            version = getattr(module, '__version__', '?')
        versions.append(f'{distribution}=={version}')
    python_version = sys.version.split()[0]
    if sys.version_info < (3, 11):
        yield 'FAIL', 'python', f'{python_version}: need >= 3.11 (ISO dates with offsets / Z)'
    else:
        yield 'PASS', 'python', python_version
    if missing:
        yield 'FAIL', 'python packages', 'missing: ' + ', '.join(missing)
    else:
        yield 'PASS', 'python packages', f'{len(versions)} importable'
    print('       versions: ' + ' '.join(versions), flush=True)


def check_root(root):
    try:
        os.makedirs(root, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root, prefix='.preflight_') as f:
            f.write(b'ok')
            f.flush()
        free_gb = shutil.disk_usage(root).free / 1e9
    except OSError as error:
        yield (
            'FAIL',
            'output root writable',
            f'{root}: {_error_summary(error)}; set --root or TRI_ROOT to a writable folder',
        )
    else:
        if free_gb >= MIN_FREE_DISK_GB:
            yield 'PASS', 'output root writable', f'{root} ({free_gb:.0f} GB free)'
        else:
            yield 'WARN', 'output root writable', f'{root} ({free_gb:.0f} GB free, < {MIN_FREE_DISK_GB} GB)'
    cache_dir = os.path.join(root, '_osm_cache')
    if _is_writable(cache_dir):
        yield 'PASS', 'OSM extract cache', f'{cache_dir} writable'
    else:
        yield (
            'WARN',
            'OSM extract cache',
            f'{cache_dir} not writable: Geofabrik extracts cannot be used, '
            'OSM falls back to Overpass/ohsome (slower, less reliable)',
        )


def check_tools():
    for tool, purpose in TOOLS:
        path = shutil.which(tool)
        if path:
            yield 'PASS', tool, path
        else:
            yield 'WARN', tool, f'not on PATH: used for {purpose}; apt-get install -y osmium-tool aria2'


def check_geoid():
    np = import_module('numpy')
    s1_grd = import_module('s1_grd')
    grid_path = s1_grd.geoid_grid_path()
    note = ''
    if not grid_path:
        print('       EGM2008 grid not on disk: downloading (pyproj sync, ~80 MB)', flush=True)
        if s1_grd.sync_geoid_grid():
            grid_path = s1_grd.geoid_grid_path()
            note = ', downloaded now'
    fix = f'python -m pyproj sync --file {s1_grd.GEOID_GRID}'
    try:
        undulation_m, model = s1_grd.geoid_undulation(np.array([85.3]), np.array([27.7]), sync=False)
    except RuntimeError as error:
        yield (
            'FAIL',
            'geoid (EGM2008)',
            f'no geoid model ({_error_summary(error)}): the radar step would be skipped (map from Sentinel-2 only). '
            f'Run: {fix}',
        )
        return
    if grid_path and '3855' in model:  # EPSG:3855 is EGM2008 height
        yield (
            'PASS',
            'geoid (EGM2008)',
            f'grid on disk in {os.path.dirname(grid_path)}{note}; N(85.3E, 27.7N) = {undulation_m[0]:.1f} m',
        )
        return
    reason = 'EGM2008 was not used' if grid_path else 'the EGM2008 grid is not on disk (PROJ network at run time)'
    yield 'WARN', 'geoid (EGM2008)', f'{model[:40]} works but {reason}; run: {fix}'


def _stac_row(catalog, collection, asset_key, start):
    requests = import_module('requests')
    today = dt.date.today()
    window = {} if collection.startswith('cop-dem') else dict(datetime=f'{today - dt.timedelta(days=90)}/{today}')
    items = list(catalog.search(collections=[collection], bbox=PROBE_BBOX, max_items=1, **window).items())
    if not items:
        return 'WARN', f'STAC {collection}', 'search OK but no item over the probe box'
    asset = items[0].assets.get(asset_key) or next(iter(items[0].assets.values()))
    is_signed = 'sig=' in asset.href or 'se=' in asset.href
    response = requests.get(asset.href, headers={'Range': 'bytes=0-1023'}, timeout=60)
    is_readable = response.status_code in (200, 206)
    read = f'{"signed" if is_signed else "UNSIGNED"} read HTTP {response.status_code}'
    return (
        'PASS' if is_signed and is_readable else 'FAIL',
        f'STAC {collection}',
        f'search + {read} ({time.time() - start:.1f} s)',
    )


def check_stac():
    requests = import_module('requests')
    pystac_client = import_module('pystac_client')
    planetary_computer = import_module('planetary_computer')
    stac_errors = (import_module('pystac_client.exceptions').APIError, requests.RequestException)
    start = time.time()
    try:
        catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    except stac_errors as error:
        yield 'FAIL', 'Planetary Computer STAC', f'not reachable ({_error_summary(error)})'
        return
    for collection, asset_key in (('sentinel-1-grd', 'vv'), ('sentinel-2-l2a', 'B04'), ('cop-dem-glo-30', 'data')):
        try:
            row = _stac_row(catalog, collection, asset_key, start)
        except stac_errors as error:
            row = 'FAIL', f'STAC {collection}', f'search or read failed ({_error_summary(error)})'
        yield row


def _fetch_weights(run, paths):
    requests = import_module('requests')
    store = import_module('store')
    try:
        for path in paths:
            if not os.path.exists(path):
                store.get(f'models/{run}/{os.path.basename(path)}', path)
    except (requests.RequestException, FileNotFoundError) as error:
        print('       fetch failed', run, _error_summary(error), flush=True)


def _is_listed_in_store(run):
    requests = import_module('requests')
    store = import_module('store')
    try:
        names = {entry['path'].split('/')[-1] for entry in store.list_files(f'models/{run}')}
    except requests.RequestException as error:
        print('       store listing failed', run, _error_summary(error), flush=True)
        return False
    return set(WEIGHT_FILES) <= names


def _local_weights_row(run, run_dir, paths):
    radar_ai = import_module('radar_ai')
    load_errors = (RuntimeError, KeyError, ValueError, OSError, pickle.UnpicklingError)
    try:
        _, config = radar_ai.load_model(paths[0], paths[1], device='cpu')
    except load_errors as error:
        return 'FAIL', f'weights {run}', f'present but do not load ({_error_summary(error)}: {str(error)[:120]})'
    return (
        'PASS',
        f'weights {run}',
        f'loads ({config.get("arch", "Unet")}/{config.get("encoder")}) from {run_dir}',
    )


def check_weights(pipeline, fetch_weights):
    has_token = import_module('store').has_token()
    if has_token:
        yield 'PASS', 'project store token', 'present (not shown)'
    else:
        yield (
            'WARN',
            'project store token',
            'absent: model weights must be local; no OSM store proxy/cache (fine if weights are in place)',
        )
    for run in pipeline.RADAR_AI_RUNS:
        run_dir = os.path.join(pipeline.MODELS_DIR, run)
        paths = [os.path.join(run_dir, name) for name in WEIGHT_FILES]
        if fetch_weights and has_token and not all(os.path.exists(path) for path in paths):
            _fetch_weights(run, paths)
        if all(os.path.exists(path) for path in paths):
            yield _local_weights_row(run, run_dir, paths)
        elif has_token and _is_listed_in_store(run):
            yield (
                'WARN',
                f'weights {run}',
                f'not in {run_dir}; fetchable from the project store at run time (or now: preflight --fetch-weights)',
            )
        else:
            yield (
                'FAIL',
                f'weights {run}',
                f'missing: put best.pt + config.json in {run_dir} (docs/RUNNING.md); '
                'without them the AI model is not used',
            )


def check_osm_sources():
    requests = import_module('requests')
    for name, url in OSM_SOURCES:
        try:
            response = requests.get(url, timeout=20, headers={'User-Agent': 'trishuli-preflight'})
        except requests.RequestException as error:
            yield 'WARN', f'OSM source {name}', f'not reachable ({_error_summary(error)})'
            continue
        yield 'PASS' if response.status_code == 200 else 'WARN', f'OSM source {name}', f'HTTP {response.status_code}'


def _memory_gb():
    """Physical memory, or None where os.sysconf does not report it (Windows)."""
    if 'SC_PHYS_PAGES' not in getattr(os, 'sysconf_names', {}):
        return None
    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9


def check_compute():
    memory_gb = _memory_gb()
    detail = f'{os.cpu_count() or 0} CPUs' + (f', {memory_gb:.0f} GB RAM' if memory_gb else '')
    if memory_gb is None or memory_gb >= MIN_MEMORY_GB:
        yield 'PASS', 'CPU / memory', detail
    else:
        yield 'WARN', 'CPU / memory', detail + f' (< {MIN_MEMORY_GB} GB: areas above ~1,000 km2 may run out of memory)'
    torch = import_module('torch')
    if torch.cuda.is_available():
        yield 'PASS', 'torch / CUDA', f'torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)} (DL with D4 TTA)'
    else:
        yield 'PASS', 'torch / CUDA', f'torch {torch.__version__}, CPU only (DL without TTA, ~8 min per 1,000 km2)'


def _run(section, check, rows):
    print(f'-- {section}', flush=True)
    for status, name, detail in check:
        rows.append((status, name, detail))
        print(f'  {status:<4} {name:<28} {detail}', flush=True)


def _report(rows):
    """Print the summary table; returns the exit code."""
    counts = {status: sum(row[0] == status for row in rows) for status in ('PASS', 'WARN', 'FAIL')}
    width = max(len(row[1]) for row in rows)
    print('\n' + '=' * 100 + '\nPREFLIGHT TABLE', flush=True)
    for status, name, detail in rows:
        print(f'{status:<4}  {name:<{width}}  {detail}', flush=True)
    print('=' * 100, flush=True)
    if counts['FAIL']:
        verdict = 'NOT READY (fix the FAIL rows)'
    else:
        verdict = 'READY' + (' (see WARN rows)' if counts['WARN'] else '')
    print(
        f'PREFLIGHT RESULT: {counts["PASS"]} PASS, {counts["WARN"]} WARN, {counts["FAIL"]} FAIL -> {verdict}',
        flush=True,
    )
    return 1 if counts['FAIL'] else 0


def main(root=None, fetch_weights=False):
    print(f'PREFLIGHT {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC', flush=True)
    rows = []
    _run('packages', check_packages(), rows)
    if any(status == 'FAIL' for status, _, _ in rows):  # the remaining checks import these packages
        return _report(rows)
    pipeline = import_module('pipeline')
    root = pipeline.DEFAULT_ROOT if root is None else root
    _run('output root', check_root(root), rows)
    _run('tools', check_tools(), rows)
    _run('geoid', check_geoid(), rows)
    _run('STAC', check_stac(), rows)
    _run('weights', check_weights(pipeline, fetch_weights), rows)
    _run('OSM sources', check_osm_sources(), rows)
    _run('compute', check_compute(), rows)
    return _report(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=None, help='output root to test (default: as pipeline.py)')
    parser.add_argument(
        '--fetch-weights', action='store_true', help='download missing model weights from the project store'
    )
    args = parser.parse_args()
    sys.exit(main(args.root, args.fetch_weights))
