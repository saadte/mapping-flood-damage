"""Pre-event OpenStreetMap layers from Geofabrik dated extracts or from ohsome / Overpass snapshot queries."""

import datetime as dt
import json
import math
import os
import re
import shutil
import struct
import subprocess
import time
import zlib
from typing import NamedTuple
from urllib.parse import quote

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

import store

USER_AGENT = {'User-Agent': 'trishuli-flood-mapping/0.1 (educational hackathon prototype)'}
OHSOME_URL = 'https://api.ohsome.org/v1/elements/geometry'
OHSOME_TIMEOUT_S = 240
OVERPASS_URLS = [
    'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
]
# Attic queries over large boxes time out on the public mirrors, so the last fallback splits the box 3 x 3.
OVERPASS_TILES = 3
GEOFABRIK_URL = 'https://download.geofabrik.de'
MIN_REGION_COVER = 0.8
STORE_MAX_BYTES = 1e9  # larger extracts are not cached in the project store
REMAINDER_BUDGET_S = 150.0
NET_LAYERS = ('roads', 'places', 'health', 'admin')
AOI_LAYERS = ('buildings', 'roads', 'bridges', 'places', 'health', 'waterways', 'admin')
HEALTH_AMENITIES = ['hospital', 'clinic', 'doctors', 'health_post']

# layer -> (ohsome filter, Overpass selectors)
LAYERS = {
    'buildings': (
        'building=* and building!=no and geometry:polygon',
        ['way["building"]["building"!="no"]', 'relation["building"]'],
    ),
    'roads': ('highway=* and geometry:line', ['way["highway"]']),
    'bridges': (
        '(bridge=* and bridge!=no) or man_made=bridge',
        ['way["bridge"]["bridge"!="no"]', 'way["man_made"="bridge"]', 'node["man_made"="bridge"]'],
    ),
    'places': (
        'place in (city, town, village, hamlet, isolated_dwelling, suburb, neighbourhood, locality, farm)',
        [
            'node["place"~"^(city|town|village|hamlet|isolated_dwelling|suburb|neighbourhood|locality|farm)$"]',
            'way["place"~"^(city|town|village|hamlet|suburb|neighbourhood)$"]',
        ],
    ),
    'health': (
        'amenity in (hospital, clinic, doctors, health_post) or healthcare=*',
        ['nwr["amenity"~"^(hospital|clinic|doctors|health_post)$"]', 'nwr["healthcare"]'],
    ),
    'waterways': ('waterway in (river, stream, canal) and geometry:line', ['way["waterway"~"^(river|stream|canal)$"]']),
    # Overpass leaves out admin_level 2: whole-country relations are too large for the public mirrors.
    'admin': (
        'boundary=administrative and admin_level in (2, 4, 5, 6, 7, 8, 9) and geometry:polygon',
        ['relation["boundary"="administrative"]["admin_level"~"^(4|5|6|7|8|9)$"]'],
    ),
}

KEEP_TAGS = [
    'osm_type',
    'osm_id',
    'osm_way_id',
    '@osmId',
    'name',
    'name:en',
    'name:ne',
    'highway',
    'bridge',
    'bridge:structure',
    'layer',
    'surface',
    'tracktype',
    'smoothness',
    'motor_vehicle',
    'access',
    'foot',
    'width',
    'lanes',
    'ref',
    'ford',
    'tunnel',
    'maxweight',
    'building',
    'building:levels',
    'building:material',
    'amenity',
    'healthcare',
    'healthcare:speciality',
    'beds',
    'place',
    'population',
    'waterway',
    'natural',
    'water',
    'man_made',
    'power',
    'plant:source',
    'plant:output:electricity',
    'generator:source',
    'office',
    'government',
    'admin_level',
    'boundary',
    'landuse',
    'is_in',
    'operator',
]

# Tags that GDAL's OSM driver leaves inside 'other_tags' and the layer selection needs as columns.
EXTRA_TAG_COLUMNS = [
    'bridge',
    'healthcare',
    'amenity',
    'place',
    'layer',
    'surface',
    'tracktype',
    'ford',
    'tunnel',
    'population',
    'name:en',
    'name:ne',
    'ref',
    'admin_level',
    'boundary',
    'building',
    'waterway',
    'highway',
]

AREA_TAGS = ('building', 'water', 'landuse', 'amenity', 'healthcare', 'power', 'place', 'office')


class Region(NamedTuple):
    base_url: str
    name: str
    cover: float  # fraction of the queried box inside the region
    area: float  # square degrees
    level: str  # continent, country or subregion
    geometry: BaseGeometry


def net_bbox(bbox, buffer_km):
    """bbox (west, south, east, north) grown by buffer_km on every side."""
    mid_lat_deg = (bbox[1] + bbox[3]) / 2.0
    lat_step_deg = buffer_km / 110.57
    lon_step_deg = buffer_km / (111.32 * max(math.cos(math.radians(mid_lat_deg)), 0.1))
    return [
        round(bbox[0] - lon_step_deg, 5),
        round(max(bbox[1] - lat_step_deg, -85.0), 5),
        round(bbox[2] + lon_step_deg, 5),
        round(min(bbox[3] + lat_step_deg, 85.0), 5),
    ]


def _empty_frame():
    return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')


def _concat_frames(frames):
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        return _empty_frame()
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry='geometry', crs=frames[0].crs)


def _write_layer(frame, path):
    """frame reduced to KEEP_TAGS (as strings) and geometry, written to path unless empty."""
    kept = [column for column in frame.columns if column in KEEP_TAGS]
    frame = frame[[*kept, 'geometry']].copy()
    for column in kept:
        frame[column] = frame[column].astype('string')
    if len(frame):
        frame.to_file(path, driver='GPKG')
    return frame


def _write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=1)


def _ohsome(bbox, timestamp, ohsome_filter):
    response = requests.post(
        OHSOME_URL,
        data={'bboxes': ','.join(map(str, bbox)), 'time': timestamp, 'filter': ohsome_filter, 'properties': 'tags'},
        headers=USER_AGENT,
        timeout=OHSOME_TIMEOUT_S,
    )
    response.raise_for_status()
    features = response.json().get('features', [])
    if not features:
        return _empty_frame()
    return gpd.GeoDataFrame.from_features(features, crs='EPSG:4326')


def _is_area(tags):
    return (
        any(key in tags for key in AREA_TAGS)
        or tags.get('natural') == 'water'
        or tags.get('waterway') == 'riverbank'
        or tags.get('area') == 'yes'
    )


def _overpass_geometry(element):
    if element['type'] == 'node':
        return Point(element['lon'], element['lat'])
    if element['type'] == 'way':
        points = [(node['lon'], node['lat']) for node in element.get('geometry') or [] if node]
        if len(points) < 2:
            return None
        is_closed = points[0] == points[-1] and len(points) >= 4
        return Polygon(points) if is_closed and _is_area(element.get('tags', {})) else LineString(points)
    if element['type'] == 'relation':
        outer_lines = []
        for member in element.get('members', []):
            if member.get('geometry') and member.get('role', 'outer') in ('outer', ''):
                points = [(node['lon'], node['lat']) for node in member['geometry'] if node]
                if len(points) >= 2:
                    outer_lines.append(LineString(points))
        polygons = list(polygonize(unary_union(outer_lines))) if outer_lines else []
        return MultiPolygon(polygons) if polygons else None
    return None


def _overpass_frame(elements):
    rows = []
    for element in elements:
        geometry = _overpass_geometry(element)
        if geometry is None:
            continue
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
            if not geometry.is_valid:
                continue
        rows.append(dict(element.get('tags', {}), osm_type=element['type'], osm_id=element['id'], geometry=geometry))
    if not rows:
        return _empty_frame()
    return gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')


def _overpass(bbox, timestamp, selectors, timeout_s, rounds):
    """(frame, mirror URL) from the first Overpass mirror that answers, trying every mirror `rounds` times."""
    west, south, east, north = bbox
    body = ''.join(f'{selector}({south},{west},{north},{east});' for selector in selectors)
    query = f'[out:json][timeout:{timeout_s}][maxsize:2000000000][date:"{timestamp}"];({body});out geom;'
    last_error = None
    for _ in range(rounds):
        for url in OVERPASS_URLS:
            try:
                response = requests.post(url, data={'data': query}, headers=USER_AGENT, timeout=timeout_s + 60)
                response.raise_for_status()
                elements = response.json().get('elements', [])
            except requests.RequestException as error:
                last_error = error
                print('  overpass failed', url.split('/')[2], repr(error)[:120], flush=True)
                time.sleep(20)
                continue
            return _overpass_frame(elements), url
    raise RuntimeError(f'all overpass endpoints failed: {last_error!r}')


def _overpass_tiled(bbox, timestamp, selectors, timeout_s, rounds):
    """Overpass query split into OVERPASS_TILES x OVERPASS_TILES boxes, merged without duplicates."""
    west, south, east, north = bbox
    n = OVERPASS_TILES
    parts = []
    source = None
    for i in range(n):
        for j in range(n):
            tile = (
                west + (east - west) * i / n,
                south + (north - south) * j / n,
                west + (east - west) * (i + 1) / n,
                south + (north - south) * (j + 1) / n,
            )
            frame, url = _overpass(tile, timestamp, selectors, timeout_s, rounds)
            if len(frame):
                parts.append(frame)
                source = source or url
    merged = _concat_frames(parts)
    if len(merged):
        merged = merged.drop_duplicates(subset=['osm_type', 'osm_id']).reset_index(drop=True)
    return merged, f'{source} (tiled {n}x{n})'


def _query(source, layer, bbox, timestamp, overpass_timeout_s, overpass_rounds):
    """(frame, source description) of one layer from one source."""
    ohsome_filter, overpass_selectors = LAYERS[layer]
    if source == 'ohsome':
        return _ohsome(bbox, timestamp, ohsome_filter), 'ohsome'
    if source == 'overpass':
        return _overpass(bbox, timestamp, overpass_selectors, overpass_timeout_s, overpass_rounds)
    return _overpass_tiled(bbox, timestamp, overpass_selectors, min(300, overpass_timeout_s), min(2, overpass_rounds))


def _first_answer(layer, bbox, timestamp, sources, deadline, overpass_timeout_s, overpass_rounds):
    """(frame, source description, errors) from the first source that answers; frame is None when none does."""
    errors = {}
    for source in sources:
        if deadline and time.time() > deadline:
            errors[f'{layer}_{source}_error'] = 'skipped: time budget spent'
            print(f'  {layer}: {source} skipped (time budget)', flush=True)
            continue
        try:
            frame, description = _query(source, layer, bbox, timestamp, overpass_timeout_s, overpass_rounds)
        except (requests.RequestException, RuntimeError) as error:
            errors[f'{layer}_{source}_error'] = repr(error)[:200]
            print(f'  {layer}: {source} failed', repr(error)[:160], flush=True)
            continue
        return frame, description, errors
    return None, None, errors


def fetch_osm(  # noqa: PLR0913
    bbox,
    timestamp,
    out_dir,
    layers,
    prefer='ohsome',
    overpass_timeout=600,
    overpass_rounds=3,
    budget_s=None,
    prefix='',
    optional=None,
):
    """OSM layers of bbox at an ISO timestamp from ohsome / Overpass, written to out_dir/<prefix><layer>.gpkg."""
    os.makedirs(out_dir, exist_ok=True)
    deadline = time.time() + budget_s if budget_s else None
    if optional is None:  # a run with a time budget must not stop on a missing minor layer
        optional = tuple(name for name in LAYERS if name not in ('roads', 'places')) if deadline else ()
    if prefer == 'ohsome':
        sources = ('ohsome', 'overpass', 'overpass_tiled')
    else:
        sources = ('overpass', 'ohsome', 'overpass_tiled')
    frames, log = {}, {}
    for name in layers:
        path = os.path.join(out_dir, f'{prefix}{name}.gpkg')
        if os.path.exists(path):
            frames[name] = gpd.read_file(path)
            log[name] = dict(n=len(frames[name]), source='cached')
            continue
        start = time.time()
        frame, source, errors = _first_answer(
            name, bbox, timestamp, sources, deadline, overpass_timeout, overpass_rounds
        )
        log.update(errors)
        if frame is None:
            if name not in optional:
                raise RuntimeError(f'OSM layer {name}: all sources failed {log}')
            log[name] = dict(n=0, source='unavailable')
            print(f'{name}: unavailable (all sources failed / budget)', flush=True)
            continue
        frames[name] = _write_layer(frame.loc[:, ~frame.columns.duplicated()], path)
        log[name] = dict(n=len(frames[name]), source=source, secs=round(time.time() - start, 1))
        print(name, log[name], flush=True)
    _write_json(
        dict(bbox=list(bbox), timestamp=timestamp, layers=log), os.path.join(out_dir, f'{prefix}osm_fetch_log.json')
    )
    return frames


def _get_text(url):
    """Text of a URL, through the project store when this machine cannot reach the host (some Colab VMs)."""
    try:
        response = requests.get(url, timeout=60, headers=USER_AGENT)
        response.raise_for_status()
    except requests.RequestException as error:
        print('direct GET failed, using store proxy:', repr(error)[:120], flush=True)
        return store.fetch_text(url)
    return response.text


def _get_cached_extract(cache_path, pbf_path):
    """Copy the extract from the project store cache to pbf_path; False when the store does not have it."""
    try:
        if not any(entry['path'] == cache_path for entry in store.list_files('osm_cache')):
            return False
        store.get(cache_path, pbf_path)
    except requests.RequestException as error:
        print('store cache check failed', repr(error)[:120], flush=True)
        return False
    return True


def _download(url, path):
    """Download url to path with aria2c when installed, else over plain HTTP; False on failure."""
    if shutil.which('aria2c'):
        options = ['-x', '8', '-s', '8', '-c', '--max-tries=3', '--connect-timeout=20']
        destination = ['-d', os.path.dirname(path), '-o', os.path.basename(path)]
        result = subprocess.run(['aria2c', *options, *destination, url], capture_output=True, text=True, check=False)
        if result.returncode:
            print('direct download failed:', result.stderr[-150:], flush=True)
        return result.returncode == 0
    try:
        with requests.get(url, stream=True, timeout=60, headers=USER_AGENT) as response:
            response.raise_for_status()
            with open(path + '.part', 'wb') as f:
                for chunk in response.iter_content(1 << 22):
                    f.write(chunk)
    except requests.RequestException as error:
        print('direct download failed:', repr(error)[:150], flush=True)
        return False
    os.replace(path + '.part', path)
    return True


def _download_extract(url, pbf_path):
    """Extract to pbf_path from the project store cache, else downloaded (and cached), else fetched by the store."""
    cache_path = 'osm_cache/' + os.path.basename(url)
    if _get_cached_extract(cache_path, pbf_path):
        return
    if not _download(url, pbf_path):
        print('fetching the extract through the project store', flush=True)
        store.fetch_to_store(url, cache_path)
        store.get(cache_path, pbf_path)
        return
    size_bytes = os.path.getsize(pbf_path)
    if size_bytes > STORE_MAX_BYTES:
        print(f'extract is {size_bytes / 1e9:.1f} GB: not uploaded to the store cache', flush=True)
        return
    try:
        store.put(pbf_path, cache_path)
    except requests.RequestException as error:
        print('store cache upload failed', repr(error)[:120], flush=True)


def _region_level(properties):
    if not properties.get('parent'):
        return 'continent'
    return 'country' if properties.get('iso3166-1:alpha2') else 'subregion'


def _geofabrik_regions(bbox):
    """Geofabrik regions intersecting bbox, preferred first."""
    index = json.loads(_get_text(f'{GEOFABRIK_URL}/index-v1.json'))
    query_box = box(*bbox)
    regions = []
    for feature in index['features']:
        properties = feature.get('properties', {})
        if not feature.get('geometry') or 'pbf' not in properties.get('urls', {}):
            continue
        geometry = shape(feature['geometry'])
        if not geometry.intersects(query_box):
            continue
        regions.append(
            Region(
                base_url=properties['urls']['pbf'].replace('-latest.osm.pbf', ''),
                name=properties.get('id'),
                cover=geometry.intersection(query_box).area / query_box.area,
                area=geometry.area,
                level=_region_level(properties),
                geometry=geometry,
            )
        )
    # Smallest extract that covers the box, then the best partial ones; continent extracts are several GB.
    countries = [region for region in regions if region.level != 'continent']
    full = sorted([r for r in countries if r.cover >= MIN_REGION_COVER], key=lambda r: r.area)
    partial = sorted([r for r in countries if r.cover < MIN_REGION_COVER], key=lambda r: (-r.cover, r.area))
    continents = []
    if os.environ.get('TRI_OSM_ALLOW_CONTINENT', '0') == '1':
        continents = sorted([r for r in regions if r.level == 'continent'], key=lambda r: r.area)
    if not (full or partial or continents):
        print(
            'geofabrik: only continent-level extracts cover this area (set TRI_OSM_ALLOW_CONTINENT=1 to use them)',
            flush=True,
        )
    return full + partial + continents


def _dated_extract(region_url, before_date, want_date):
    """(url, YYMMDD) of the latest dated extract before before_date (YYYY-MM-DD), or of want_date when it qualifies."""
    html = _get_text(region_url + '.html')
    name = region_url.rsplit('/', 1)[1]
    dates = sorted(set(re.findall(name + r'-(\d{6})\.osm\.pbf', html)))
    before = before_date.replace('-', '')[2:8]
    earlier = [date for date in dates if date < before]
    if not earlier:
        return None
    date = want_date if want_date in earlier else earlier[-1]
    return f'{region_url}-{date}.osm.pbf', date


def _extract_day(yymmdd):
    return dt.date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6]))


def _end_of_day_utc(yymmdd):
    """Latest edit time an extract named YYMMDD can contain: it is cut during that day."""
    return f'20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}T23:59:59Z'


def _varint(data, i):
    value = shift = 0
    while True:
        byte = data[i]
        i += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return value, i


def _protobuf_fields(data):
    """(field number, value) pairs of a protobuf message, up to the first unsupported wire type."""
    i = 0
    while i < len(data):
        key, i = _varint(data, i)
        number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, i = _varint(data, i)
        elif wire_type == 2:
            size, i = _varint(data, i)
            value = data[i : i + size]
            i += size
        elif wire_type == 1:
            value = data[i : i + 8]
            i += 8
        elif wire_type == 5:
            value = data[i : i + 4]
            i += 4
        else:
            return
        yield number, value


def _replication_timestamp(pbf_path):
    """Replication timestamp in the .osm.pbf header as 'YYYY-MM-DDTHH:MM:SSZ', or None when the header has none."""
    with open(pbf_path, 'rb') as f:
        header_size = struct.unpack('>I', f.read(4))[0]
        blob_header = dict(_protobuf_fields(f.read(header_size)))
        blob = dict(_protobuf_fields(f.read(blob_header[3])))  # BlobHeader field 3: blob size
    header_block = blob.get(1) or zlib.decompress(blob[3])  # Blob field 1: raw data, field 3: zlib data
    for number, value in _protobuf_fields(header_block):
        if number == 32:  # HeaderBlock.osmosis_replication_timestamp
            return dt.datetime.fromtimestamp(value, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    return None


def _check_before_cutoff(replication_ts, extract_date, cutoff):
    """Refuse an extract that may contain edits made at or after the cutoff."""
    if replication_ts is None:
        if (dt.date.fromisoformat(cutoff[:10]) - _extract_day(extract_date)).days < 1:
            raise RuntimeError(f'OSM extract {extract_date} has no timestamp and is not at least 1 day before {cutoff}')
    elif replication_ts[:19] >= cutoff[:19]:
        raise RuntimeError(f'OSM extract timestamp {replication_ts} is not before {cutoff} - refusing (post-event OSM)')


def _other_tags(value):
    """Tags in GDAL's hstore 'other_tags' column as a dict."""
    if not isinstance(value, str):
        return {}
    return dict(re.findall(r'"([^"]+)"=>"([^"]*)"', value))


def _add_tag_columns(frame):
    tags = frame['other_tags'].map(_other_tags) if 'other_tags' in frame else [{}] * len(frame)
    for key in EXTRA_TAG_COLUMNS:
        if key not in frame:
            frame[key] = [tag.get(key) for tag in tags]


def _select(frame, mask):
    return frame[mask.fillna(False).astype(bool)] if len(frame) else frame


def _clip_extract(pbf_path, extent, cache_dir):
    """Path of the extract clipped to extent with osmium (cached); the full extract when osmium is missing or fails."""
    # GDAL's OSM driver is slow on multi-country extracts and can return nothing from them.
    if not shutil.which('osmium'):
        print('WARNING osmium not installed: reading the full extract with GDAL (slow for large regions)', flush=True)
        return pbf_path
    key = json.dumps([[round(x, 5) for x in extent], os.path.basename(pbf_path), 'smart-mp-boundary'])
    clip_path = os.path.join(cache_dir, f'clip_{quote(key, safe="")}.osm.pbf')
    if not os.path.exists(clip_path):
        west, south, east, north = extent
        # the 'smart' strategy with boundary relations keeps administrative areas whole
        for type_filter in (['-S', 'types=multipolygon,boundary'], []):
            command = ['osmium', 'extract', '-b', f'{west},{south},{east},{north}', '-s', 'smart', *type_filter]
            result = subprocess.run(
                [*command, pbf_path, '-o', clip_path, '--overwrite'], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                break
            print('osmium extract failed', type_filter, result.stderr[-300:], flush=True)
    return clip_path if os.path.exists(clip_path) else pbf_path


def _read_layers(pbf_path, extent, cache_dir):
    """Layers of a Geofabrik extract within extent (west, south, east, north)."""
    source = _clip_extract(pbf_path, extent, cache_dir)
    os.environ['OSM_MAX_TMPFILE_SIZE'] = '4000'  # MB of node cache that GDAL's OSM driver keeps in memory
    lines = gpd.read_file(source, layer='lines', bbox=tuple(extent))
    polygons = gpd.read_file(source, layer='multipolygons', bbox=tuple(extent))
    points = gpd.read_file(source, layer='points', bbox=tuple(extent))
    for frame in (lines, polygons, points):
        _add_tag_columns(frame)
    return {
        'roads': _select(lines, lines['highway'].notna()),
        'places': _concat_frames(
            [_select(points, points['place'].notna()), _select(polygons, polygons['place'].notna())]
        ),
        'health': _concat_frames(
            [
                _select(points, points['amenity'].isin(HEALTH_AMENITIES) | points['healthcare'].notna()),
                _select(polygons, polygons['amenity'].isin(HEALTH_AMENITIES) | polygons['healthcare'].notna()),
            ]
        ),
        'admin': _select(
            polygons, (polygons['boundary'].astype(str) == 'administrative') & polygons['admin_level'].notna()
        ),
        'buildings': _select(polygons, polygons['building'].notna() & (polygons['building'].astype(str) != 'no')),
        'bridges': _select(lines, lines['bridge'].notna() & (lines['bridge'].astype(str) != 'no')),
        'waterways': _select(lines, lines['waterway'].isin(['river', 'stream', 'canal'])),
    }


def _main_extract(extent, before_date, want_date):
    """(region, url, YYMMDD) of the first preferred region with a dated extract before before_date."""
    for region in _geofabrik_regions(extent)[:8]:
        try:
            dated = _dated_extract(region.base_url, before_date, want_date)
        except requests.RequestException as error:
            print('geofabrik region', region.name, 'unusable:', repr(error)[:120], flush=True)
            continue
        if dated is None:
            print('geofabrik region', region.name, 'has no dated extract before', before_date, flush=True)
            continue
        url, extract_date = dated
        print(
            'geofabrik region',
            region.name,
            f'({region.level})',
            f'covers {region.cover:.2f} of the extent',
            '-> dated extract',
            extract_date,
            flush=True,
        )
        return region, url, extract_date
    raise RuntimeError('no Geofabrik country/sub-region with a dated extract before the event')


def _remainder_from_geofabrik(main_region, remainder, bounds, cutoff, extract_date, names, cache_dir):
    """Layers of the remainder from a neighbouring region's dated extract, and the source log; ({}, {}) if none fits."""
    for region in _geofabrik_regions(bounds)[:6]:
        if region.name == main_region.name or region.geometry.intersection(remainder).area < 0.5 * remainder.area:
            continue
        try:
            dated = _dated_extract(region.base_url, cutoff[:10], extract_date)
            if dated is None:
                print('remainder region', region.name, 'has no dated extract before', cutoff[:10], flush=True)
                continue
            url, date = dated
            pbf_path = os.path.join(cache_dir, os.path.basename(url))
            if not os.path.exists(pbf_path):
                _download_extract(url, pbf_path)
        except requests.RequestException as error:
            print('remainder region', region.name, 'unusable:', repr(error)[:150], flush=True)
            continue
        replication_ts = _replication_timestamp(pbf_path)
        if replication_ts is not None and replication_ts[:19] >= cutoff[:19]:
            print('remainder region', region.name, 'unusable: extract timestamp', replication_ts, flush=True)
            continue
        frames = {name: frame for name, frame in _read_layers(pbf_path, bounds, cache_dir).items() if name in names}
        print('remainder from geofabrik', region.name, date, {k: len(v) for k, v in frames.items()}, flush=True)
        return frames, dict(source=url, extract_date=date, replication_timestamp=replication_ts)
    return {}, {}


def _remainder_from_attic(bounds, aoi_part, snapshot_utc, names, cache_dir):
    """Layers of the remainder from an Overpass / ohsome attic query at the extract's snapshot time."""
    key = json.dumps([bounds, snapshot_utc])
    attic_dir = os.path.join(cache_dir, 'remainder_' + quote(key, safe=''))
    options = dict(
        prefer='overpass', overpass_timeout=180, overpass_rounds=1, budget_s=REMAINDER_BUDGET_S, optional=tuple(LAYERS)
    )
    frames = fetch_osm(tuple(bounds), snapshot_utc, attic_dir, layers=names[:3], **options)
    if len(names) > 3:  # area layers only where the remainder overlaps the area
        frames.update(fetch_osm(tuple(aoi_part.bounds), snapshot_utc, attic_dir, layers=names[3:], **options))
    return frames


def _fetch_remainder(main_region, remainder, aoi_box, cutoff, extract_date, snapshot_utc, write_aoi, cache_dir):
    """(layer names, frames, source log) for the part of the extent outside the main region's extract."""
    bounds = list(remainder.bounds)
    aoi_part = box(*bounds).intersection(aoi_box)
    names = ['roads', 'places', 'health']
    if write_aoi and not aoi_part.is_empty:
        names += ['buildings', 'bridges', 'waterways']
    frames, source_log = _remainder_from_geofabrik(
        main_region, remainder, bounds, cutoff, extract_date, names, cache_dir
    )
    if not frames:
        frames = _remainder_from_attic(bounds, aoi_part, snapshot_utc, names, cache_dir)
    return names, frames, source_log


def _merge_remainder(layers, frames, names, region_geometry):
    """Add the remainder features outside the main region to layers; returns the count added per layer."""
    added = {}
    for name in names:
        frame = frames.get(name)
        if frame is None or not len(frame):
            continue
        frame = frame[~frame.geometry.representative_point().within(region_geometry)]
        if len(frame):
            layers[name] = _concat_frames([layers[name], frame])
            added[name] = len(frame)
    return added


def _intersecting(frame, area):
    return frame[frame.intersects(area)] if len(frame) else frame


def fetch_osm_geofabrik(
    bbox, event_date, out_dir, cache_dir, net_bbox=None, write_aoi=True, cutoff_utc=None, want_date=None
):
    """Pre-event OSM from the latest Geofabrik dated extract before the cutoff, as (layers, log)."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    extent = [float(x) for x in (net_bbox or bbox)]
    cutoff = cutoff_utc or (event_date if 'T' in event_date else event_date + 'T00:00:00')
    region, url, extract_date = _main_extract(extent, cutoff[:10], want_date)
    pbf_path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(pbf_path):
        _download_extract(url, pbf_path)
    replication_ts = _replication_timestamp(pbf_path)
    _check_before_cutoff(replication_ts, extract_date, cutoff)
    snapshot_utc = replication_ts or _end_of_day_utc(extract_date)
    print('OSM snapshot', os.path.basename(url), 'replication timestamp', replication_ts, flush=True)
    layers = _read_layers(pbf_path, extent, cache_dir)
    extent_box, aoi_box = box(*extent), box(*bbox)
    remainder = extent_box.difference(region.geometry)
    fraction = remainder.area / extent_box.area
    remainder_log = dict(fraction=round(fraction, 4))
    warnings = []
    if fraction >= 0.01:
        names, frames, source_log = _fetch_remainder(
            region, remainder, aoi_box, cutoff, extract_date, snapshot_utc, write_aoi, cache_dir
        )
        added = _merge_remainder(layers, frames, names, region.geometry)
        missing = [name for name in names if name not in frames]
        remainder_log.update(source_log)
        remainder_log.update(bbox=list(remainder.bounds), snapshot=snapshot_utc, added=added, missing=missing)
        if missing:
            warnings.append(
                f'OSM: {100 * fraction:.0f}% of the analysed extent lies outside the {region.name} extract and its '
                f'{", ".join(missing)} could not be fetched: features there are missing.'
            )
    for warning in warnings:
        print('WARNING', warning, flush=True)
    log = dict(
        source=url,
        extract_date=extract_date,
        replication_timestamp=replication_ts,
        snapshot_utc=snapshot_utc,
        cutoff_utc=cutoff,
        region=region.name,
        region_level=region.level,
        region_cover=round(region.cover, 4),
        extent=extent,
        bbox=[float(x) for x in bbox],
        remainder=remainder_log,
        warnings=warnings,
    )
    written = {}
    if net_bbox:  # road network extent of the cut-off analysis
        net_log = dict(log, layers={})
        for name in NET_LAYERS:
            written['net_' + name] = _write_layer(layers[name], os.path.join(out_dir, f'net_{name}.gpkg'))
            net_log['layers'][name] = len(written['net_' + name])
            print('geofabrik', 'net_' + name, len(written['net_' + name]), flush=True)
        _write_json(net_log, os.path.join(out_dir, 'net_fetch_log.json'))
    if write_aoi:
        for name in AOI_LAYERS:
            frame = _intersecting(layers[name], aoi_box) if net_bbox or name == 'admin' else layers[name]
            written[name] = _write_layer(frame, os.path.join(out_dir, f'{name}.gpkg'))
            log[name] = len(written[name])
            print('geofabrik', name, log[name], flush=True)
        _write_json(log, os.path.join(out_dir, 'osm_fetch_log.json'))
        if not (log.get('roads') or log.get('buildings')):
            raise RuntimeError(
                'Geofabrik extract produced no roads/buildings for the bbox - falling back to another OSM source'
            )
    return written, log


def _snapshot_entry(log_path):
    """Source and snapshot time of one fetch log, or None when the log is missing or has no time."""
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        log = json.load(f)
    snapshot = log.get('snapshot_utc') or log.get('replication_timestamp')
    if not snapshot and log.get('extract_date'):
        snapshot = _end_of_day_utc(str(log['extract_date']))
    snapshot = snapshot or log.get('timestamp')
    if not snapshot:
        return None
    return dict(
        source=log.get('source') or 'overpass/ohsome',
        extract_date=log.get('extract_date'),
        replication_timestamp=log.get('replication_timestamp'),
        snapshot_utc=str(snapshot),
    )


def snapshot_record(osm_dir):
    """Snapshot of the area ('aoi') and network ('net') OSM in osm_dir, from the fetch logs."""
    return dict(
        aoi=_snapshot_entry(os.path.join(osm_dir, 'osm_fetch_log.json')),
        net=_snapshot_entry(os.path.join(osm_dir, 'net_fetch_log.json')),
    )


def check_snapshot(osm_dir, event_utc):
    """(ok, message); ok is False when an OSM snapshot does not end before event_utc ('YYYY-MM-DDTHH:MM:SS')."""
    record_path = os.path.join(osm_dir, 'snapshot.json')
    if os.path.exists(record_path):
        with open(record_path) as f:
            record = json.load(f)
    else:
        record = snapshot_record(osm_dir)
    late = []
    for key in ('aoi', 'net'):
        entry = record.get(key)
        if entry and str(entry.get('snapshot_utc', '')).replace('Z', '')[:19] >= str(event_utc)[:19]:
            late.append(f'{key}: {entry.get("snapshot_utc")} from {entry.get("source")}')
    if late:
        return False, (
            f'OSM snapshot is not before the event start {event_utc} UTC ({"; ".join(late)}): '
            'damage and cut-off results are not computed from OpenStreetMap edited after the event.'
        )
    if not record.get('aoi'):
        return True, (
            'OSM snapshot time could not be verified (no fetch log); the OpenStreetMap data is assumed to be pre-event.'
        )
    return True, ''
