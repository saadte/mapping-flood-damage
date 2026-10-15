"""Write flood maps, settlement lists and situation reports in English and Nepali."""

import base64
import csv
import datetime as dt
import html
import io
import json
import os
import re

import folium
import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.features import shapes
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from scipy import ndimage
from shapely.geometry import shape

import dashboard_ui
import impact
import sitrep_nepali
import store

MIN_POLYGON_HA = 0.2  # 20 pixels of 10 m; smaller hazard polygons are not drawn but count in the km2 totals
SOURCE_BITS = dict(s1=1, s2=2, dl=4, ai_optical=16, inferred=32)  # bits of hazard/sources.tif
OBSERVED_SOURCE_LABELS = dict(
    s1='radar change (S1)', s2='optical change (S2)', dl='radar water model (S1)', ai_optical='optical water model (S2)'
)
FOOTPRINT_COLORS = dict(likely='#d7191c', possible='#fdae61')
PARTS = ('hazard', 'buildings', 'roads', 'cutoff')
PART_LABELS = dict(
    hazard='Flood / debris map',
    buildings='Building damage',
    roads='Road and bridge damage',
    cutoff='Cut-off settlements',
)
PRIORITY_ROWS = 10

ACCESS_EN = dict(
    walk_in='on foot',
    air_only='no mapped road or trail',
    detour='long detour',
    road='road',
    no_road_pre='no road before the event',
)
ACCESS_SHORT_EN = dict(
    walk_in='on foot', air_only='no road/trail', detour='detour', road='road', no_road_pre='no road before'
)
CONFIDENCE_EN = dict(high='strong evidence', possible='possible', bridge='if bridges failed')
CONFIDENCE_SHORT_EN = dict(high='strong', possible='possible', bridge='if bridges failed')
ROAD_CLASS_EN = dict(major='highways / primary', secondary='secondary / tertiary', local='local roads', track='tracks')

NOT_COMPUTED_EN = 'not computed (see note above)'
ROAD_BREAK_RULE_EN = (
    'Road segments are at most 50 m long. A segment is *likely impassable* when at least 50% of it, '
    'or a continuous 50 m length, lies within the likely flood extent.'
)
STRONG_EVIDENCE_RULE_EN = (
    'A road break has *strong evidence* when optical change persists in later images, radar change occurs on two '
    'orbits, or radar and optical change overlap. The radar water model uses the same images as radar change '
    'and is counted with that radar evidence.'
)
ROAD_ACCESS_RULE_EN = 'a settlement has road access when a motorable road passes within 500 m'

# One A4 page: about 75 lines of 10.5 px text fit between 9 mm margins.
PAGE_CSS = (
    "body{font:10.5px/1.32 system-ui,'Segoe UI',sans-serif;max-width:820px;margin:10px auto}"
    'h2{font-size:15px;margin:0 0 2px}'
    'h3{font-size:12.5px;margin:6px 0 2px}ul{margin:1px 0;padding-left:16px}li{margin:0}p{margin:2px 0}'
    'table{border-collapse:collapse;margin:2px 0;width:100%;font-size:9.5px}'
    'td,th{border:1px solid #ccc;padding:0 3px;vertical-align:top}'
    'th{background:#f3f4f6}'
    'sub{font-size:8.5px;line-height:1.2;display:block}@page{size:A4;margin:9mm}'
)


def read_json(event_dir, relative_path):
    """Parsed JSON file of the event folder, or None when the file does not exist."""
    path = os.path.join(event_dir, relative_path)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs


def marker_position(geometry):
    """[lat, lon] of a point, or of a point inside a line or polygon."""
    point = geometry if geometry.geom_type == 'Point' else geometry.representative_point()
    return [point.y, point.x]


def s1_dates(event_dir):
    """Pre- and post-event dates per Sentinel-1 track; change detection uses all pre dates and the first two post."""
    by_track = {}
    for scene in read_json(event_dir, 's1/index.json') or []:
        by_track.setdefault(scene['track'], {'pre': [], 'post': []})[scene['role']].append(scene['day'])
    return {
        track: dict(pre=sorted(days['pre']), post=sorted(days['post'])[:2]) for track, days in sorted(by_track.items())
    }


def input_dates_text(event_dir, summary):
    """Sentinel-1 and Sentinel-2 dates used, as month-day text."""
    s1_text = (
        '; '.join(
            f'{track} (pre {len(days["pre"])} dates {days["pre"][0][5:]}–{days["pre"][-1][5:]}, '
            f'post {", ".join(day[5:] for day in days["post"])})'
            for track, days in s1_dates(event_dir).items()
            if days['pre'] and days['post']
        )
        or 'n/a'
    )
    s2 = summary.get('s2_days') or {}
    first_clear = s2['early'][0][5:] if s2.get('early') else 'none'
    s2_text = (
        f'pre {", ".join(day[5:] for day in s2.get("pre", [])) or "none"}; first clear after {first_clear}; '
        f'after {", ".join(day[5:] for day in s2.get("post", [])) or "none"}'
    )
    return s1_text, s2_text


def post_image_range(event_dir, summary):
    """First and newest post-event image dates used (Sentinel-1 post pairs, Sentinel-2 first-clear and after dates)."""
    s2 = summary.get('s2_days') or {}
    days = sorted(
        {day for dates in s1_dates(event_dir).values() for day in dates['post']}
        | set(s2.get('post') or [])
        | set(s2.get('early') or [])
    )
    return (days[0], days[-1]) if days else (None, None)


def data_years(event_dir, summary):
    """Years of the Sentinel data used."""
    years = {summary['date'][:4]}
    years |= {scene['day'][:4] for scene in read_json(event_dir, 's1/index.json') or []}
    s2 = summary.get('s2_days') or {}
    for key in ('pre', 'post', 'early'):
        years |= {day[:4] for day in s2.get(key, [])}
    return years


def uses_sen1floods11(event_dir):
    """Whether a model trained on Sen1Floods11 ran: a radar model in the DL config, or the optical model."""
    dl_config = read_json(event_dir, 'evidence/dl_config.json')
    has_radar_model = bool(dl_config) and any('s1f11' in str(run) for run in dl_config.get('runs') or [])
    return has_radar_model or bool(read_json(event_dir, 'evidence/ais2_config.json'))


def attribution(years, is_sen1floods11_used, is_dl_used):
    """Required data attribution and the training-data citation of the flood model."""
    training_data = 'Kuro Siwo (Bountos et al., 2024)'
    if is_sen1floods11_used:
        training_data += ' and Sen1Floods11 (Bonafilia et al., 2020)'
    if is_dl_used:
        model = f'Flood model trained on {training_data}.'
    else:
        model = f'AI flood model (trained on {training_data}) not used in this run.'
    return (
        f'Contains modified Copernicus Sentinel data {", ".join(sorted(years))}. Produced using Copernicus WorldDEM-30 '
        '© DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European '
        'Union and ESA; all rights reserved. © OpenStreetMap contributors (pre-event snapshot; derived OSM layers are '
        'ODbL). ' + model
    )


def hillshade_rgb(elevation, pixel_m):
    gy, gx = np.gradient(elevation, pixel_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth, altitude = np.radians(315), np.radians(45)
    shade = np.sin(altitude) * np.cos(slope) + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
    gray = np.clip(shade * 255, 0, 255).astype('uint8')
    return np.stack([gray] * 3, -1)


def stretched_rgb(bands):
    """Bands -> uint8 RGB stretched between the 2nd and 98th percentile of the non-zero values."""
    rgb = np.moveaxis(bands, 0, -1)
    valid = rgb[rgb > 0]
    low, high = (np.percentile(valid, 2), np.percentile(valid, 98)) if valid.size else (0, 1)
    return np.clip((rgb - low) / max(high - low, 1e-6) * 255, 0, 255).astype('uint8')


def background_overlay(tif, max_px, is_hillshade):
    """Raster -> (JPEG data URI in EPSG:3857, [[south, west], [north, east]]) for a Leaflet image overlay."""
    with rasterio.open(tif) as src:
        transform, width, height = calculate_default_transform(src.crs, 'EPSG:3857', src.width, src.height, *src.bounds)
        factor = max(1.0, max(width, height) / max_px)
        out_width, out_height = int(width / factor), int(height / factor)
        out_transform = transform * Affine.scale(width / out_width, height / out_height)
        band_indexes = [1] if is_hillshade else [1, 2, 3]
        bands = np.zeros((len(band_indexes), out_height, out_width), 'float32')
        for i, band in enumerate(band_indexes):
            reproject(
                src.read(band).astype('float32'),
                bands[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=out_transform,
                dst_crs='EPSG:3857',
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=0,
            )
    rgb = hillshade_rgb(bands[0], abs(out_transform.a)) if is_hillshade else stretched_rgb(bands)
    jpeg = io.BytesIO()
    Image.fromarray(rgb).save(jpeg, format='JPEG', quality=80)
    west, south, east, north = transform_bounds(
        'EPSG:3857',
        'EPSG:4326',
        out_transform.c,
        out_transform.f + out_height * out_transform.e,
        out_transform.c + out_width * out_transform.a,
        out_transform.f,
    )
    return 'data:image/jpeg;base64,' + base64.b64encode(jpeg.getvalue()).decode(), [[south, west], [north, east]]


def mask_polygons(mask, transform, crs, label):
    """Polygons of at least 5000 m2 in a boolean mask, simplified to 15 m; None when there are none."""
    rows = []
    for geometry, _ in shapes(mask.astype('uint8'), mask=mask, transform=transform):
        polygon = shape(geometry)
        if polygon.area >= 5000.0:
            rows.append(dict(layer=label, area_ha=round(polygon.area / 1e4, 2), geometry=polygon.simplify(15)))
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=crs) if rows else None


def model_only_layers(event_dir):
    """[(name, colour, dash, polygons)]: water predictions from each model and the extent estimated beneath cloud."""
    layers = []
    dl_path = os.path.join(event_dir, 'evidence', 'dl_flood_prob.tif')
    if os.path.exists(dl_path):
        probability, transform, crs = read_band(dl_path)
        hand_m, _, _ = read_band(os.path.join(event_dir, 'dem', 'hand10.tif'))
        flood_mask = (np.nan_to_num(probability, nan=0) > 0.5) & (hand_m < 40)
        polygons = mask_polygons(flood_mask, transform, crs, 'Radar water model: flood water, p > 0.5')
        if polygons is not None:
            layers.append(('Radar water model alone (Sentinel-1): flood water, low ground', '#2b8cbe', None, polygons))
    optical_path = os.path.join(event_dir, 'evidence', 'ais2_new.tif')
    if os.path.exists(optical_path):
        new_water, transform, crs = read_band(optical_path)
        polygons = mask_polygons(new_water == 1, transform, crs, 'Optical water model: new water')
        if polygons is not None:
            layers.append(
                ('Optical water model alone (Sentinel-2): new water', '#7b3294', None, polygons)
            )
    sources_path = os.path.join(event_dir, 'hazard', 'sources.tif')
    if os.path.exists(sources_path):
        sources, transform, crs = read_band(sources_path)
        inferred_mask = (sources & SOURCE_BITS['inferred']) > 0
        polygons = mask_polygons(inferred_mask, transform, crs, 'Flood extent estimated beneath cloud')
        if polygons is not None:
            layers.append(
                ('Flood extent estimated beneath cloud', '#e66101', '5,5', polygons)
            )
    return layers


def likely_evidence(source_share, label):
    """Evidence fields of one 'likely' polygon from the share of its pixels flagged by each source."""
    evidence = [text for key, text in OBSERVED_SOURCE_LABELS.items() if source_share[key][label] >= 0.1]
    observed_count = len(evidence)
    inferred_pct = int(100 * source_share['inferred'][label])
    if source_share['inferred'][label] >= 0.1:
        evidence.append(f'estimated beneath cloud ({inferred_pct}% of this area)')
    return dict(
        s1_pct=int(100 * source_share['s1'][label]),
        s2_pct=int(100 * source_share['s2'][label]),
        dl_pct=int(100 * source_share['dl'][label]),
        ai_optical_pct=int(100 * source_share['ai_optical'][label]),
        inferred_pct=inferred_pct,
        evidence=', '.join(evidence) or 'mixed',
        n_sources=observed_count,
    )


def hazard_polygons(hazard_tif, sources_tif):
    """Hazard raster -> polygons with class, area and, for 'likely', the share of each evidence source."""
    hazard, transform, crs = read_band(hazard_tif)
    sources, _, _ = read_band(sources_tif)
    rows = []
    for value, name in ((2, 'likely'), (1, 'possible')):
        labels, count = ndimage.label(hazard == value)
        if count == 0:
            continue
        pixel_count = np.bincount(labels.ravel(), minlength=count + 1).astype(float)
        source_share = {}
        if name == 'likely':
            for key, bit in SOURCE_BITS.items():
                flagged = ((sources & bit) > 0).ravel()
                source_share[key] = np.bincount(labels.ravel(), weights=flagged, minlength=count + 1) / np.maximum(
                    pixel_count, 1
                )
        for geometry, label in shapes(labels.astype('int32'), mask=labels > 0, transform=transform):
            polygon = shape(geometry)
            if polygon.area < MIN_POLYGON_HA * 1e4:
                continue
            row = dict(cls=name, area_ha=round(polygon.area / 1e4, 2), geometry=polygon.simplify(15))
            if name == 'likely':
                row.update(likely_evidence(source_share, int(label)))
            else:
                row.update(evidence='weaker / single-source evidence', n_sources=0)
            rows.append(row)
    if not rows:
        return gpd.GeoDataFrame(
            dict(cls=[], area_ha=[], evidence=[], n_sources=[]), geometry=gpd.GeoSeries([], crs=crs), crs=crs
        )
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=crs)


def shorten(text, max_chars):
    text = ' '.join(str(text).split())
    return text if len(text) <= max_chars else text[: max_chars - 1] + '…'


def ai_model_status(event_dir, summary):
    """(is_used, text). The model is used when its output existed when the hazard map was fused."""
    if (summary.get('fusion_params') or {}).get('dl_available'):
        runs = read_json(event_dir, 'evidence/dl_config.json')['runs']
        return True, f'U-Net ensemble trained from scratch ({", ".join(runs)})'
    errors = summary.get('errors') or {}
    if 'dl' in errors:
        return False, 'radar water prediction failed: ' + shorten(errors['dl'], 110)
    if summary.get('no_observation'):
        return False, 'no satellite observation for this area and date'
    if not (summary.get('data_availability') or {}).get('s1_tracks', ['?']):
        return False, 'no Sentinel-1 radar pair for this area and date (the model reads radar)'
    return False, 'no AI output for this run (no usable Sentinel-1 pair, or model weights unavailable)'


def impact_failure(summary):
    """Reason shared by all impact parts when the impact step gave no result, else None."""
    errors = summary.get('errors') or {}
    impacts = summary.get('impacts')
    if summary.get('no_observation'):
        return 'no satellite observation for this area and date'
    if 'impacts' in errors:
        return 'impact step failed: ' + shorten(errors['impacts'], 110)
    if 'osm' in errors and not impacts:
        return 'OpenStreetMap could not be fetched: ' + shorten(errors['osm'], 110)
    if not impacts:
        return 'the impact step did not run'
    return None


def product_availability(event_dir, summary):
    """Return missing-output reasons, model availability and recorded warnings."""
    errors = summary.get('errors') or {}
    availability = {}
    if 'fusion' in errors:
        availability['hazard'] = f'the flood map could not be built ({shorten(errors["fusion"], 100)})'
    elif not os.path.exists(os.path.join(event_dir, 'hazard', 'hazard.tif')):
        availability['hazard'] = 'the flood map could not be built'
    else:
        availability['hazard'] = None
    failure = impact_failure(summary)
    impact_status = (summary.get('impacts') or {}).get('availability') or {}
    for part, key in (('buildings', 'buildings'), ('roads', 'roads'), ('cutoff', 'q3')):
        if failure:
            availability[part] = failure
        else:
            availability[part] = None if impact_status[key] == 'ok' else impact_status[key]
    availability['dl'] = ai_model_status(event_dir, summary)
    banners = [f'{PART_LABELS[part]}: not computed ({availability[part]}).' for part in PARTS if availability[part]]
    is_dl_used, dl_text = availability['dl']
    if not is_dl_used:
        banners.append(f'Radar water model not used ({dl_text}).')
    availability['banners'] = banners
    warnings = []
    data_warnings = (summary.get('data_availability') or {}).get('warnings') or []
    for warning in list(summary.get('warnings') or []) + list(data_warnings):
        if warning and warning not in warnings:
            warnings.append(warning)
    availability['warnings'] = warnings
    return availability


def date_text(iso):
    """'2026-08-26' -> '26 Aug 2026'."""
    day = dt.date.fromisoformat(str(iso)[:10])
    return f'{day.day} {day:%b %Y}'


def number_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def rounded_or_none(value, digits=2):
    value = number_or_none(value)
    return None if value is None else round(value, digits)


def text_or_none(value):
    """Text value, or None for None, NaN and the 'None' / 'nan' strings left by GeoPackage round trips."""
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    text = str(value).strip()
    return None if text in ('', 'None', 'nan', 'NaN', '<NA>') else text


def area_text(municipality, ward):
    municipality, ward = text_or_none(municipality), text_or_none(ward)
    if municipality and ward:
        return ward if municipality.split(' ')[0].lower() in ward.lower() else f'{municipality}, {ward}'
    return municipality or ward


def footprint_figures(summary):
    hazard_km2 = summary.get('hazard_km2') or {}
    return dict(
        likely_km2=float(hazard_km2.get('likely', 0)),
        possible_km2=float(hazard_km2.get('possible', 0)),
        unobserved_km2=float(hazard_km2.get('unobserved', 0)),
        by_source_km2=dict(hazard_km2.get('likely_by_source') or {}),
    )


def building_figures(impacts):
    counts = impacts.get('buildings') or {}
    return dict(
        likely=int(counts.get('likely_affected', 0)),
        possible=int(counts.get('possibly_affected', 0)),
        unobserved=int(counts.get('unobserved', 0)),
        total=int(sum(counts.values())),
    )


def road_figures(impacts):
    vehicle_km = impacts.get('roads_km_vehicle') or {}
    path_km = impacts.get('paths_km') or {}
    return dict(
        vehicle_blocked_km=vehicle_km.get('blocked', 0),
        vehicle_possible_km=vehicle_km.get('possibly_blocked', 0),
        vehicle_total_km=impacts.get('roads_km_vehicle_total', 0),
        path_blocked_km=path_km.get('blocked', 0),
        path_possible_km=path_km.get('possibly_blocked', 0),
        path_total_km=impacts.get('paths_km_total', 0),
        by_class=impacts.get('roads_km_by_class') or {},
    )


def bridge_figures(impacts):
    vehicle = impacts.get('bridges_vehicle') or {}
    foot = impacts.get('bridges_foot') or {}
    return dict(
        vehicle_blocked=vehicle.get('blocked', 0),
        vehicle_possible=vehicle.get('possibly_blocked', 0),
        vehicle_total=impacts.get('bridges_vehicle_total', 0),
        foot_blocked=foot.get('blocked', 0),
        foot_possible=foot.get('possibly_blocked', 0),
        foot_total=impacts.get('bridges_foot_total', 0),
    )


def access_figures(cutoff):
    """Counts of the cut-off analysis (summary.json impacts.q3)."""
    buildings = cutoff.get('buildings') or {}
    units = cutoff.get('units') or {}
    settlements = cutoff.get('settlements') or {}
    named = settlements.get('named_by_status') or {}
    named_units = settlements.get('named_noroad_units')
    return dict(
        has_buildings=bool(buildings),
        has_footprint_class=bool(buildings.get('fp_class')),
        buildings_high=buildings.get('no_road_high', 0),
        buildings_any=buildings.get('no_road_high', 0)
        + buildings.get('no_road_possible', 0)
        + buildings.get('no_road_bridge', 0),
        units_high=units.get('noroad_high', 0),
        units_any=units.get('noroad_any', 0),
        air_high=buildings.get('air_only_high', 0),
        air_any=buildings.get('air_only', 0),
        trail_cut=buildings.get('trail_cut', 0),
        footprint_buildings=buildings.get('in_footprint', 0),
        unverified_buildings=buildings.get('unverified', 0),
        unverified_named=settlements.get('unverified_named', 0),
        outlying=units.get('outlying_noroad_bld', 0),
        partial=units.get('partial', 0),
        settlements_high=named.get('cut_off', 0),
        settlements_possible=named.get('possibly_cut_off', 0),
        settlements_bridge=named.get('bridge_uncertain', 0),
        settlements_detour=named.get('long_detour', 0),
        settlements_no_road_before=named.get('no_road_access_pre_event', 0) + named.get('not_connected_pre_event', 0),
        named_units=None
        if named_units is None
        else dict(
            high=named_units.get('high', 0),
            possible=named_units.get('possible', 0),
            bridge=named_units.get('bridge', 0),
        ),
        unnamed=settlements.get('unnamed_places', 0),
        cross_border=settlements.get('cross_border', 0),
        lost_exit=settlements.get('lost_exit_only', 0),
        lost_exit_names=settlements.get('lost_exit_only_names') or [],
        net=cutoff.get('net') or {},
    )


def health_figures(health):
    by_status = health.get('by_status') or {}
    return dict(
        total=health.get('total', 0),
        in_footprint=by_status.get('in_footprint', 0),
        possibly_in_footprint=by_status.get('possibly_in_footprint', 0),
        cut_off=by_status.get('cut_off', 0),
        possibly_cut_off=by_status.get('possibly_cut_off', 0),
        names_in_footprint=health.get('in_footprint') or [],
        names_cut_off=health.get('cut_off') or [],
    )


def report_context(event_dir, event, summary, availability):
    """Collect the counts, dates and settlement names used by the report and dashboard."""
    impacts = summary.get('impacts') or {}
    cutoff = impacts.get('q3') or {}
    shown_cutoff = {} if availability['cutoff'] else cutoff
    first_post_date, newest_date = post_image_range(event_dir, summary)
    is_dl_used = availability['dl'][0]
    return dict(
        event=event,
        title=dashboard_ui.pretty_event_name(event),
        date=summary['date'][:10],
        availability=availability,
        generated=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M'),
        areas=cutoff.get('areas') or {},
        bs_date_en=sitrep_nepali.bs_text_en(summary['date'][:10]),
        bs_date_ne=sitrep_nepali.bs_text_ne(summary['date'][:10]),
        first_post_date=first_post_date,
        newest_date=newest_date,
        footprint=None if availability['hazard'] else footprint_figures(summary),
        plain_km2=float((summary.get('fusion_params') or {}).get('plain_area_km2') or 0.0),
        buildings=None if availability['buildings'] else building_figures(impacts),
        roads=None if availability['roads'] else road_figures(impacts),
        bridges=None if availability['roads'] else bridge_figures(impacts),
        access=access_figures(shown_cutoff) if shown_cutoff else None,
        health=health_figures(shown_cutoff['health']) if shown_cutoff.get('health') else None,
        priority=shown_cutoff.get('priority') or [],
        n_priority=(shown_cutoff.get('units') or {}).get('n_priority', 0),
        hardest=shown_cutoff.get('hardest_hit') or [],
        repairs=shown_cutoff.get('repairs') or [],
        inputs=input_dates_text(event_dir, summary),
        osm_snapshot=summary.get('osm_snapshot'),
        dl=availability['dl'],
        attribution=attribution(data_years(event_dir, summary), uses_sen1floods11(event_dir), is_dl_used),
    )


def english_only(label):
    """'English (नेपाली)' -> 'English' for the narrow print table; the Nepali sitrep and dashboard keep both."""
    text = str(label)
    i = text.rfind(' (')
    if i > 0 and text.endswith(')') and any(0x900 < ord(ch) < 0x980 for ch in text[i:]):
        return text[:i]
    return text


def km_text(value):
    value = float(value or 0)
    return f'{value:,.1f}' if value < 100 else f'{value:,.0f}'


def table_cell(text):
    """Text for a markdown table cell; an OSM name with '|' must not split the cell."""
    return ' '.join(str(text).replace('|', '/').split())


def walk_text(row, is_short=False):
    if row.get('access') != 'walk_in' or row.get('walk_km') is None:
        return '—'
    hours = row.get('walk_h')
    text = f'{row["walk_km"]:.1f} km' + (f' ≈ {max(0.5, round(hours * 2) / 2):g} h' if hours is not None else '')
    if is_short:
        return text + (f' ({english_only(row["roadhead"])})' if row.get('roadhead') else '')
    return text + (f' from near {row["roadhead"]}' if row.get('roadhead') else ' from the road end')


def markdown_inline_html(text):
    """Escaped HTML of one markdown line, with *x* in bold."""
    return re.sub(r'\*([^*]+)\*', r'<b>\1</b>', html.escape(text))


def repair_text(repair):
    length_m = repair.get('length_m') or 0
    length = f'{length_m / 1000:.1f} km' if length_m >= 1000 else f'{length_m:.0f} m'
    road = repair.get('road') or ROAD_CLASS_EN.get(repair.get('highway'), repair.get('highway') or 'road')
    bridge = 'including a bridge ' if repair.get('bridge') else ''
    near = f' near {repair["near"]}' if repair.get('near') else ''
    settlement_count = repair.get('n_settlements') or 0
    settlements = ''
    if settlement_count:
        settlements = f' ({settlement_count} named settlement{"s" if settlement_count != 1 else ""})'
    gain = int(repair.get('gain') or 0)
    return f'{length} {bridge}on {road}{near} → reconnects {gain:,} {repair.get("gain_unit", "buildings")}{settlements}'


def plain_text_en(plain_km2):
    """Describe the flood map on large plains."""
    if plain_km2 < 1:
        return ''
    return (
        f'The water models provide the flood extent on {plain_km2:,.0f} km² of large low plain in this area. '
        'Water present before the flood can show little change, and drainage positions from the elevation model '
        'are unreliable on plains, so the change masks and river-connection requirement are not used there. '
    )


def inferred_text_en(footprint):
    """Describe the flood extent estimated beneath cloud."""
    km2 = float(((footprint or {}).get('by_source_km2') or {}).get('continuity') or 0.0)
    if km2 < 0.005:
        return ''
    return (
        f'Over {km2:.1f} km² of cloud-covered valley ground, the likely extent is obtained by including ground '
        'below the flood level estimated along the river. These areas are marked *inferred* in the map and '
        'are excluded when assessing strong evidence of a road break. '
    )


def header_lines_en(context):
    areas = context['areas']
    where = ', '.join(english_only(name) for name in (areas.get('districts') or [])[:3]) or ', '.join(
        english_only(name) for name in (areas.get('municipalities') or [])[:3]
    )
    title = f'# Situation report — {context["title"]}' + (f' — {where}' if where else '')
    event_date = date_text(context['date'])
    generated = f'generated {context["generated"]} UTC'
    if not context['newest_date']:
        return [
            title,
            f'*No post-event satellite image was usable · event {event_date} · {generated}.*',
            '',
        ]
    bs_date = f' ({context["bs_date_en"]})' if context['bs_date_en'] else ''
    subtitle = (
        f'*As of satellite images up to **{date_text(context["newest_date"])}** (first image after the event: '
        f'{date_text(context["first_post_date"])}) · event {event_date}{bs_date} · {generated} · infrastructure '
        'status has not been checked on the ground.*'
    )
    return [title, subtitle, '']


def notes_en(availability):
    lines = [f'> **{PART_LABELS[part]} not computed:** {availability[part]}.' for part in PARTS if availability[part]]
    is_dl_used, dl_text = availability['dl']
    if not is_dl_used:
        lines.append(f'> **Radar water model not used:** {dl_text}.')
    return lines + [f'> **Data warning:** {warning}' for warning in availability['warnings']]


def settlements_suffix(unit_count):
    if not unit_count:
        return ''
    return f' in {unit_count} settlement' + ('s' if unit_count != 1 else '')


def access_rows_en(access):
    if access is None:
        return [('Without road access', NOT_COMPUTED_EN, '')]
    if not access['has_buildings']:
        any_count = access['settlements_high'] + access['settlements_possible'] + access['settlements_bridge']
        return [('Settlements without road access', f'**{access["settlements_high"]}**', f'{any_count}')]
    label = 'Buildings without road access' + (' (outside the flood extent)' if access['has_footprint_class'] else '')
    return [
        (
            label,
            f'**{access["buildings_high"]:,}**' + settlements_suffix(access['units_high']),
            f'{access["buildings_any"]:,}' + settlements_suffix(access['units_any']),
        ),
        ('Buildings with no road or trail connection', f'{access["air_high"]:,}', f'{access["air_any"]:,}'),
    ]


def key_figures_en(context):
    footprint, buildings, roads, bridges, health = (
        context[key] for key in ('footprint', 'buildings', 'roads', 'bridges', 'health')
    )
    rows = []
    if footprint:
        likely_km2, possible_km2 = footprint['likely_km2'], footprint['possible_km2']
        rows.append(('Flood / debris extent', f'**{likely_km2:,.1f} km²**', f'{likely_km2 + possible_km2:,.1f} km²'))
    else:
        rows.append(('Flood / debris extent', NOT_COMPUTED_EN, ''))
    if buildings:
        rows.append(
            (
                'Buildings in the flood extent',
                f'**{buildings["likely"]:,}** of {buildings["total"]:,} mapped',
                f'{buildings["likely"] + buildings["possible"]:,}',
            )
        )
    else:
        rows.append(('Buildings in the flood extent', NOT_COMPUTED_EN, ''))
    if roads:
        rows += [
            (
                'Motorable roads likely impassable',
                f'**{km_text(roads["vehicle_blocked_km"])} km** of {km_text(roads["vehicle_total_km"])} km',
                f'{km_text(roads["vehicle_blocked_km"] + roads["vehicle_possible_km"])} km',
            ),
            (
                'Footpaths and trails affected',
                f'{km_text(roads["path_blocked_km"])} km of {km_text(roads["path_total_km"])} km',
                f'{km_text(roads["path_blocked_km"] + roads["path_possible_km"])} km',
            ),
        ]
    else:
        rows.append(('Roads likely impassable', NOT_COMPUTED_EN, ''))
    if bridges:
        rows.append(
            (
                'Bridges with damage evidence: motorable · foot',
                f'**{bridges["vehicle_blocked"]}** of {bridges["vehicle_total"]} · '
                f'**{bridges["foot_blocked"]}** of {bridges["foot_total"]}',
                f'{bridges["vehicle_blocked"] + bridges["vehicle_possible"]} · '
                f'{bridges["foot_blocked"] + bridges["foot_possible"]}',
            )
        )
    if health:
        rows.append(
            (
                'Health facilities in the flood extent',
                f'**{health["in_footprint"]}** of {health["total"]}',
                f'{health["in_footprint"] + health["possibly_in_footprint"]}',
            )
        )
    rows += access_rows_en(context['access'])
    return ['## Key figures', '| | Likely | Likely + possible |', '|---|---|---|'] + [
        f'| {label} | {likely} | {any_value} |' for label, likely, any_value in rows
    ]


def key_messages_en(context):
    access, health = context['access'], context['health']
    messages = []
    if context['hardest']:
        messages.append(
            '**Buildings within the flood extent:** '
            + ', '.join(f'{unit["label"]} ({unit["n_fp"]})' for unit in context['hardest'][:4])
            + '.'
        )
    if context['repairs']:
        messages.append(
            '**Road repair candidates:** '
            + ' '.join(f'#{repair["rank"]} {repair_text(repair)}.' for repair in context['repairs'][:3])
        )
    if health and (health['names_in_footprint'] or health['names_cut_off']):
        parts = []
        if health['names_in_footprint']:
            parts.append('in the flood extent: ' + ', '.join(health['names_in_footprint'][:4]))
        if health['names_cut_off']:
            parts.append('cut off: ' + ', '.join(health['names_cut_off'][:4]))
        messages.append('**Health facilities:** ' + '; '.join(parts) + '.')
    if access and access['lost_exit']:
        examples = f' (e.g. {", ".join(access["lost_exit_names"][:3])})' if access['lost_exit_names'] else ''
        messages.append(
            f'**{access["lost_exit"]} settlements** can still reach a local town or health facility but have lost the '
            f'main road out of the area{examples}.'
        )
    if not messages:
        return []
    return ['## Affected settlements'] + ['- ' + message for message in messages]


def priority_table_en(context):
    access = context['access']
    rows = context['priority'][:PRIORITY_ROWS]
    if not access or not rows:
        return []
    in_footprint = ' (+ in extent)' if access['has_footprint_class'] else ' (in footprint)'
    lines = [
        f'## Priority: settlements without road access (top {PRIORITY_ROWS} by buildings)',
        f'| # | Settlement (area) | Buildings without road{in_footprint} | Access now | Walk from the road '
        '| Evidence |',
        '|---|---|---|---|---|---|',
    ]
    for rank, unit in enumerate(rows, 1):
        area = area_text(unit.get('muni'), unit.get('ward'))
        name = table_cell(english_only(unit['label'])) + (f' ({table_cell(english_only(area))})' if area else '')
        access_now = ACCESS_SHORT_EN.get(unit.get('access'), '—')
        evidence = CONFIDENCE_SHORT_EN.get(unit.get('conf'), '—') + (' †' if unit.get('spof') else '')
        lines.append(
            f'| {rank} | {name} | {unit["n_noroad"]:,} ({unit["n_fp"]:,}) | {access_now} | '
            f'{table_cell(walk_text(unit, is_short=True))} | {evidence} |'
        )
    more = max(0, int(context['n_priority'] or 0) - len(rows))
    footnote = f'{more} more are in the dashboard (Priorities tab) and in priorities.csv. ' if more else ''
    if any(unit.get('spof') for unit in rows):
        footnote += '† A single uncertain break decides access: check that crossing first. '
    lines.append(f'*{footnote}Walking times are estimates from the elevation model.*')
    return lines


def damage_lines_en(context):
    buildings, roads, bridges, health = (context[key] for key in ('buildings', 'roads', 'bridges', 'health'))
    lines = ['## Damage (counted on pre-event OpenStreetMap)']
    if buildings:
        lines.append(
            f'- Buildings: **{buildings["likely"]:,} likely** in the flood extent, {buildings["possible"]:,} possibly, '
            f'{buildings["unobserved"]:,} not observable; {buildings["total"]:,} mapped in total. These counts show '
            'exposure; the damage is not verified.'
        )
    else:
        lines.append(f'- Buildings: {NOT_COMPUTED_EN}.')
    if roads and roads['by_class']:
        lines.append(
            '- Motorable roads likely impassable by class: '
            + '; '.join(
                f'{ROAD_CLASS_EN.get(road_class, road_class)} {km_text(km.get("blocked", 0))} of '
                f'{km_text(km.get("total", 0))} km'
                for road_class, km in roads['by_class'].items()
            )
            + '.'
        )
    elif not roads:
        lines.append(f'- Roads and bridges: {NOT_COMPUTED_EN}.')
    if bridges:
        lines.append(
            '- Bridges (one count per OSM bridge way, judged by the road on each side): motorable '
            f'{bridges["vehicle_blocked"]} likely and {bridges["vehicle_possible"]} possibly damaged of '
            f'{bridges["vehicle_total"]}; foot and suspension {bridges["foot_blocked"]} likely and '
            f'{bridges["foot_possible"]} possibly of {bridges["foot_total"]}.'
        )
    if health:
        names = f'; cut off: {", ".join(health["names_cut_off"][:5])}' if health['names_cut_off'] else ''
        lines.append(
            f'- Health facilities ({health["total"]} in the area): {health["in_footprint"]} in the flood extent, '
            f'{health["cut_off"]} cut off by road (+{health["possibly_cut_off"]} possibly){names}.'
        )
    return lines


def cutoff_lines_en(access):
    if not access:
        return []
    net = access['net']
    buffer_km = net.get('buffer_km')
    if buffer_km:
        extent = f'The network reaches {buffer_km:g} km beyond the area; roads outside the area count as open.'
    else:
        extent = 'The network covers only the area.'
    lines = [
        '## Who is cut off',
        '- A settlement is *cut off* when no road route leads from it to a town, a hospital or a main road leaving the '
        f'analysed network. {extent} Destinations: {net.get("n_hospitals", 0)} hospitals, '
        f'{net.get("n_towns", 0)} towns, {net.get("n_exits", 0)} main-road exits; {ROAD_ACCESS_RULE_EN}.',
    ]
    trail = f' ({access["trail_cut"]:,} buildings there also lost their trail)' if access['trail_cut'] else ''
    rest = (
        f'{access["settlements_detour"]} with a long detour; {access["settlements_no_road_before"]} had no road before '
        f'the event{trail}.'
    )
    units = access['named_units']
    if units is not None:
        partial = f'; {access["partial"]} more in part' if access['partial'] else ''
        lines.append(
            '- Named settlements without road access (the settlement point, or most of its buildings outside the '
            f'footprint): **{units["high"]}** with strong evidence, {units["possible"]} possibly, {units["bridge"]} if '
            f'bridges with water under the span also failed{partial}; {rest}'
        )
        if access['footprint_buildings']:
            lines.append(
                f'- Buildings within the flood extent ({access["footprint_buildings"]:,}) are counted separately '
                'from buildings without road access.'
            )
    else:
        lines.append(
            f'- Named settlements without road access: **{access["settlements_high"]}** with strong evidence, '
            f'{access["settlements_possible"]} possibly, {access["settlements_bridge"]} if bridges with water under '
            f'the span also failed; {rest}'
        )
    outlying = f'; {access["outlying"]:,} outlying buildings without road access' if access['outlying'] else ''
    cross_border = (
        f' {access["cross_border"]} places across the national border were not assessed.'
        if access['cross_border']
        else ''
    )
    lines.append(
        f'- Not listed by name: {access["unnamed"]} unnamed OSM place points (their buildings count with the nearest '
        f'named settlement){outlying}.{cross_border}'
    )
    return lines


def confidence_lines_en(context):
    footprint, access = context['footprint'], context['access']
    newest = date_text(context['newest_date']) if context['newest_date'] else 'the images used'
    unobserved = f' ({footprint["unobserved_km2"]:.1f} km² could not be observed)' if footprint else ''
    unverified = ''
    if access and access['unverified_buildings']:
        unverified = (
            f' **Access unverified:** {access["unverified_buildings"]:,} buildings ({access["unverified_named"]} named '
            'settlements) count as connected only through roads that could not be observed.'
        )
    buffer_km = (access or {}).get('net', {}).get('buffer_km')
    network = f' (road network +{buffer_km:g} km)' if buffer_km else ''
    is_dl_used, dl_text = context['dl']
    s1_text, s2_text = context['inputs']
    return [
        '## Observations and uncertainty',
        '- The likely extent uses radar change, optical change and water predictions on low ground near rivers. '
        'Mountain detections must connect to a river channel. Possible detections include weaker change or '
        'water predictions with less support. '
        + plain_text_en(context['plain_km2'])
        + inferred_text_en(footprint)
        + ROAD_BREAK_RULE_EN
        + ' '
        + STRONG_EVIDENCE_RULE_EN,
        f'- The most recent images are from {newest}. Cloud and radar shadow leave some ground without '
        f'observations{unobserved}. Buildings and roads absent from pre-event OpenStreetMap are omitted. '
        'Road status is calculated from overlap with the flood extent, and bridges are assessed from their approaches.'
        + unverified,
        '',
        f'<sub>S1: {s1_text} · S2: {s2_text} · OSM: {context["osm_snapshot"]}{network} · '
        f'Radar water model: {"used" if is_dl_used else "not used (" + dl_text + ")"} · '
        f'Polygons under {MIN_POLYGON_HA:g} ha are not drawn.</sub>',
        f'<sub>{context["attribution"]}</sub>',
    ]


def sitrep_en(context):
    """English one-page situation report (markdown)."""
    lines = header_lines_en(context) + notes_en(context['availability'])
    lines += key_figures_en(context)
    lines += key_messages_en(context)
    lines += priority_table_en(context)
    lines += damage_lines_en(context)
    lines += cutoff_lines_en(context['access'])
    lines += confidence_lines_en(context)
    return '\n'.join(lines)


def write_text(path, text, encoding='utf-8', newline=None):
    with open(path, 'w', encoding=encoding, newline=newline) as f:
        f.write(text)


def write_sitreps(report_dir, context):
    """sitrep.md / .html (English) and sitrep_ne.md / .html (Nepali); returns both markdown texts."""
    md_en = sitrep_en(context)
    write_text(os.path.join(report_dir, 'sitrep.md'), md_en)
    write_text(
        os.path.join(report_dir, 'sitrep.html'),
        f"<!doctype html><html><head><meta charset='utf-8'><title>Sitrep</title><style>{PAGE_CSS}</style></head><body>"
        + dashboard_ui.markdown_html(md_en)
        + '</body></html>',
    )
    md_ne = sitrep_nepali.markdown(context)
    write_text(os.path.join(report_dir, 'sitrep_ne.md'), md_ne)
    write_text(
        os.path.join(report_dir, 'sitrep_ne.html'),
        "<!doctype html><html lang='ne'><head><meta charset='utf-8'><title>स्थिति प्रतिवेदन</title><style>"
        + PAGE_CSS.replace("system-ui,'Segoe UI'", "'Noto Sans Devanagari','Nirmala UI',system-ui")
        + '</style></head><body>'
        + dashboard_ui.markdown_html(md_ne)
        + '</body></html>',
    )
    return md_en, md_ne


PRIORITIES_COLUMNS = (
    ('rank', '#x_rank'),
    ('settlement', '#loc+name'),
    ('settlement_ne', '#loc+name+i_ne'),
    ('municipality', '#adm3+name'),
    ('ward', '#adm4+name'),
    ('lat', '#geo+lat'),
    ('lon', '#geo+lon'),
    ('buildings_total', '#x_buildings+total'),
    ('buildings_in_footprint', '#x_buildings+affected'),
    ('buildings_no_road', '#x_buildings+noroad'),
    ('buildings_no_road_strong', '#x_buildings+noroad+strong'),
    ('access_now', '#x_access'),
    ('walk_km', '#x_walk+km'),
    ('walk_h', '#x_walk+hours'),
    ('road_end_near', '#loc+roadhead'),
    ('evidence', '#x_confidence'),
)
SETTLEMENTS_COLUMNS = (
    ('id', '#loc+id'),
    ('settlement', '#loc+name'),
    ('settlement_ne', '#loc+name+i_ne'),
    ('place', '#loc+type'),
    ('status', '#x_status'),
    ('access_now', '#x_access'),
    ('municipality', '#adm3+name'),
    ('ward', '#adm4+name'),
    ('lat', '#geo+lat'),
    ('lon', '#geo+lon'),
    ('road_km_before', '#x_road+km+before'),
    ('road_km_after', '#x_road+km+after'),
    ('walk_km', '#x_walk+km'),
    ('walk_h', '#x_walk+hours'),
    ('nearest_target', '#loc+target'),
    ('buildings_total', '#x_buildings+total'),
    ('buildings_in_footprint', '#x_buildings+affected'),
    ('buildings_no_road', '#x_buildings+noroad'),
)
BREAKS_COLUMNS = (
    ('break_id', '#x_break+id'),
    ('repair_rank', '#x_rank'),
    ('lat', '#geo+lat'),
    ('lon', '#geo+lon'),
    ('length_m', '#x_length+m'),
    ('bridge', '#x_bridge'),
    ('strong_evidence', '#x_confidence'),
    ('road_class', '#x_road+type'),
    ('road', '#x_road+name'),
    ('near', '#loc+name'),
    ('buildings_reconnected_if_repaired_alone', '#x_buildings+reconnected'),
    ('settlements_routing_through', '#x_settlements+count'),
    ('settlements', '#loc+name+list'),
)


def is_blank(value):
    if value is None:
        return True
    if isinstance(value, float) and not np.isfinite(value):
        return True
    return isinstance(value, str) and text_or_none(value) is None


def csv_text(columns, rows):
    """CSV with the HXL hashtag row under the header."""
    text = io.StringIO()
    writer = csv.writer(text, lineterminator='\r\n')
    writer.writerow([name for name, _ in columns])
    writer.writerow([tag for _, tag in columns])
    for row in rows:
        writer.writerow(['' if is_blank(value) else value for value in row])
    return text.getvalue()


def named_units(units):
    """Settlement units keyed by settlement id (named units only)."""
    if units is None:
        return {}
    return {str(unit['uid'])[1:]: unit for _, unit in units.iterrows() if str(unit['uid']).startswith('U')}


def priorities_csv(units):
    selected = units[
        (units['n_noroad'] > 0) | units['point_status'].isin(['cut_off', 'possibly_cut_off', 'bridge_uncertain'])
    ]
    selected = selected.sort_values(['n_noroad', 'n_noroad_high', 'n_bld'], ascending=False)
    rows = []
    for rank, (_, unit) in enumerate(selected.iterrows(), 1):
        rows.append(
            [
                rank,
                str(unit['label']),
                sitrep_nepali.label_ne(str(unit['label']), unit['place'], text_or_none(unit.get('near'))),
                text_or_none(unit.get('adm_muni')),
                text_or_none(unit.get('adm_ward')),
                round(unit.geometry.y, 5),
                round(unit.geometry.x, 5),
                int(unit['n_bld']),
                int(unit['n_fp']),
                int(unit['n_noroad']),
                int(unit['n_noroad_high']),
                text_or_none(unit.get('access')),
                rounded_or_none(unit.get('walk_km'), 1),
                rounded_or_none(unit.get('walk_h'), 1),
                text_or_none(unit.get('roadhead')),
                CONFIDENCE_EN.get(text_or_none(unit.get('conf')), text_or_none(unit.get('conf'))),
            ]
        )
    return csv_text(PRIORITIES_COLUMNS, rows)


def settlements_csv(settlements, units):
    unit_by_sid = named_units(units)
    rows = []
    for _, place in settlements.iterrows():
        name = text_or_none(place.get('label')) or text_or_none(place.get('name'))
        if not name or name == 'unnamed':
            continue
        unit = unit_by_sid.get(str(place.get('sid')))
        rows.append(
            [
                text_or_none(place.get('sid')),
                name,
                sitrep_nepali.local_name(name),
                text_or_none(place.get('place')),
                text_or_none(place.get('status')),
                text_or_none(place.get('access')),
                text_or_none(place.get('adm_muni')),
                text_or_none(place.get('adm_ward')),
                round(place.geometry.y, 5),
                round(place.geometry.x, 5),
                rounded_or_none(place.get('dist_pre_km')),
                rounded_or_none(place.get('dist_post_km')),
                rounded_or_none(place.get('walk_km'), 1),
                rounded_or_none(place.get('walk_h'), 1),
                text_or_none(place.get('target')),
                None if unit is None else int(unit['n_bld']),
                None if unit is None else int(unit['n_fp']),
                None if unit is None else int(unit['n_noroad']),
            ]
        )
    return csv_text(SETTLEMENTS_COLUMNS, rows)


def breaks_csv(runs, repairs):
    repair_rank = {}
    for repair in repairs:
        for run in str(repair.get('runs', '')).split(','):
            if run:
                repair_rank.setdefault(int(run), repair['rank'])
    rows = [
        [
            int(run['run']),
            repair_rank.get(int(run['run'])),
            round(run.geometry.y, 5),
            round(run.geometry.x, 5),
            rounded_or_none(run['length_m'], 0),
            bool(run['bridge']),
            bool(run['strong']),
            text_or_none(run.get('highway')),
            text_or_none(run.get('road')),
            text_or_none(run.get('near')),
            int(number_or_none(run.get('gain_alone')) or 0),
            int(number_or_none(run.get('n_settlements_route')) or 0),
            text_or_none(run.get('settlements_route')),
        ]
        for _, run in runs.sort_values('gain_alone', ascending=False).iterrows()
    ]
    return csv_text(BREAKS_COLUMNS, rows)


def csv_exports(context, layers):
    """{file name: text} of priorities.csv, settlements.csv and breaks.csv, for the parts that exist."""
    exports = {}
    if layers['units'] is not None:
        exports['priorities.csv'] = priorities_csv(layers['units'])
    if layers['settlements'] is not None and len(layers['settlements']):
        exports['settlements.csv'] = settlements_csv(layers['settlements'], layers['units'])
    if layers['runs'] is not None:
        exports['breaks.csv'] = breaks_csv(layers['runs'], context['repairs'])
    return exports


def settlement_entries(settlements, units):
    """Settlements for the dashboard search, cards and markers (short keys keep the page small)."""
    if settlements is None:
        return []
    unit_by_sid = named_units(units)
    entries = []
    for _, place in settlements.iterrows():
        name = text_or_none(place.get('label')) or text_or_none(place.get('name'))
        if not name or name == 'unnamed':
            continue
        unit = unit_by_sid.get(str(place.get('sid')))
        y, x = marker_position(place.geometry)
        walk = walk_text(
            dict(
                access=text_or_none(place.get('access')),
                walk_km=rounded_or_none(place.get('walk_km'), 1),
                walk_h=rounded_or_none(place.get('walk_h'), 1),
                roadhead=None if unit is None else text_or_none(unit.get('roadhead')),
            )
        )
        entries.append(
            dict(
                k=str(place['sid']),
                l=name,
                n=sitrep_nepali.local_name(name),
                a=area_text(place.get('adm_muni'), place.get('adm_ward')),
                y=round(y, 5),
                x=round(x, 5),
                s=str(place.get('status')),
                c=text_or_none(place.get('access')),
                p=str(place.get('place')),
                b=None if unit is None else int(unit['n_bld']),
                f=None if unit is None else int(unit['n_fp']),
                r=None if unit is None else int(unit['n_noroad']),
                w=None if walk == '—' else walk,
                d0=rounded_or_none(place.get('dist_pre_km'), 1),
                d1=rounded_or_none(place.get('dist_post_km'), 1),
                t=text_or_none(place.get('target')),
            )
        )
    return entries


def unit_key(uid):
    """Settlement id of a named unit ('U<sid>'), else a key that matches no settlement."""
    return str(uid)[1:] if str(uid).startswith('U') else 'unit:' + str(uid)


def priority_entries(priority):
    entries = []
    for unit in priority:
        walk = walk_text(unit)
        entries.append(
            dict(
                k=unit_key(unit.get('uid')),
                l=unit['label'],
                n=sitrep_nepali.label_ne(unit['label'], unit.get('place'), unit.get('near')),
                a=area_text(unit.get('muni'), unit.get('ward')),
                y=unit.get('lat'),
                x=unit.get('lon'),
                nb=unit.get('n_noroad'),
                nf=unit.get('n_fp'),
                nh=unit.get('n_high'),
                c=unit.get('access'),
                q=unit.get('conf'),
                w=None if walk == '—' else walk,
                spof=bool(unit.get('spof')),
                t=unit.get('target'),
                fb=unit.get('first_break_km'),
            )
        )
    return entries


def dashboard_data(context, layers, csvs):
    """JSON embedded in the dashboard (priorities, search index, cards, exports); the browser escapes the strings."""
    hardest = [
        dict(
            l=unit['label'],
            y=unit.get('lat'),
            x=unit.get('lon'),
            f=unit.get('n_fp'),
            b=unit.get('n_bld'),
            k=str(unit['uid'])[1:] if str(unit.get('uid', '')).startswith('U') else None,
        )
        for unit in context['hardest']
    ]
    repairs = [
        dict(
            r=repair['rank'],
            txt=repair_text(repair),
            y=repair.get('lat'),
            x=repair.get('lon'),
            g=repair.get('gain'),
            br=bool(repair.get('bridge')),
        )
        for repair in context['repairs']
    ]
    health = []
    if layers['health'] is not None:
        for _, facility in layers['health'][layers['health']['in_aoi'].astype(bool)].iterrows():
            health.append(
                dict(
                    l=str(facility['label']),
                    kind=str(facility['kind']),
                    s=str(facility['status']),
                    y=round(facility.geometry.y, 5),
                    x=round(facility.geometry.x, 5),
                )
            )
    availability = context['availability']
    priority = priority_entries(context['priority'])
    return dict(
        prio=priority,
        prio_msg=f'Cut-off analysis not computed ({availability["cutoff"]}).' if availability['cutoff'] else None,
        prio_more=max(0, int(context['n_priority'] or 0) - len(priority)),
        hard=hardest,
        rep=repairs,
        health=health,
        settl=settlement_entries(layers['settlements'], layers['units']),
        csv=csvs,
        banners=availability['banners'],
        warnings=availability['warnings'],
        fpc=bool((context['access'] or {}).get('has_footprint_class')),
    )


def add_backgrounds(event_map, event_dir):
    """Add the optical composites and DEM hillshade from this run."""
    backgrounds = [
        (os.path.join(event_dir, 'dem', 'dem30_buf.tif'), 'Terrain (Copernicus DEM hillshade)', True, True),
        (os.path.join(event_dir, 's2', 'pre_rgb.tif'), 'Sentinel-2 before (composite)', False, False),
        (os.path.join(event_dir, 's2', 'early_rgb.tif'), 'Sentinel-2 first clear image after', False, False),
        (os.path.join(event_dir, 's2', 'post_rgb.tif'), 'Sentinel-2 after (composite)', False, False),
    ]
    for path, name, is_hillshade, show in backgrounds:
        if os.path.exists(path):
            uri, bounds = background_overlay(path, max_px=2400, is_hillshade=is_hillshade)  # ~35 m pixels
            folium.raster_layers.ImageOverlay(uri, bounds, name=name, opacity=1.0, show=show, overlay=True).add_to(
                event_map
            )


def add_extents(event_map, summary):
    bbox = summary['bbox']
    area = folium.FeatureGroup(name='Analysed area')
    folium.Rectangle(
        [[bbox[1], bbox[0]], [bbox[3], bbox[2]]], color='#111827', weight=1.5, fill=False, dash_array='2,4'
    ).add_to(area)
    area.add_to(event_map)
    net = ((summary.get('impacts') or {}).get('q3') or {}).get('net') or {}
    osm_net = summary.get('osm_net') or {}
    if osm_net.get('bbox') and net.get('network') == 'buffered':
        net_bbox = osm_net['bbox']
        extent = folium.FeatureGroup(
            name=f'Road network for the cut-off analysis (area + {osm_net.get("buffer_km", 0):g} km)', show=False
        )
        folium.Rectangle(
            [[net_bbox[1], net_bbox[0]], [net_bbox[3], net_bbox[2]]],
            color='#6b7280',
            weight=1,
            fill=False,
            dash_array='6,6',
        ).add_to(extent)
        extent.add_to(event_map)


def footprint_style(feature):
    color = FOOTPRINT_COLORS[feature['properties']['cls']]
    return dict(color=color, fillColor=color, weight=0.5, fillOpacity=0.55)


def add_footprint(event_map, hazard):
    hazard = hazard.to_crs(4326)
    for footprint_class, name, show in (
        ('possible', 'Flood / debris extent - possible', False),
        ('likely', 'Flood / debris extent - likely', True),
    ):
        polygons = hazard[hazard['cls'] == footprint_class]
        if not len(polygons):
            continue
        fields = [
            column
            for column in ('cls', 'area_ha', 'evidence', 's1_pct', 's2_pct', 'dl_pct', 'ai_optical_pct', 'inferred_pct')
            if column in polygons and polygons[column].notna().any()
        ]
        folium.GeoJson(
            polygons[[*fields, 'geometry']].fillna(''),
            name=name,
            show=show,
            style_function=footprint_style,
            tooltip=folium.GeoJsonTooltip(fields),
        ).add_to(event_map)


def add_model_only_layers(event_map, event_dir):
    for name, color, dash, polygons in model_only_layers(event_dir):
        folium.GeoJson(
            polygons.to_crs(4326),
            name=name,
            show=False,
            style_function=lambda _feature, color=color, dash=dash: dict(
                color=color, fillColor=color, weight=1.2, fillOpacity=0.35, dashArray=dash
            ),
            tooltip=folium.GeoJsonTooltip(['layer', 'area_ha']),
        ).add_to(event_map)


def escape_text_properties(geojson):
    for feature in geojson.get('features', []):
        properties = feature['properties']
        properties.update({key: html.escape(str(value)) for key, value in properties.items() if isinstance(value, str)})


def add_roads(event_map, report_dir):
    for file_name, name, style, show in (
        ('roads_blocked.geojson', 'Roads likely impassable', dict(color='#000000', weight=4), True),
        (
            'roads_possibly_blocked.geojson',
            'Roads possibly impassable',
            dict(color='#555555', weight=3, dashArray='6,6'),
            False,
        ),
    ):
        path = os.path.join(report_dir, file_name)
        if not os.path.exists(path):
            continue
        geojson = read_json(report_dir, file_name)
        if not geojson.get('features'):
            continue
        escape_text_properties(geojson)
        properties = geojson['features'][0]['properties']
        fields = [
            key for key in ('highway', 'name', 'bridge', 'network', 'likely_run_m', 'strong') if key in properties
        ]
        folium.GeoJson(
            geojson,
            name=name,
            show=show,
            style_function=lambda _feature, style=style: style,
            tooltip=folium.GeoJsonTooltip(fields),
        ).add_to(event_map)


def add_bridges(event_map, report_dir):
    path = os.path.join(report_dir, 'bridges_flagged.geojson')
    if not os.path.exists(path):
        return
    bridges = gpd.read_file(path)
    likely_count = int((bridges['status'] == 'blocked').sum())
    group = folium.FeatureGroup(
        name=f'Bridges with damage evidence (per OSM bridge way): {likely_count} likely, '
        f'{len(bridges) - likely_count} possibly'
    )
    for _, bridge in bridges.iterrows():
        kind = dict(vehicle='motorable', foot='foot / suspension').get(str(bridge.get('net')), 'bridge')
        is_likely = bridge['status'] == 'blocked'
        folium.RegularPolygonMarker(
            marker_position(bridge.geometry),
            number_of_sides=4,
            radius=6,
            rotation=45,
            color='black',
            fill=True,
            fill_color=FOOTPRINT_COLORS['likely'] if is_likely else FOOTPRINT_COLORS['possible'],
            fill_opacity=1,
            tooltip=html.escape(
                f'{kind} bridge {bridge.get("name") or ""}: {"likely" if is_likely else "possibly"} damaged'
            ),
        ).add_to(group)
    group.add_to(event_map)


def add_flow_path(event_map, event_dir):
    """Flood path from the upstream point given with --source."""
    path_geojson = read_json(event_dir, 'flowpath/path.geojson')
    if path_geojson is None:
        return
    folium.GeoJson(
        path_geojson,
        name='Flood path from the configured upstream point (D8)',
        style_function=lambda _feature: dict(color='#2c7bb6', weight=3),
    ).add_to(event_map)
    settlements_path = os.path.join(event_dir, 'flowpath', 'settlements_along_path.geojson')
    if not os.path.exists(settlements_path):
        return
    group = folium.FeatureGroup(name='Settlements along that flood path', show=False)
    for _, place in gpd.read_file(settlements_path).iterrows():
        folium.CircleMarker(
            marker_position(place.geometry),
            radius=4,
            color='#2c7bb6',
            fill=True,
            tooltip=html.escape(
                f'{place.get("name")}: {place.get("km_downstream", place.get("km"))} km downstream '
                f'({place.get("tier")})'
            ),
        ).add_to(group)
    group.add_to(event_map)


def usual_routes(routes, breaks):
    """Usual pre-event route of each cut-off settlement and the breaks on it, keyed by settlement id."""
    by_sid = {}
    for _, route in routes.iterrows():
        geometry = route.geometry
        if geometry is None or geometry.is_empty:
            continue
        lines = [geometry] if geometry.geom_type == 'LineString' else list(geometry.geoms)
        by_sid.setdefault(
            str(route['sid']),
            dict(
                t=str(route['target']),
                km=rounded_or_none(route['route_km']),
                nb=int(route['n_breaches']),
                fb=rounded_or_none(route['first_breach_km']),
                bk=rounded_or_none(route.get('breached_km')),
                l=str(route['label']),
                c=[[[round(y, 5), round(x, 5)] for x, y in line.coords] for line in lines],
                b=[],
            ),
        )
    if breaks is not None:
        for _, route_break in breaks.iterrows():
            sid = str(route_break['sid'])
            if sid in by_sid:
                by_sid[sid]['b'].append(
                    [
                        round(route_break.geometry.y, 5),
                        round(route_break.geometry.x, 5),
                        float(route_break['km_from_settlement']),
                        bool(route_break['bridge']),
                        bool(route_break['strong']),
                        round(float(route_break.get('length_m', 0)) / 1000, 2),
                        str(route_break.get('tier', '')),
                    ]
                )
    return by_sid


def add_route_breaks(event_map, breaks):
    unique_breaks = breaks.drop_duplicates('seg')
    group = folium.FeatureGroup(name=f'Breaks on the usual routes of cut-off settlements ({len(unique_breaks)})')
    for _, route_break in unique_breaks.iterrows():
        names = sorted(set(breaks.loc[breaks['seg'] == route_break['seg'], 'settlement'].astype(str)))
        folium.Marker(
            [route_break.geometry.y, route_break.geometry.x],
            icon=folium.DivIcon(
                html="<div style='color:#d7191c;font:bold 15px/15px sans-serif;text-shadow:0 0 3px #fff'>"
                '&#10006;</div>',
                icon_size=(15, 15),
                icon_anchor=(7, 7),
            ),
            tooltip=html.escape(
                f'{"bridge" if route_break["bridge"] else "road"} break on the usual route of: '
                f'{", ".join(names[:6])}{" …" if len(names) > 6 else ""}'
            ),
        ).add_to(group)
    group.add_to(event_map)


def add_exits(event_map, exits):
    group = folium.FeatureGroup(name=f'Main roads leaving the analysed network ({len(exits)})', show=False)
    for _, road in exits.iterrows():
        folium.CircleMarker(
            marker_position(road.geometry),
            radius=5,
            color='#0072B2',
            fill=True,
            fill_opacity=0.9,
            weight=1,
            tooltip=html.escape(str(road.get('name'))),
        ).add_to(group)
    group.add_to(event_map)


def add_buildings(event_map, report_dir):
    buildings = gpd.read_file(os.path.join(report_dir, 'buildings_likely.geojson'))
    group = folium.FeatureGroup(name=f'Buildings in the likely footprint ({len(buildings)})', show=False)
    for _, building in buildings.head(5000).iterrows():
        folium.CircleMarker(marker_position(building.geometry), radius=1.5, color='#7b3294', fill=True).add_to(group)
    group.add_to(event_map)


def network_item_html(net):
    if net.get('network') == 'buffered':
        extent = f'the area plus {format(net.get("buffer_km") or 0, "g")} km'
    else:
        extent = 'the area only (the wider network could not be fetched)'
    snapshot = f', {html.escape(str(net.get("snapshot")))}' if net.get('snapshot') else ''
    border = ' Routes do not cross the national border (taken from OSM).' if net.get('border_known') else ''
    return (
        f'<li><b>Road network for the cut-off analysis</b>: {extent} (pre-event OSM{snapshot}). Destinations: '
        f'{net.get("n_exits", 0)} main-road exits, {net.get("n_hospitals", 0)} hospitals, '
        f'{net.get("n_towns", 0)} towns.{border}</li>'
    )


def data_tab_html(_event, summary, context):
    """Write acquisition dates, map definitions, observation limits and attribution."""
    net = ((summary.get('impacts') or {}).get('q3') or {}).get('net') or {}
    s1_text, s2_text = context['inputs']
    is_dl_used, dl_text = context['dl']
    access = context['access'] or {}
    footprint_class = (
        ' Buildings within the flood extent are counted separately from buildings without road access.'
        if access.get('has_footprint_class')
        else ''
    )
    unverified = (
        ' <b>Access unverified</b>: connected only through roads that could not be observed (cloud and radar shadow).'
        if access.get('unverified_buildings')
        else ''
    )
    if context['newest_date']:
        newest = (
            f'The images end on {html.escape(date_text(context["newest_date"]))}. Repairs, temporary tracks and new '
            'damage after that date do not show.'
        )
    else:
        newest = 'No post-event image was usable.'
    return (
        '<h3>Inputs</h3><ul>'
        '<li><b>Sentinel-1</b> (calibrated and terrain-corrected in this pipeline), tracks: '
        f'{html.escape(s1_text)}</li>'
        f'<li><b>Sentinel-2 L2A</b>: {html.escape(s2_text)}</li>'
        '<li><b>Copernicus DEM GLO-30</b>: terrain, HAND, flow directions, walking times</li>'
        f'<li><b>OpenStreetMap before the event</b>: {html.escape(str(summary.get("osm_snapshot")))}</li>'
        + (network_item_html(net) if net else '')
        + f'<li><b>Radar water model</b>: {"used: " if is_dl_used else "<b>not used</b>: "}'
        + f'{html.escape(dl_text)}</li></ul>'
        '<h3>Map classes</h3><ul><li><b>Likely</b>: outside large plains, flat-ground detections require agreement '
        'between two masks: radar change, optical change or the radar water model. In mountains, detections must '
        'connect to a river channel draining over 20 km². <b>Possible</b> marks weaker '
        'or single-source detections.</li>'
        f'<li>{markdown_inline_html(plain_text_en(context.get("plain_km2") or 0))}'
        f'{markdown_inline_html(inferred_text_en(context.get("footprint")))}</li>'
        f'<li>{markdown_inline_html(ROAD_BREAK_RULE_EN)} Bridges are judged by the road on each side. '
        f'{markdown_inline_html(STRONG_EVIDENCE_RULE_EN)}</li>'
        '<li><b>Cut off</b>: no road route leads to a town, a hospital or a main road leaving the analysed network; '
        f'{ROAD_ACCESS_RULE_EN}. Access now: <b>on foot</b> = a footpath reaches a road that is still connected '
        '(walking time estimated from the elevation model); <b>no mapped road or trail</b> = no mapped path avoids the '
        f'footprint. Buildings count with the nearest named settlement within 1.5 km.{footprint_class}{unverified}</li>'
        '<li><b>Repair candidates</b> are ranked by the buildings reconnected when the affected road segments reopen. '
        'The candidates require an engineering assessment.</li></ul>'
        '<h3>Limits</h3><ul>'
        f'<li>Flood polygons under {MIN_POLYGON_HA:g} ha (20 pixels of 10 m) are not drawn. The km² totals include '
        'them.</li>'
        f'<li>{newest}</li>'
        '<li>The hazard is mapped only inside the analysed area; roads outside it count as open. Buildings and roads '
        'missing from OpenStreetMap are not counted.</li></ul>'
        f'<p><small>Generated {context["generated"]} UTC.</small></p>'
        f'<h3>Attribution</h3><p><small>{html.escape(context["attribution"])}</small></p>'
    )


def write_dashboard(event_dir, event, summary, context, sitreps, layers, csvs):
    """dashboard.html: folium map with the panel, header cards and legend of dashboard_ui."""
    report_dir = os.path.join(event_dir, 'report')
    availability = context['availability']
    bbox = summary['bbox']
    event_map = folium.Map(
        location=[(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2], zoom_start=11, tiles=None, control_scale=True
    )
    add_backgrounds(event_map, event_dir)
    add_extents(event_map, summary)
    if layers['hazard'] is not None:
        add_footprint(event_map, layers['hazard'])
    add_model_only_layers(event_map, event_dir)
    if not availability['roads']:
        add_roads(event_map, report_dir)
        add_bridges(event_map, report_dir)
    add_flow_path(event_map, event_dir)
    routes = {}
    if layers['routes'] is not None:
        routes = usual_routes(layers['routes'], layers['breaks'])
        if layers['breaks'] is not None:
            add_route_breaks(event_map, layers['breaks'])
    if layers['exits'] is not None:
        add_exits(event_map, layers['exits'])
    if not availability['buildings'] and os.path.exists(os.path.join(report_dir, 'buildings_likely.geojson')):
        add_buildings(event_map, report_dir)
    layer_control = folium.LayerControl(collapsed=True)  # an open list hides much of the map
    layer_control.add_to(event_map)
    head, body, script = dashboard_ui.dashboard_html(
        map_name=event_map.get_name(),
        layer_control_name=layer_control.get_name(),
        context=context,
        sitreps=sitreps,
        data=dashboard_data(context, layers, csvs),
        flow_grid_data=dashboard_ui.flow_grid(event_dir, bbox),
        data_tab=data_tab_html(event, summary, context),
        routes=routes,
    )
    root = event_map.get_root()
    root.header.add_child(folium.Element(head))
    root.html.add_child(folium.Element(body))
    root.script.add_child(folium.Element(script))
    root.title = f'Flood impact — {context["title"]}'
    event_map.save(os.path.join(report_dir, 'dashboard.html'))


def read_impact_layer(event_dir, file_name):
    path = os.path.join(event_dir, 'impact', file_name)
    return gpd.read_file(path) if os.path.exists(path) else None


def write_building_layer(event_dir, report_dir):
    buildings = read_impact_layer(event_dir, 'buildings_affected.gpkg')
    if buildings is None:
        return
    likely = buildings[buildings['status'] == 'likely_affected'].copy()
    likely['geometry'] = likely.geometry.representative_point()
    likely[['status', 'geometry']].to_crs(4326).to_file(
        os.path.join(report_dir, 'buildings_likely.geojson'), driver='GeoJSON'
    )


def write_road_layers(event_dir, report_dir):
    """Write road-status layers and one point per assessed OSM bridge way."""
    segments = read_impact_layer(event_dir, 'road_segments_affected.gpkg')
    if segments is None:
        return
    columns = ['highway', 'name', 'bridge', 'status', 'likely_run_m', 'n_sources', 'strong', 's1s2_support', 'geometry']
    roads = segments[segments['status'].isin(['blocked', 'possibly_blocked'])][
        [column for column in columns if column in segments]
    ].copy()
    if 'likely_run_m' in roads:
        roads['likely_run_m'] = roads['likely_run_m'].round(0)
    roads['network'] = impact.network_type(roads['highway'])
    roads[roads['status'] == 'blocked'].to_crs(4326).to_file(
        os.path.join(report_dir, 'roads_blocked.geojson'), driver='GeoJSON'
    )
    roads[roads['status'] == 'possibly_blocked'].to_crs(4326).to_file(
        os.path.join(report_dir, 'roads_possibly_blocked.geojson'), driver='GeoJSON'
    )
    bridges = impact.bridge_table(segments)
    bridges = bridges[bridges['status'].isin(['blocked', 'possibly_blocked'])]
    if len(bridges):
        bridges.to_crs(4326).to_file(os.path.join(report_dir, 'bridges_flagged.geojson'), driver='GeoJSON')


def read_settlements(event_dir, report_dir):
    """Settlement points inside the area in EPSG:4326; also writes settlements.geojson."""
    settlements = read_impact_layer(event_dir, 'settlements_vehicle.gpkg')
    if settlements is None:
        return None
    settlements = settlements.copy()
    settlements['geometry'] = settlements.geometry.representative_point()
    settlements = settlements[settlements['in_aoi'].astype(bool)]
    columns = [
        'sid',
        'label',
        'name',
        'name:en',
        'name:ne',
        'place',
        'status',
        'access',
        'dist_pre_km',
        'dist_post_km',
        'walk_km',
        'walk_h',
        'target',
        'adm_muni',
        'adm_ward',
        'named',
        'cross_border',
        'geometry',
    ]
    settlements = settlements[[column for column in columns if column in settlements]].to_crs(4326)
    settlements.to_file(os.path.join(report_dir, 'settlements.geojson'), driver='GeoJSON')
    return settlements


def read_cutoff_layers(event_dir, report_dir, is_cutoff_computed):
    """Cut-off analysis layers in EPSG:4326 (None for a missing layer)."""
    names = ('units', 'runs', 'routes', 'breaks', 'health', 'exits')
    if not is_cutoff_computed:
        return dict.fromkeys(('settlements', *names))
    files = dict(
        units='settlement_units.gpkg',
        runs='break_runs.gpkg',
        routes='access_routes.gpkg',
        breaks='access_breaches.gpkg',
        health='health_status.gpkg',
        exits='net_exits.gpkg',
    )
    layers = dict(settlements=read_settlements(event_dir, report_dir))
    for name in names:
        layer = read_impact_layer(event_dir, files[name])
        layers[name] = None if layer is None else layer.to_crs(4326)
    return layers


def build(event_dir, event, push):
    """Writes the report folder of one event."""
    report_dir = os.path.join(event_dir, 'report')
    os.makedirs(report_dir, exist_ok=True)
    summary = read_json(event_dir, 'summary.json')
    availability = product_availability(event_dir, summary)
    hazard = None
    hazard_tif = os.path.join(event_dir, 'hazard', 'hazard.tif')
    if os.path.exists(hazard_tif):
        hazard = hazard_polygons(hazard_tif, os.path.join(event_dir, 'hazard', 'sources.tif'))
        hazard.to_crs(4326).to_file(os.path.join(report_dir, 'hazard.geojson'), driver='GeoJSON')
    if not availability['buildings']:
        write_building_layer(event_dir, report_dir)
    if not availability['roads']:
        write_road_layers(event_dir, report_dir)
    layers = read_cutoff_layers(event_dir, report_dir, not availability['cutoff'])
    layers['hazard'] = hazard
    context = report_context(event_dir, event, summary, availability)
    sitreps = write_sitreps(report_dir, context)
    csvs = csv_exports(context, layers)
    for file_name, text in csvs.items():
        write_text(os.path.join(report_dir, file_name), text, encoding='utf-8-sig', newline='')
    write_dashboard(event_dir, event, summary, context, sitreps, layers, csvs)
    if push:
        for file_name in os.listdir(report_dir):
            store.put(os.path.join(report_dir, file_name), f'events/{event}/report/{file_name}')
    print(sitreps[0], flush=True)
