"""Hazard impacts on buildings, roads and bridges, and road access of settlements, from pre-event OSM."""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, substring

import fusion

# hazard raster classes (0 = no evidence)
POSSIBLE = 1
LIKELY = 2

MOTOR_HIGHWAYS = frozenset(
    {
        'motorway',
        'trunk',
        'primary',
        'secondary',
        'tertiary',
        'unclassified',
        'residential',
        'service',
        'living_street',
        'motorway_link',
        'trunk_link',
        'primary_link',
        'secondary_link',
        'tertiary_link',
        'road',
        'track',
    }
)
FOOT_HIGHWAYS = MOTOR_HIGHWAYS | {'path', 'footway', 'steps', 'pedestrian', 'bridleway', 'cycleway'}
FOOT_ONLY_HIGHWAYS = FOOT_HIGHWAYS - MOTOR_HIGHWAYS
SPEED_KMH = {
    'motorway': 60,
    'trunk': 40,
    'primary': 35,
    'secondary': 30,
    'tertiary': 25,
    'unclassified': 20,
    'residential': 15,
    'service': 10,
    'track': 10,
    'road': 15,
    'path': 3,
    'footway': 3,
    'steps': 2,
    'pedestrian': 4,
    'bridleway': 3,
    'cycleway': 5,
}
DEFAULT_SPEED_KMH = 10
OFF_NETWORK_WALK_KMH = 4.0
ROAD_CLASS = dict(
    motorway='major',
    trunk='major',
    primary='major',
    motorway_link='major',
    trunk_link='major',
    primary_link='major',
    secondary='secondary',
    tertiary='secondary',
    secondary_link='secondary',
    tertiary_link='secondary',
    unclassified='local',
    residential='local',
    living_street='local',
    service='local',
    road='local',
    track='track',
)
ROAD_CLASS_ORDER = ('major', 'secondary', 'local', 'track')
# where these roads leave the fetched network, a settlement reaches the wider road network and the towns beyond it
EXIT_HIGHWAYS = frozenset(
    {
        'motorway',
        'trunk',
        'primary',
        'secondary',
        'motorway_link',
        'trunk_link',
        'primary_link',
        'secondary_link',
    }
)
EXIT_FALLBACK_HIGHWAYS = EXIT_HIGHWAYS | {'tertiary', 'tertiary_link'}
EXIT_MARGIN_M = 500.0
SETTLEMENT_PLACES = (
    'city',
    'town',
    'village',
    'hamlet',
    'isolated_dwelling',
    'suburb',
    'neighbourhood',
)
TOWN_PLACES = ('city', 'town')

ROAD_SAMPLE_STEP_M = 5.0
STATUS_SEVERITY = {'open': 0, 'unobserved': 1, 'possibly_blocked': 2, 'blocked': 3}
STATUS_OF_SEVERITY = {severity: status for status, severity in STATUS_SEVERITY.items()}
NOT_BRIDGE_TAGS = ('', 'no', 'none', 'nan', '<na>')
# road segments outside the hazard grid: no hazard known, so unobserved and open for routing
OUTSIDE_GRID_COLUMNS = dict(
    frac_likely=0.0,
    frac_possible=0.0,
    frac_observed=0.0,
    frac_in=0.0,
    n_sources=0,
    strong=False,
    s1s2_support=False,
    likely_run_m=0.0,
    status='unobserved',
)

# post-event scenarios by the breaks they close: strong evidence; all likely; likely and bridges flagged under the span
SCENARIOS = ('strong', 'likely', 'bridge')
SCENARIOS_BY_EVIDENCE = ('bridge', 'likely', 'strong')
TIER_OF_SCENARIO = {'strong': 'cut_off', 'likely': 'possibly_cut_off', 'bridge': 'bridge_uncertain'}
NO_ROAD_TIERS = ('cut_off', 'possibly_cut_off', 'bridge_uncertain')
CUT_OFF_TIERS = ('cut_off', 'possibly_cut_off')
LISTED_TIERS = (*NO_ROAD_TIERS, 'long_detour')
PLACE_ACCESS_COLUMNS = (
    'tier',
    'access',
    'dist_pre_km',
    'dist_post_km',
    'detour_km',
    'detour_h',
    'walk_h',
    'walk_km',
    'trail_cut',
    'exit_pre',
    'exit_post',
    'svc_post',
    'lost_exit_only',
    'unverified',
)
PRIORITY_MIN_BUILDINGS = 50
ROUTE_SIMPLIFY_M = 15.0
NAME_SEARCH_M = 5000.0

# access rules, recorded in summary.json (net.q3_options) and read by the report
ACCESS_RULES = dict(
    fp_class=True,
    prio_min_bld=PRIORITY_MIN_BUILDINGS,
    roadhead_km=0.0,
    roadhead_h=0.0,
    dangle_m=0.0,
    road_rule='frac',
    strong_rule='v2',
    unverified=True,
    unit_rule='point_or_majority',
)


def _values_at(raster, transform, xs, ys, fill=0):
    """Raster values at map coordinates; `fill` outside the raster."""
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    inside = (rows >= 0) & (rows < raster.shape[0]) & (cols >= 0) & (cols < raster.shape[1])
    values = np.full(len(rows), fill, dtype=raster.dtype)
    values[inside] = raster[rows[inside], cols[inside]]
    return values


def _point_xy(gdf):
    points = gdf.geometry.representative_point()
    return np.c_[points.x.values, points.y.values] if len(gdf) else np.zeros((0, 2))


def _text_or_none(value):
    return None if value is None or str(value) in ('None', 'nan', '') else str(value)


def _value_counts(series):
    return {str(k): int(v) for k, v in series.value_counts().items()}


def assess_buildings(bld, hazard, observable, transform, buffer_px=1):
    """Hazard status of each building at its representative point, with the hazard classes grown by `buffer_px`."""
    # a point per building: buffered footprints would overlap in dense villages
    buildings = bld.copy()
    points = buildings.geometry.representative_point()
    xs, ys = points.x.values, points.y.values
    likely_near = ndimage.binary_dilation(hazard == LIKELY, iterations=buffer_px)
    possible_near = ndimage.binary_dilation(hazard >= POSSIBLE, iterations=buffer_px + 1)
    buildings['haz_point'] = _values_at(hazard, transform, xs, ys)
    buildings['likely_near'] = _values_at(likely_near.astype('uint8'), transform, xs, ys).astype(bool)
    buildings['possible_near'] = _values_at(possible_near.astype('uint8'), transform, xs, ys).astype(bool)
    buildings['observed'] = _values_at(observable.astype('uint8'), transform, xs, ys).astype(bool)
    status = np.full(len(buildings), 'not_affected', dtype=object)
    status[buildings['possible_near'].values] = 'possibly_affected'
    status[buildings['likely_near'].values] = 'likely_affected'
    status[~buildings['observed'].values & (status == 'not_affected')] = 'unobserved'
    buildings['status'] = status
    return buildings


def _vertex_key(xy):
    return (round(xy[0], 2), round(xy[1], 2))


def explode_segments(roads, max_length_m=50.0):
    """Road ways split at shared vertices, then into equal pieces of at most `max_length_m`."""
    vertex_count = defaultdict(int)
    lines_by_road = []
    for road_idx, road in roads.iterrows():
        geometry = road.geometry
        lines = list(geometry.geoms) if isinstance(geometry, MultiLineString) else [geometry]
        lines_by_road.append((road_idx, road, lines))
        for line in lines:
            for xy in line.coords:
                vertex_count[_vertex_key(xy)] += 1
    rows = []
    for road_idx, road, lines in lines_by_road:
        for line in lines:
            coords = list(line.coords)
            inner_cuts = [j for j in range(1, len(coords) - 1) if vertex_count[_vertex_key(coords[j])] >= 2]
            for first, last in pairwise([0, *inner_cuts, len(coords) - 1]):
                piece = LineString(coords[first : last + 1])
                length = piece.length
                if length <= 0:
                    continue
                n_pieces = max(1, math.ceil(length / max_length_m))
                for j in range(n_pieces):
                    part = (
                        substring(piece, length * j / n_pieces, length * (j + 1) / n_pieces) if n_pieces > 1 else piece
                    )
                    rows.append(
                        dict(
                            road_idx=road_idx,
                            highway=road.get('highway'),
                            name=road.get('name'),
                            bridge=road.get('bridge'),
                            osm_id=road.get('osm_id'),
                            ref=road.get('ref'),
                            geometry=part,
                        )
                    )
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=roads.crs)


def _max_per_segment(segment_of_point, values, n_segments):
    out = np.zeros(n_segments, 'int64')
    np.maximum.at(out, segment_of_point, values)
    return out


def assess_roads(
    segments, hazard, observable, transform, sources=None, strict_sources=None, run_m=50.0, dl_only_possible=False
):
    """Hazard, evidence (`sources`, `strict_source_bits`) and status of each segment, sampled every 5 m along it."""
    geoms = segments.geometry.values
    length_m = shapely.length(geoms)
    n_points = np.maximum(2, (length_m / ROAD_SAMPLE_STEP_M).astype(int) + 1)
    segment_of_point = np.repeat(np.arange(len(geoms)), n_points)
    first_point = np.r_[0, np.cumsum(n_points)[:-1]]
    point_rank = np.arange(segment_of_point.size) - first_point[segment_of_point]
    points = shapely.line_interpolate_point(
        geoms[segment_of_point], point_rank / (n_points[segment_of_point] - 1), normalized=True
    )
    xs, ys = shapely.get_x(points), shapely.get_y(points)
    hazard_at = _values_at(hazard, transform, xs, ys)
    observed_at = _values_at(observable.astype('uint8'), transform, xs, ys)
    n = len(geoms)
    likely = (hazard_at == LIKELY).astype('int64')
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows, cols = np.asarray(rows), np.asarray(cols)
    in_grid = (rows >= 0) & (rows < hazard.shape[0]) & (cols >= 0) & (cols < hazard.shape[1])
    # longest run of consecutive likely points: a cumulative count that restarts at each segment start and gap
    restart = (likely == 0) | (point_rank == 0)
    cumulative = np.cumsum(likely)
    run_length = cumulative - np.maximum.accumulate(np.where(restart, cumulative - likely, 0))
    likely_run_m = _max_per_segment(segment_of_point, run_length, n) * length_m / np.maximum(n_points - 1, 1)
    frac_likely = np.bincount(segment_of_point, weights=likely, minlength=n) / n_points
    frac_possible = np.bincount(segment_of_point, weights=(hazard_at >= POSSIBLE), minlength=n) / n_points
    frac_observed = np.bincount(segment_of_point, weights=observed_at, minlength=n) / n_points
    out = segments.copy()
    out['frac_likely'] = frac_likely
    out['frac_possible'] = frac_possible
    out['frac_observed'] = frac_observed
    out['frac_in'] = np.bincount(segment_of_point, weights=in_grid, minlength=n) / n_points
    if sources is None:
        out['n_sources'] = 1
        out['strong'] = True
        out['s1s2_support'] = True
    else:
        bits = _values_at(sources.astype('uint8'), transform, xs, ys).astype('int64')
        strict_bits = (
            bits
            if strict_sources is None
            else _values_at(strict_sources.astype('uint8'), transform, xs, ys).astype('int64')
        )
        source_count = (
            ((strict_bits & 1) > 0).astype(int)
            + ((strict_bits & 2) > 0).astype(int)
            + ((strict_bits & 4) > 0).astype(int)
        )
        out['n_sources'] = _max_per_segment(segment_of_point, source_count, n)
        # Radar change and radar water model share images and do not give independent evidence.
        strong = (
            ((strict_bits & 2) > 0)
            | (((strict_bits & 1) > 0) & ((strict_bits & 8) > 0))
            | (((strict_bits & 1) > 0) & ((bits & 2) > 0))
        )
        out['strong'] = _max_per_segment(segment_of_point, strong.astype('int64'), n) > 0
        s1s2_likely = (((bits & 3) > 0) & (hazard_at == LIKELY)).astype('int64')
        out['s1s2_support'] = _max_per_segment(segment_of_point, s1s2_likely, n) > 0
    out['likely_run_m'] = likely_run_m
    status = np.full(n, 'open', dtype=object)
    status[frac_possible > 0.2] = 'possibly_blocked'
    status[(likely_run_m >= run_m) | (frac_likely >= 0.5)] = 'blocked'
    if dl_only_possible:
        status[(status == 'blocked') & ~out['s1s2_support'].values.astype(bool)] = 'possibly_blocked'
    _set_bridge_status(status, out, frac_likely, hazard, transform)
    status[(frac_observed < 0.5) & (status == 'open')] = 'unobserved'
    out['status'] = status
    return out


def _set_bridge_status(status, segments, frac_likely, hazard, transform):
    """Status of bridge ways from the hazard 15 m and 30 m beyond either end, since the span crosses the channel."""
    bridge_mask = (segments['bridge'].notna() & (segments['bridge'].astype(str) != 'no')).values
    if not bridge_mask.any():
        return
    geoms = segments.geometry.values
    likely_near = ndimage.binary_dilation(hazard == LIKELY, iterations=1).astype('uint8')
    possible_near = ndimage.binary_dilation(hazard >= POSSIBLE, iterations=1).astype('uint8')
    bridge_index = np.where(bridge_mask)[0]
    road_index = segments['road_idx'].values
    approaches, ways = [], []
    # the segments of one OSM way share road_idx and follow the way's direction
    for road in np.unique(road_index[bridge_index]):
        way = bridge_index[road_index[bridge_index] == road]
        start = np.array(geoms[way[0]].coords[0])
        end = np.array(geoms[way[-1]].coords[-1])
        direction = end - start
        direction = direction / max(np.hypot(*direction), 1e-6)
        approaches.append([start - direction * 15, start - direction * 30, end + direction * 15, end + direction * 30])
        ways.append(way)
    approach_xy = np.array(approaches).reshape(-1, 2)
    likely_at_approach = _values_at(likely_near, transform, approach_xy[:, 0], approach_xy[:, 1]).reshape(-1, 4).any(1)
    possible_at_approach = (
        _values_at(possible_near, transform, approach_xy[:, 0], approach_xy[:, 1]).reshape(-1, 4).any(1)
    )
    for i, way in enumerate(ways):
        if likely_at_approach[i]:
            status[way] = 'blocked'
        elif possible_at_approach[i] or (frac_likely[way] > 0).any():
            status[way] = 'possibly_blocked'
        else:
            status[way] = 'open'


def assess_roads_grid(
    segments, hazard, observable, transform, *, sources=None, strict_sources=None, run_m=50.0, dl_only_possible=False
):
    """`assess_roads` for the segments touching the hazard grid; segments beyond it are unobserved, without sampling."""
    height, width = hazard.shape
    x_min, x_max = sorted((transform.c, transform.c + width * transform.a))
    y_min, y_max = sorted((transform.f, transform.f + height * transform.e))
    bounds = segments.geometry.bounds
    touches = (
        (bounds['maxx'] >= x_min) & (bounds['minx'] <= x_max) & (bounds['maxy'] >= y_min) & (bounds['miny'] <= y_max)
    ).values
    if touches.all():
        return assess_roads(segments, hazard, observable, transform, sources, strict_sources, run_m, dl_only_possible)
    outside = segments[~touches].copy()
    for column, value in OUTSIDE_GRID_COLUMNS.items():
        outside[column] = value
    parts = [outside]
    if touches.any():
        inside = assess_roads(
            segments[touches], hazard, observable, transform, sources, strict_sources, run_m, dl_only_possible
        )
        parts = [inside, outside]
    combined = pd.concat(parts)
    return gpd.GeoDataFrame(combined.loc[segments.index], geometry='geometry', crs=segments.crs)


def _bridge_mask(segments):
    tag = segments['bridge']
    return (tag.notna() & ~tag.astype(str).str.strip().str.lower().isin(NOT_BRIDGE_TAGS)).values


def network_type(highway):
    """'vehicle' (motorable), 'foot' (paths, trails, steps) or 'other' per highway value."""
    values = np.asarray(pd.Series(highway).astype(str).values)
    return np.where(
        np.isin(values, list(MOTOR_HIGHWAYS)),
        'vehicle',
        np.where(np.isin(values, list(FOOT_ONLY_HIGHWAYS)), 'foot', 'other'),
    )


def _km_by_status(segments):
    return {str(k): round(float(v), 2) for k, v in segments.groupby('status')['km'].sum().items()}


def roads_summary(segments):
    """Affected of total for roads (km; motorable vs footpaths, by road class) and bridges (per OSM bridge way)."""
    network = network_type(segments['highway'])
    vehicle = segments[network == 'vehicle']
    foot = segments[network == 'foot']
    road_class = vehicle['highway'].astype(str).map(ROAD_CLASS).fillna('local')
    bridges = bridge_table(segments)
    return dict(
        roads_km=_km_by_status(segments),
        roads_km_vehicle=_km_by_status(vehicle),
        roads_km_vehicle_total=round(float(vehicle['km'].sum()), 1),
        paths_km=_km_by_status(foot),
        paths_km_total=round(float(foot['km'].sum()), 1),
        roads_km_by_class={
            c: dict(
                _km_by_status(vehicle[road_class == c]), total=round(float(vehicle.loc[road_class == c, 'km'].sum()), 1)
            )
            for c in ROAD_CLASS_ORDER
            if bool((road_class == c).any())
        },
        bridges_segments=_value_counts(segments[_bridge_mask(segments)]['status']),
        bridges=_value_counts(bridges['status']),
        bridges_vehicle=_value_counts(bridges[bridges['net'] == 'vehicle']['status']),
        bridges_foot=_value_counts(bridges[bridges['net'] == 'foot']['status']),
        bridges_total=len(bridges),
        bridges_vehicle_total=int((bridges['net'] == 'vehicle').sum()),
        bridges_foot_total=int((bridges['net'] == 'foot').sum()),
    )


def bridge_table(segments):
    """One row per OSM bridge way: worst segment status, network type, name, and the middle of the way."""
    bridges = segments[_bridge_mask(segments)]
    if not len(bridges):
        columns = ['road_idx', 'status', 'net', 'highway', 'name']
        return gpd.GeoDataFrame(
            {k: [] for k in columns}, geometry=gpd.GeoSeries([], crs=segments.crs), crs=segments.crs
        )
    rows = []
    for road_idx, way in bridges.groupby('road_idx', sort=False):
        severity = max(STATUS_SEVERITY.get(str(s), 0) for s in way['status'])
        rows.append(
            dict(
                road_idx=road_idx,
                status=STATUS_OF_SEVERITY[severity],
                net=str(network_type(way['highway'].iloc[:1])[0]),
                highway=str(way['highway'].iloc[0]),
                name=_text_or_none(way['name'].iloc[0]),
                geometry=way.geometry.iloc[len(way) // 2].interpolate(0.5, normalized=True),
            )
        )
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=segments.crs)


def _persistent_s2(s2p, s2e, s2cal_p, s2cal_e, pre_ndvi, s2e_wat):
    """Where the fusion keeps S2 evidence: change persisting in both clear images, and new water in the first one."""
    post_change = s2p == 1
    early_change = (s2e == 1) if s2e is not None else np.zeros(post_change.shape, bool)
    if s2cal_p is not None and pre_ndvi is not None:
        bare = np.nan_to_num(pre_ndvi, nan=1.0) < 0.3
        calibrated_early = (s2cal_e == 1) if s2cal_e is not None else np.zeros(post_change.shape, bool)
        post_change = post_change | (bare & (s2cal_p == 1))
        early_change = early_change | (bare & calibrated_early)
    if s2e is None:
        keep = post_change
    else:
        post_seen, early_seen = s2p != 255, s2e != 255
        keep = np.where(
            early_seen & post_seen, post_change & early_change, np.where(early_seen, early_change, post_change)
        )
    if s2e_wat is not None:
        keep = keep | (s2e_wat == 1)
    return keep


def strict_source_bits(  # noqa: PLR0913
    src,
    params,
    s2p=None,
    s2e=None,
    s2cal_p=None,
    s2cal_e=None,
    pre_ndvi=None,
    s2e_wat=None,
    dl=None,
    slope=None,
    s1n=None,
):
    """Evidence bits of sources.tif (1 = S1, 2 = S2, 4 = DL) at the fusion thresholds, plus 8 = S1 on >= 2 tracks."""
    bits = np.asarray(src).astype('uint8').copy()
    if s2p is not None:
        keep = _persistent_s2(s2p, s2e, s2cal_p, s2cal_e, pre_ndvi, s2e_wat)
        bits = np.where(keep, bits, bits & np.uint8(253)).astype('uint8')
    if dl is not None:
        flat_thr = float(params['dl_thr'])
        mountain_thr = float(params['dl_thr_mtn'])
        probability = np.nan_to_num(dl, nan=0.0)
        if slope is None:
            dl_kept = probability > mountain_thr
        else:
            flat = ndimage.binary_dilation(fusion.flat_mask(slope, float(params['flat_deg'])), iterations=3)
            dl_kept = np.where(flat, probability > flat_thr, probability > mountain_thr)
        bits = np.where(dl_kept, bits, bits & np.uint8(251)).astype('uint8')
    if s1n is not None:
        two_tracks = (np.nan_to_num(np.asarray(s1n, 'float32')) >= 2) & ((bits & 1) > 0)
        bits = (bits | np.where(two_tracks, 8, 0)).astype('uint8')
    return bits


def _tobler_kmh(grade):
    return 6.0 * np.exp(-3.5 * np.abs(grade + 0.05))


def add_walk_time(segments, dem, transform):
    """Add `walk_h`: Tobler's hiking function on the DEM, mean of both walking directions."""
    geoms = segments.geometry.values
    length_m = np.asarray(shapely.length(geoms), float)
    start = shapely.get_coordinates(shapely.get_point(geoms, 0))
    end = shapely.get_coordinates(shapely.get_point(geoms, -1))
    dem_m = dem.astype('float32')
    z_start = _values_at(dem_m, transform, start[:, 0], start[:, 1], fill=np.nan)
    z_end = _values_at(dem_m, transform, end[:, 0], end[:, 1], fill=np.nan)
    # |z| >= 9000 m is DEM nodata
    z_start = np.where(np.abs(z_start) < 9000, z_start, np.nan)
    z_end = np.where(np.abs(z_end) < 9000, z_end, np.nan)
    # a missing elevation counts as flat ground
    grade = np.clip(np.nan_to_num((z_end - z_start) / np.maximum(length_m, 1.0), nan=0.0), -1.0, 1.0)
    out = segments.copy()
    out['walk_h'] = 0.5 * (length_m / 1000.0 / _tobler_kmh(grade) + length_m / 1000.0 / _tobler_kmh(-grade))
    return out


@dataclass
class RoadNetwork:
    """Road graph with one edge per assessed segment; nodes are segment ends rounded to 0.1 m."""

    nodes: np.ndarray
    start: np.ndarray
    end: np.ndarray
    length_m: np.ndarray
    drive_h: np.ndarray
    walk_h: np.ndarray
    is_vehicle: np.ndarray
    is_foot: np.ndarray
    closed: dict[str, np.ndarray]
    is_bridge: np.ndarray
    highway: np.ndarray
    crosses_border: np.ndarray
    is_unverified: np.ndarray
    node_index: dict[str, tuple[np.ndarray, cKDTree | None]]
    exits: np.ndarray
    exit_classes: str | None
    surviving_index: dict = field(default_factory=dict)

    @property
    def n_nodes(self):
        return len(self.nodes)

    def mode_edges(self, mode):
        return self.is_vehicle if mode == 'vehicle' else self.is_foot


@dataclass
class Reach:
    """Reachability of the network nodes in one scenario."""

    component: np.ndarray
    targets: np.ndarray
    exits: np.ndarray
    distance_m: np.ndarray | None = None
    drive_h: np.ndarray | None = None
    predecessor: np.ndarray | None = None
    origin: np.ndarray | None = None
    walk_h: np.ndarray | None = None
    walk_km: np.ndarray | None = None
    roadhead: np.ndarray | None = None

    def reaches_target(self, node, own_targets):
        """Whether the component of each node holds a target or an exit other than the point itself."""
        component = self.component[node]
        return (self.targets[component] + self.exits[component] - own_targets) > 0


@dataclass
class Access:
    """Vehicle reach before the event and per post-event scenario, and walking access from the roadheads."""

    pre: Reach
    post: dict[str, Reach]
    unverified: Reach
    walk_pre_h: np.ndarray
    edge_of_pair: dict[tuple[int, int], int]
    n_sources_pre: int


@dataclass(frozen=True)
class AccessRules:
    snap_vehicle_m: float
    snap_foot_m: float
    detour_factor: float


def _exits(nodes, start, end, highway, is_vehicle, net_poly, main_country):
    """Main-road nodes within EXIT_MARGIN_M of the network extent's edge or beyond it, and the road classes used."""
    inner = net_poly.buffer(-EXIT_MARGIN_M)
    for classes, label in (
        (EXIT_HIGHWAYS, 'trunk/primary/secondary'),
        (EXIT_FALLBACK_HIGHWAYS, 'tertiary (no main road leaves the area)'),
    ):
        edges = is_vehicle & np.isin(highway, list(classes))
        candidates = np.unique(np.r_[start[edges], end[edges]]).astype(int)
        if len(candidates) and not inner.is_empty:
            candidates = candidates[~shapely.contains_xy(inner, nodes[candidates, 0], nodes[candidates, 1])]
        if len(candidates) and main_country is not None:
            candidates = candidates[shapely.contains_xy(main_country, nodes[candidates, 0], nodes[candidates, 1])]
        if len(candidates):
            return candidates, label
    return np.zeros(0, int), None


def build_network(segments, net_poly, border, main_country):
    """Road network of assessed segments with walk_h; segments crossing `border` are closed."""
    geoms = segments.geometry.values
    n = len(geoms)
    segment_ends = np.vstack(
        [shapely.get_coordinates(shapely.get_point(geoms, 0)), shapely.get_coordinates(shapely.get_point(geoms, -1))]
    )
    nodes, node_of_end = np.unique(np.round(segment_ends, 1), axis=0, return_inverse=True)
    node_of_end = np.asarray(node_of_end).reshape(-1)
    start, end = node_of_end[:n], node_of_end[n:]
    highway = segments['highway'].astype(str).values
    length_m = np.maximum(np.asarray(shapely.length(geoms), float), 0.01)
    speed_kmh = np.array([SPEED_KMH.get(h, DEFAULT_SPEED_KMH) for h in highway], float)
    status = segments['status'].astype(str).values
    crosses_border = np.zeros(n, bool)
    if border is not None:
        hits = shapely.STRtree(geoms).query(list(getattr(border, 'geoms', [border])), predicate='intersects')
        crosses_border[np.unique(hits[1])] = True
    is_vehicle = np.isin(highway, list(MOTOR_HIGHWAYS)) & ~crosses_border
    is_foot = np.isin(highway, list(FOOT_HIGHWAYS)) & ~crosses_border
    is_bridge = _bridge_mask(segments)
    blocked = status == 'blocked'
    closed = dict(
        strong=blocked & segments['strong'].astype(bool).values,
        likely=blocked,
        bridge=blocked | (is_bridge & (status == 'possibly_blocked')),
    )
    vehicle_nodes = np.unique(np.r_[start[is_vehicle], end[is_vehicle]]).astype(int)
    foot_nodes = np.unique(np.r_[start[is_foot], end[is_foot]]).astype(int)
    exits, exit_classes = (
        _exits(nodes, start, end, highway, is_vehicle, net_poly, main_country)
        if len(vehicle_nodes)
        else (np.zeros(0, int), None)
    )
    return RoadNetwork(
        nodes=nodes,
        start=start,
        end=end,
        length_m=length_m,
        drive_h=length_m / 1000.0 / speed_kmh,
        walk_h=segments['walk_h'].values.astype(float),
        is_vehicle=is_vehicle,
        is_foot=is_foot,
        closed=closed,
        is_bridge=is_bridge,
        highway=highway,
        crosses_border=crosses_border,
        # unobserved inside the analysed grid: radar shadow or layover without a clear optical view
        is_unverified=(status == 'unobserved') & (segments['frac_in'].values.astype(float) >= 0.5),
        node_index=dict(
            vehicle=(vehicle_nodes, cKDTree(nodes[vehicle_nodes]) if len(vehicle_nodes) else None),
            foot=(foot_nodes, cKDTree(nodes[foot_nodes]) if len(foot_nodes) else None),
        ),
        exits=exits,
        exit_classes=exit_classes,
    )


def _adjacency(network, edge_mask, weight):
    """Upper-triangle adjacency of the masked edges, lightest edge per node pair -> (matrix, edges, lows, highs)."""
    edges = np.where(edge_mask)[0]
    low = np.minimum(network.start[edges], network.end[edges])
    high = np.maximum(network.start[edges], network.end[edges])
    edge_weight = np.maximum(np.asarray(weight, float)[edges], 1e-9)
    not_loop = low != high
    edges, low, high, edge_weight = edges[not_loop], low[not_loop], high[not_loop], edge_weight[not_loop]
    order = np.lexsort((edge_weight, high, low))
    edges, low, high, edge_weight = edges[order], low[order], high[order], edge_weight[order]
    first = np.r_[True, (low[1:] != low[:-1]) | (high[1:] != high[:-1])] if len(low) else np.zeros(0, bool)
    matrix = sparse.csr_matrix(
        (edge_weight[first], (low[first], high[first])), shape=(network.n_nodes, network.n_nodes)
    )
    return matrix, edges[first], low[first], high[first]


def _shortest_from(matrix, sources):
    """Multi-source Dijkstra -> (distance, predecessor, nearest source); unreachable nodes: inf, -9999, -9999."""
    n = matrix.shape[0]
    sources = np.unique(np.asarray(sources, int))
    if not len(sources):
        return np.full(n, np.inf), np.full(n, -9999), np.full(n, -9999)
    distance, predecessor, origin = csgraph.dijkstra(
        matrix, directed=False, indices=sources, min_only=True, return_predecessors=True
    )
    return np.asarray(distance), np.asarray(predecessor), np.asarray(origin)


def _component_reach(matrix, targets_per_node, exit_per_node):
    n_components, component = csgraph.connected_components(matrix, directed=False)
    targets = np.bincount(component, weights=targets_per_node, minlength=n_components)
    exits = np.bincount(component, weights=exit_per_node, minlength=len(targets))
    return Reach(component=component, targets=targets, exits=exits)


def solve_access(network, target_nodes, target_usable_post):
    """Reachability of every node from the targets and exits before the event and in each post-event scenario."""
    n = network.n_nodes
    targets = np.asarray(target_nodes, int)
    usable_post = np.asarray(target_usable_post, bool) & (targets >= 0)
    targets_pre = targets[targets >= 0]
    exits = np.asarray(network.exits, int)
    exit_per_node = np.zeros(n)
    exit_per_node[exits] = 1.0
    targets_per_node_post = np.bincount(targets[usable_post], minlength=n).astype(float)
    sources_pre = np.r_[targets_pre, exits]
    sources_post = np.r_[targets[usable_post], exits]
    pre_matrix, edge_index, pair_low, pair_high = _adjacency(network, network.is_vehicle, network.length_m)
    pre = _component_reach(pre_matrix, np.bincount(targets_pre, minlength=n).astype(float), exit_per_node)
    pre.distance_m, pre.predecessor, pre.origin = _shortest_from(pre_matrix, sources_pre)
    pre.drive_h = _shortest_from(_adjacency(network, network.is_vehicle, network.drive_h)[0], sources_pre)[0]
    post = {}
    for scenario in SCENARIOS:
        open_vehicle = network.is_vehicle & ~network.closed[scenario]
        post_matrix = _adjacency(network, open_vehicle, network.length_m)[0]
        reach = _component_reach(post_matrix, targets_per_node_post, exit_per_node)
        reach.distance_m = _shortest_from(post_matrix, sources_post)[0]
        if scenario == 'likely':
            reach.drive_h = _shortest_from(_adjacency(network, open_vehicle, network.drive_h)[0], sources_post)[0]
        # walking over the open foot network from every node still reachable by vehicle ('roadheads')
        open_foot = network.is_foot & ~network.closed[scenario]
        roadheads = np.where(np.isfinite(reach.distance_m))[0]
        reach.walk_h, _, reach.roadhead = _shortest_from(_adjacency(network, open_foot, network.walk_h)[0], roadheads)
        reach.walk_km = _shortest_from(_adjacency(network, open_foot, network.length_m)[0], roadheads)[0] / 1000.0
        post[scenario] = reach
    walk_pre_h = _shortest_from(
        _adjacency(network, network.is_foot, network.walk_h)[0], np.where(np.isfinite(pre.distance_m))[0]
    )[0]
    open_verified = network.is_vehicle & ~(network.closed['likely'] | network.is_unverified)
    unverified = _component_reach(
        _adjacency(network, open_verified, network.length_m)[0], targets_per_node_post, exit_per_node
    )
    return Access(
        pre=pre,
        post=post,
        unverified=unverified,
        walk_pre_h=walk_pre_h,
        edge_of_pair=dict(
            zip(zip(pair_low.tolist(), pair_high.tolist(), strict=True), edge_index.tolist(), strict=True)
        ),
        n_sources_pre=len(np.unique(sources_pre)),
    )


def _nearest_of(nodes, tree, xy):
    xy = np.asarray(xy, float).reshape(-1, 2)
    if tree is None or not len(xy):
        return np.full(len(xy), -1), np.full(len(xy), np.inf)
    distance, nearest = tree.query(xy)
    return nodes[nearest], distance


def _snap(network, xy, mode):
    """Nearest node of the 'vehicle' or 'foot' network -> (node, distance m); (-1, inf) when that network is empty."""
    nodes, tree = network.node_index[mode]
    return _nearest_of(nodes, tree, xy)


def _snap_surviving(network, xy, scenario, mode):
    """Nearest node with an open `mode` edge in a scenario, so that points inside a breach attach to intact road."""
    key = (scenario, mode)
    if key not in network.surviving_index:
        edges = network.mode_edges(mode) & ~network.closed[scenario]
        nodes = np.unique(np.r_[network.start[edges], network.end[edges]]).astype(int)
        network.surviving_index[key] = (nodes, cKDTree(network.nodes[nodes]) if len(nodes) else None)
    nodes, tree = network.surviving_index[key]
    return _nearest_of(nodes, tree, xy)


def _attach(network, xy, scenario, mode, max_m):
    """Post-event attachment -> (node or 0, attached within max_m, distance m)."""
    node, distance = _snap_surviving(network, xy, scenario, mode)
    return np.where(node >= 0, node, 0), (node >= 0) & (distance <= max_m), distance


def _walk_access(access, tier, detour, no_road_pre, foot_post):
    """Access class of each point, and the walk to the nearest roadhead for points cut off from the roads."""
    n = len(tier)
    access_class = np.full(n, 'road', dtype=object)
    access_class[detour] = 'detour'
    walk_h, walk_km, roadhead = np.full(n, np.nan), np.full(n, np.nan), np.full(n, -1)
    for scenario, tier_name in TIER_OF_SCENARIO.items():
        in_tier = tier == tier_name
        if not in_tier.any():
            continue
        node, attached, distance_m = foot_post[scenario]
        reach = access.post[scenario]
        path_h = reach.walk_h[node[in_tier]]
        walkable = attached[in_tier] & np.isfinite(path_h)
        access_class[in_tier] = np.where(walkable, 'walk_in', 'air_only')
        walk_h[in_tier] = np.where(walkable, path_h + distance_m[in_tier] / 1000.0 / OFF_NETWORK_WALK_KMH, np.nan)
        walk_km[in_tier] = np.where(walkable, reach.walk_km[node[in_tier]] + distance_m[in_tier] / 1000.0, np.nan)
        roadhead[in_tier] = np.where(walkable, reach.roadhead[node[in_tier]], -1)
    access_class[no_road_pre] = 'no_road_pre'
    return access_class, walk_h, walk_km, roadhead


def classify_access(network, access, xy, rules, own_pre=None, own_post=None):
    """Classify access; own_pre and own_post count destinations represented by the point itself."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    n = len(xy)
    own_pre = np.zeros(n) if own_pre is None else np.asarray(own_pre, float)
    own_post = np.zeros(n) if own_post is None else np.asarray(own_post, float)
    vehicle_node, vehicle_m = _snap(network, xy, 'vehicle')
    foot_node, foot_m = _snap(network, xy, 'foot')
    has_vehicle = vehicle_node >= 0
    vehicle_node0 = np.where(has_vehicle, vehicle_node, 0)
    foot_node0 = np.where(foot_node >= 0, foot_node, 0)
    pre, likely = access.pre, access.post['likely']
    no_road = ~has_vehicle | (vehicle_m > rules.snap_vehicle_m)
    reaches_pre = pre.reaches_target(vehicle_node0, own_pre)
    road_served = ~no_road & reaches_pre
    not_connected = ~no_road & ~reaches_pre
    vehicle_post = {s: _attach(network, xy, s, 'vehicle', rules.snap_vehicle_m) for s in SCENARIOS}
    foot_post = {s: _attach(network, xy, s, 'foot', rules.snap_foot_m) for s in SCENARIOS}
    reaches_post = {
        s: vehicle_post[s][1] & access.post[s].reaches_target(vehicle_post[s][0], own_post) for s in SCENARIOS
    }
    tier = np.full(n, 'connected', dtype=object)
    # Access status follows the strongest observed break that disconnects the point.
    for scenario in SCENARIOS_BY_EVIDENCE:
        tier[road_served & ~reaches_post[scenario]] = TIER_OF_SCENARIO[scenario]
    node_likely, attached_likely, likely_m = vehicle_post['likely']
    is_own_target = own_pre > 0
    pre_m = np.where(has_vehicle, pre.distance_m[vehicle_node0], np.inf)
    post_m = np.where(attached_likely, likely.distance_m[node_likely], np.inf)
    detour = (
        road_served
        & reaches_post['bridge']
        & ~is_own_target
        & np.isfinite(pre_m)
        & np.isfinite(post_m)
        & (post_m + likely_m > rules.detour_factor * (pre_m + vehicle_m) + 1000.0)
    )
    tier[detour] = 'long_detour'
    tier[not_connected] = 'not_connected_pre_event'
    tier[no_road] = 'no_road_access_pre_event'
    reaches_verified = attached_likely & access.unverified.reaches_target(node_likely, own_post)
    no_road_pre = no_road | not_connected
    access_class, walk_h, walk_km, roadhead = _walk_access(access, tier, detour, no_road_pre, foot_post)
    foot_node_likely, foot_attached_likely, _ = foot_post['likely']
    trail_cut = (
        no_road_pre
        & (foot_node >= 0)
        & (foot_m <= rules.snap_foot_m)
        & np.isfinite(access.walk_pre_h[foot_node0])
        & ~(foot_attached_likely & np.isfinite(likely.walk_h[foot_node_likely]))
    )
    kept = road_served & ~is_own_target
    dist_pre_km = np.where(kept & np.isfinite(pre_m), (pre_m + vehicle_m) / 1000.0, np.nan)
    dist_post_km = np.where(kept & np.isfinite(post_m), (post_m + likely_m) / 1000.0, np.nan)
    extra_h = np.where(has_vehicle & attached_likely, likely.drive_h[node_likely] - pre.drive_h[vehicle_node0], np.nan)
    exit_pre = has_vehicle & (pre.exits[pre.component[vehicle_node0]] > 0)
    component_likely = likely.component[node_likely]
    exit_post = attached_likely & (likely.exits[component_likely] > 0)
    target_post = attached_likely & ((likely.targets[component_likely] - own_post) > 0)
    return dict(
        vehicle_node=vehicle_node,
        snap_m=vehicle_m,
        tier=tier,
        access=access_class,
        dist_pre_km=dist_pre_km,
        dist_post_km=dist_post_km,
        detour_km=np.where(detour, dist_post_km - dist_pre_km, np.nan),
        detour_h=np.where(detour & np.isfinite(extra_h), extra_h, np.nan),
        walk_h=walk_h,
        walk_km=walk_km,
        roadhead_node=roadhead,
        trail_cut=trail_cut,
        target_node=np.where(kept, pre.origin[vehicle_node0], -1),
        exit_pre=exit_pre,
        exit_post=exit_post,
        svc_post=target_post,
        lost_exit_only=road_served & exit_pre & ~exit_post & target_post,
        unverified=road_served & np.isin(tier, ('connected', 'long_detour')) & ~reaches_verified,
    )


def break_runs(network):
    """Connected sets of likely-blocked motorable segments (repair candidates) -> (run id per edge or -1, n_runs)."""
    blocked = np.where(network.is_vehicle & network.closed['likely'])[0]
    run = np.full(len(network.start), -1)
    if not len(blocked):
        return run, 0
    adjacency = sparse.csr_matrix(
        (np.ones(len(blocked)), (network.start[blocked], network.end[blocked])),
        shape=(network.n_nodes, network.n_nodes),
    )
    _, component = csgraph.connected_components(adjacency, directed=False)
    run_ids, run_of_edge = np.unique(component[network.start[blocked]], return_inverse=True)
    run[blocked] = np.asarray(run_of_edge).reshape(-1)
    return run, len(run_ids)


def _find_root(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def greedy_repairs(network, likely, run, n_runs, weight_nodes, unit_nodes, route_runsets, k):
    """Greedy repair ranking after Gangwal et al. (2022) -> (picks, gain of each run repaired alone)."""
    # gain of a candidate: weight of the target-less components it joins to one with a target; ties go to the shorter
    component = likely.component
    n_components = int(component.max()) + 1
    has_target = ((likely.targets + likely.exits) > 0).tolist()
    weight = np.bincount(component[np.asarray(weight_nodes, int)], minlength=n_components).tolist()
    run_edges = np.where(run >= 0)[0]
    components_of_run = [set() for _ in range(n_runs)]
    for run_id, start_component, end_component in zip(
        run[run_edges].tolist(),
        component[network.start[run_edges]].tolist(),
        component[network.end[run_edges]].tolist(),
        strict=True,
    ):
        components_of_run[run_id].add(start_component)
        components_of_run[run_id].add(end_component)
    run_length = np.bincount(run[run_edges], weights=network.length_m[run_edges], minlength=n_runs).tolist()
    # candidates: each run alone, and the runs on each cut-off settlement's usual route together
    candidates = [((r,), components_of_run[r], run_length[r]) for r in range(n_runs)]
    seen = set()
    for runset in route_runsets:
        runs = tuple(sorted({int(x) for x in runset if 0 <= int(x) < n_runs}))
        if len(runs) > 1 and runs not in seen:
            seen.add(runs)
            candidates.append(
                (runs, set().union(*[components_of_run[r] for r in runs]), sum(run_length[r] for r in runs))
            )
    parent = list(range(n_components))
    gain_alone = np.zeros(n_runs)
    picks, used = [], set()
    for rank in range(k):
        best = None
        for index, (runs, components, cost) in enumerate(candidates):
            if index in used:
                continue
            roots = {_find_root(parent, c) for c in components}
            gain = sum(weight[r] for r in roots if not has_target[r]) if any(has_target[r] for r in roots) else 0.0
            if rank == 0 and len(runs) == 1:
                gain_alone[runs[0]] = gain
            if gain > 0 and (best is None or gain > best[0] or (gain == best[0] and cost < best[2])):
                best = (gain, index, cost, roots)
        if best is None:
            break
        gain, index, cost, roots = best
        used.add(index)
        joined = {r for r in roots if not has_target[r]}
        settlements = [label for label, c in unit_nodes if _find_root(parent, c) in joined]
        root = next(r for r in roots if has_target[r])
        for r in roots:
            parent[_find_root(parent, r)] = root
        has_target[root] = True
        picks.append(
            dict(
                rank=rank + 1,
                runs=list(candidates[index][0]),
                gain=float(gain),
                length_m=float(cost),
                settlements=settlements,
            )
        )
    return picks, gain_alone


def _route_breaks(network, edges, geoms, road_names, run, sid, label):
    """Breaks along a route: runs of consecutive edges closed in the bridge scenario -> (breaks, route length m)."""
    breaks, run_sets = [], []
    current = None
    route_m = 0.0
    for edge in edges:
        if not network.closed['bridge'][edge]:
            current = None
        else:
            tier = (
                'strong'
                if network.closed['strong'][edge]
                else ('likely' if network.closed['likely'][edge] else 'bridge')
            )
            if current is None:
                current = dict(
                    sid=sid,
                    settlement=label,
                    seg=int(edge),
                    highway=str(network.highway[edge]),
                    road=_text_or_none(road_names[edge]),
                    bridge=False,
                    strong=False,
                    tier=tier,
                    km_from_settlement=round(route_m / 1000, 2),
                    length_m=0.0,
                    geometry=geoms[edge].interpolate(0.5, normalized=True),
                )
                breaks.append(current)
                run_sets.append(set())
            current['bridge'] |= bool(network.is_bridge[edge])
            current['strong'] |= bool(network.closed['strong'][edge])
            current['length_m'] += float(network.length_m[edge])
            if SCENARIOS_BY_EVIDENCE.index(tier) > SCENARIOS_BY_EVIDENCE.index(current['tier']):
                current['tier'] = tier
            if run[edge] >= 0:
                run_sets[-1].add(int(run[edge]))
        route_m += float(network.length_m[edge])
    for brk, run_ids in zip(breaks, run_sets, strict=True):
        brk['run'] = ','.join(str(x) for x in sorted(run_ids))
        brk['length_m'] = round(brk['length_m'], 1)
    return breaks, route_m


def usual_routes(network, access, segments, selected, run, target_names):
    """Usual pre-event route of each (sid, label, status, vehicle node) to its nearest target -> (routes, breaks)."""
    pre = access.pre
    geoms = segments.geometry.values
    road_names = segments['name'].values
    routes, breaks = [], []
    for sid, label, status, node in selected:
        if node < 0 or not np.isfinite(pre.distance_m[node]):
            continue
        path = [int(node)]
        while pre.predecessor[path[-1]] >= 0:
            path.append(int(pre.predecessor[path[-1]]))
        edges = [access.edge_of_pair[(min(a, b), max(a, b))] for a, b in pairwise(path)]
        route_breaks, route_m = _route_breaks(network, edges, geoms, road_names, run, sid, label)
        breaks += route_breaks
        lines = [geoms[edge] for edge in edges]
        line = linemerge(MultiLineString([g for g in lines if g.geom_type == 'LineString'])) if lines else None
        run_ids = sorted({int(x) for brk in route_breaks for x in brk['run'].split(',') if x})
        routes.append(
            dict(
                sid=sid,
                label=label,
                status=status,
                target=target_names.get(path[-1], 'nearest town / hospital'),
                route_km=round(route_m / 1000, 2),
                n_breaches=len(route_breaks),
                n_strong=int(sum(brk['strong'] for brk in route_breaks)),
                breached_km=round(sum(brk['length_m'] for brk in route_breaks) / 1000, 2),
                first_breach_km=route_breaks[0]['km_from_settlement'] if route_breaks else None,
                runs=','.join(map(str, run_ids)),
                geometry=line.simplify(ROUTE_SIMPLIFY_M) if line is not None else None,
            )
        )
    crs = segments.crs
    return (
        gpd.GeoDataFrame(routes, geometry='geometry', crs=crs) if routes else None,
        gpd.GeoDataFrame(breaks, geometry='geometry', crs=crs) if breaks else None,
    )


def osm_label(row):
    """'English (local)' label from OSM name / name:en; None if unnamed."""
    local, english = _text_or_none(row.get('name')), _text_or_none(row.get('name:en'))
    if english:
        return f'{english} ({local})' if local and local != english else english
    return local


def stable_ids(df, prefix=''):
    """Id from the OSM type and id (ohsome '@osmId', Overpass osm_type/osm_id, GDAL osm_id), else the row position."""
    ids, seen = [], set()
    for position, (_, row) in enumerate(df.iterrows()):
        osm_id = _text_or_none(row.get('@osmId')) or _text_or_none(row.get('osm_id'))
        way_id = _text_or_none(row.get('osm_way_id'))
        osm_type = _text_or_none(row.get('osm_type'))
        if osm_id and '/' in osm_id:
            feature_id = osm_id.replace('/', '')
        elif osm_id:
            geom_type = row.geometry.geom_type if row.geometry is not None else 'Point'
            feature_id = (osm_type[0] if osm_type else ('n' if geom_type == 'Point' else 'r')) + osm_id
        elif way_id:
            feature_id = 'w' + way_id
        else:
            feature_id = f'p{position}'
        feature_id = prefix + feature_id
        while feature_id in seen:
            feature_id += '_'
        seen.add(feature_id)
        ids.append(feature_id)
    return ids


def _main_country(countries, aoi_poly, net_poly):
    """Country covering most of the area, and its border inside the network extent when the extent crosses it."""
    if not len(countries):
        return {}
    share = countries.geometry.intersection(aoi_poly).area.values
    if not share.max() > 0:
        return {}
    best = int(np.argmax(share))
    country = countries.geometry.iloc[best]
    shapely.prepare(country)  # many point-in-polygon tests follow
    out = dict(main_country=country, country_name=osm_label(countries.iloc[best]))
    if net_poly.difference(country).area > 1e-3 * net_poly.area:
        border = country.boundary.intersection(net_poly.buffer(2000))
        border = border.difference(net_poly.exterior.buffer(150))  # admin polygons clipped to the extent: drop the clip
        if not border.is_empty:
            out['border'] = border
    return out


def admin_context(admin, aoi_poly, net_poly):
    """Pre-event OSM admin areas: main country, its border and the levels named as municipality, ward and district."""
    if admin is None or not len(admin) or 'admin_level' not in admin:
        return {}
    areas = admin[admin.geometry.notna() & admin.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    if 'boundary' in areas:
        areas = areas[areas['boundary'].astype(str).isin(['administrative', 'None', 'nan', '<NA>'])]
    areas['lvl'] = areas['admin_level'].astype(str).str.extract(r'^(\d+)')[0]
    areas = areas[areas['lvl'].notna()]
    if not len(areas):
        return {}
    areas['geometry'] = areas.geometry.buffer(0)
    context = dict(levels={})
    context.update(_main_country(areas[areas['lvl'] == '2'], aoi_poly, net_poly))
    present = set(areas['lvl'])
    context['muni_level'] = next((level for level in ('7', '8', '6') if level in present), None)
    context['ward_level'] = next((level for level in ('9', '10') if level in present), None)
    context['district_level'] = next(
        (level for level in ('6', '5', '4') if level in present and level != context['muni_level']), None
    )
    for level in {context['muni_level'], context['ward_level'], context['district_level']} - {None}:
        context['levels'][level] = areas[areas['lvl'] == level].reset_index(drop=True)
    return context


def admin_names(context, xy, level):
    """Name of the admin area of `level` containing each point; None outside."""
    areas = context.get('levels', {}).get(level)
    names = np.full(len(xy), None, dtype=object)
    if areas is None or not len(xy):
        return names
    point_index, area_index = shapely.STRtree(areas.geometry.values).query(
        shapely.points(xy[:, 0], xy[:, 1]), predicate='within'
    )
    labels = [osm_label(row) or '' for _, row in areas.iterrows()]
    for p, a in zip(point_index.tolist(), area_index.tolist(), strict=True):
        if names[p] is None:
            names[p] = labels[a] or None
    return names


def area_names(context, level, aoi_poly, min_share=0.05, max_names=4):
    """Names of the admin areas of `level` covering at least `min_share` of the area, largest share first."""
    areas = context.get('levels', {}).get(level)
    if areas is None:
        return []
    share = areas.geometry.intersection(aoi_poly).area.values / max(aoi_poly.area, 1.0)
    order = np.argsort(-share)
    return [osm_label(areas.iloc[i]) for i in order[:max_names] if share[i] >= min_share and osm_label(areas.iloc[i])]


def _in_country(country, xy):
    if country is None:
        return np.ones(len(xy), bool)
    return shapely.contains_xy(country, xy[:, 0], xy[:, 1]) if len(xy) else np.zeros(0, bool)


def _named_place_index(places):
    """KD-tree and labels of the named places, to name roadheads and repair sites."""
    places = places[places.geometry.notna()]
    labels = [osm_label(row) for _, row in places.iterrows()]
    xy = _point_xy(places)[np.array([label is not None for label in labels], bool)]
    return (cKDTree(xy) if len(xy) else None), [label for label in labels if label is not None]


def _nearest_names(place_index, xy):
    tree, labels = place_index
    if tree is None or not len(xy):
        return [None] * len(xy)
    distance, nearest = tree.query(np.asarray(xy, float).reshape(-1, 2))
    return [labels[j] if d <= NAME_SEARCH_M else None for d, j in zip(distance, nearest, strict=True)]


def _health_kind(facility):
    amenity, healthcare = str(facility.get('amenity')), str(facility.get('healthcare'))
    if 'hospital' in (amenity, healthcare):
        return 'hospital'
    for value in (amenity, healthcare):
        if value not in ('None', 'nan', '<NA>', ''):
            return value.replace('_', ' ')
    return 'health facility'


def _health_facilities(health, likely_near, possible_near, transform, country):
    """Health facilities with id, label, kind, position and footprint / cross-border flags."""
    if health is None or not len(health):
        return pd.DataFrame([])
    facilities = health[health.geometry.notna()].reset_index(drop=True)
    xy = _point_xy(facilities)
    ids = stable_ids(facilities, 'h')
    in_likely = _values_at(likely_near, transform, xy[:, 0], xy[:, 1]).astype(bool)
    in_possible = _values_at(possible_near, transform, xy[:, 0], xy[:, 1]).astype(bool)
    cross_border = ~_in_country(country, xy)
    rows = []
    for k, (_, facility) in enumerate(facilities.iterrows()):
        kind = _health_kind(facility)
        label = osm_label(facility)
        rows.append(
            dict(
                hid=ids[k],
                label=label or f'unnamed {kind}',
                named=label is not None,
                kind=kind,
                x=xy[k, 0],
                y=xy[k, 1],
                fp_likely=bool(in_likely[k]),
                fp_possible=bool(in_possible[k]),
                cross_border=bool(cross_border[k]),
            )
        )
    return pd.DataFrame(rows)


def _targets(facilities, town_places, network, country, snap_vehicle_m):
    """Hospitals (else any health facility) and towns of the main country, snapped to the roads -> (targets, counts)."""
    rows, info = [], {}
    n_cross_border = 0
    if len(facilities):
        domestic = facilities[~facilities['cross_border']]
        hospitals = domestic[domestic['kind'] == 'hospital']
        chosen = hospitals if len(hospitals) else domestic
        info['target_health'] = (
            'hospitals' if len(hospitals) else ('all health facilities (no hospital mapped)' if len(chosen) else 'none')
        )
        for _, facility in chosen.iterrows():
            rows.append(
                dict(
                    kind='hospital' if facility['kind'] == 'hospital' else 'health',
                    label=facility['label'],
                    x=facility['x'],
                    y=facility['y'],
                    post_ok=not facility['fp_likely'],
                    hid=facility['hid'],
                )
            )
        n_cross_border = int(facilities['cross_border'].sum())
    if town_places is not None and len(town_places) and 'place' in town_places:
        towns = town_places[town_places['place'].astype(str).isin(TOWN_PLACES)].reset_index(drop=True)
        town_xy = _point_xy(towns)
        domestic_town = _in_country(country, town_xy)
        n_cross_border += int((~domestic_town).sum())
        for k, (_, town) in enumerate(towns.iterrows()):
            if domestic_town[k]:
                label = osm_label(town) or f'unnamed {town.get("place")}'
                rows.append(dict(kind='town', label=label, x=town_xy[k, 0], y=town_xy[k, 1], post_ok=True, hid=None))
    targets = pd.DataFrame(rows, columns=['kind', 'label', 'x', 'y', 'post_ok', 'hid'])
    node, distance = _snap(network, targets[['x', 'y']].values.astype(float), 'vehicle')
    targets['node'] = np.where(distance <= snap_vehicle_m, node, -1)
    snapped = targets['node'].values >= 0
    info.update(
        n_services=int(snapped.sum()),
        n_hospitals=int(((targets['kind'] == 'hospital') & snapped).sum()),
        n_health_targets=int(((targets['kind'] == 'health') & snapped).sum()),
        n_towns=int(((targets['kind'] == 'town') & snapped).sum()),
        n_services_unusable_post=int(((~targets['post_ok'].astype(bool)) & snapped).sum()),
        n_targets_cross_border=n_cross_border,
    )
    return targets, info


def _target_names(targets, network, segments):
    """Display name of each target node: the town or health facility, or the road leaving the area at an exit."""
    names = {}
    for _, target in targets[targets['node'] >= 0].iterrows():
        label, kind = target['label'], target['kind']
        node = int(target['node'])
        says_kind = 'hospital' in label.lower() or 'health' in label.lower() or label.startswith('unnamed')
        if node not in names or kind == 'hospital':
            names[node] = label if kind == 'town' or says_kind else f'{label} ({kind})'
    road_names, refs = segments['name'].values, segments['ref'].values
    exit_nodes = set(network.exits.tolist())
    exit_edges = network.is_vehicle & np.isin(network.highway, list(EXIT_FALLBACK_HIGHWAYS))
    for edge in np.where(exit_edges)[0]:
        for node in (int(network.start[edge]), int(network.end[edge])):
            if node in exit_nodes and node not in names:
                road = _text_or_none(refs[edge]) or _text_or_none(road_names[edge])
                names[node] = 'road out of the area' + (f' ({road})' if road else f' ({network.highway[edge]})')
    return names


def _settlement_points(places, crs, aoi_poly, likely_near, transform, country):
    """Settlement places with label, id and area / footprint / border flags -> (settlements, xy, localities dropped)."""
    settlements = (
        places[places.geometry.notna()].copy() if places is not None else gpd.GeoDataFrame(geometry=[], crs=crs)
    )
    n_localities = 0
    if len(settlements) and 'place' in settlements:
        place_type = settlements['place'].astype(str)
        n_localities = int((place_type == 'locality').sum())
        settlements = settlements[place_type.isin(SETTLEMENT_PLACES)]
    settlements = settlements.reset_index(drop=True)
    settlements['label'] = [osm_label(row) or 'unnamed' for _, row in settlements.iterrows()]
    settlements['named'] = settlements['label'] != 'unnamed'
    settlements['sid'] = stable_ids(settlements, 's')
    xy = _point_xy(settlements)
    if len(settlements):
        settlements['in_aoi'] = shapely.contains_xy(aoi_poly, xy[:, 0], xy[:, 1])
        settlements['pt_fp'] = _values_at(likely_near, transform, xy[:, 0], xy[:, 1]).astype(bool)
        settlements['cross_border'] = ~_in_country(country, xy)
    else:
        settlements['in_aoi'] = True
        settlements['pt_fp'] = False
        settlements['cross_border'] = False
    return settlements, xy, n_localities


def _add_place_access(settlements, xy, targets, network, access, rules, target_names, assessed):
    """Add the access columns of `classify_access` and the status to the settlements."""
    towns = targets[(targets['kind'] == 'town') & (targets['node'] >= 0)]
    is_town_target = np.zeros(len(settlements))
    if len(settlements) and len(towns):
        distance, _ = cKDTree(towns[['x', 'y']].values.astype(float)).query(xy)
        is_town_target = (distance <= 5.0).astype(float)
    result = classify_access(network, access, xy, rules, is_town_target, is_town_target)
    for column in PLACE_ACCESS_COLUMNS:
        settlements[column] = result[column]
    settlements['veh_node'] = result['vehicle_node']
    settlements['roadhead_node'] = result['roadhead_node']
    settlements['target'] = [target_names.get(int(node)) if node >= 0 else None for node in result['target_node']]
    settlements['status'] = np.where(settlements['cross_border'], 'cross_border', settlements['tier'])
    if not assessed:
        settlements['status'] = 'not_assessed'
    settlements['snap_dist_m'] = result['snap_m']


def _building_access(bld, network, access, rules, country, assessed):
    """Access of each building, its footprint class (likely / possible) and whether it lies across the border."""
    xy = _point_xy(bld)
    result = classify_access(network, access, xy, rules)
    status = bld['status'].astype(str).values
    buildings = pd.DataFrame(
        dict(
            x=xy[:, 0],
            y=xy[:, 1],
            tier=result['tier'],
            access=result['access'],
            walk_h=result['walk_h'],
            walk_km=result['walk_km'],
            roadhead=result['roadhead_node'],
            trail_cut=result['trail_cut'],
            detour_km=result['detour_km'],
            vehicle_node=result['vehicle_node'],
            in_footprint=status == 'likely_affected',
            in_possible_footprint=status == 'possibly_affected',
            cross_border=~_in_country(country, xy),
            unverified=result['unverified'],
        )
    )
    if not assessed:
        buildings['tier'] = 'not_assessed'
        buildings['access'] = 'not_assessed'
    return buildings


def _assign_units(settlements, xy, buildings, named_index, unit_m):
    """Unit of each building and place: nearest named place within unit_m (U) or beyond (O), else a 2 km cell (G)."""
    named_tree = cKDTree(xy[named_index]) if len(named_index) else None
    named_sids = settlements['sid'].values[named_index]
    if buildings is not None:
        building_xy = buildings[['x', 'y']].to_numpy()
        if named_tree is None:
            buildings['unit'] = [f'G{int(x // 2000)}_{int(y // 2000)}' for x, y in building_xy]
        else:
            distance, nearest = named_tree.query(building_xy)
            buildings['unit'] = [
                ('U' if is_core else 'O') + sid
                for is_core, sid in zip(distance <= unit_m, named_sids[nearest], strict=True)
            ]
    if named_tree is None:
        settlements['unit'] = None
        settlements['near'] = None
        return
    distance, nearest = named_tree.query(xy)
    named = settlements['named'].values
    labels = settlements['label'].values
    settlements['unit'] = [
        ('U' if (named[i] or distance[i] <= unit_m) else 'O') + named_sids[nearest[i]] for i in range(len(settlements))
    ]
    settlements['near'] = [None if named[i] else labels[named_index[nearest[i]]] for i in range(len(settlements))]


def _building_counts(buildings):
    """Buildings by settlement, flood exposure and access status, with median walk to a connected road."""
    domestic = buildings[~buildings['cross_border']]
    by_unit = domestic.groupby('unit')
    counts = pd.DataFrame(
        dict(
            n_bld=by_unit.size(),
            n_fp=by_unit['in_footprint'].sum(),
            n_fpp=by_unit['in_possible_footprint'].sum(),
            x=by_unit['x'].mean(),
            y=by_unit['y'].mean(),
        )
    )
    # Buildings within the flood extent are counted separately from buildings without road access.
    off_footprint = domestic[~domestic['in_footprint']]
    by_unit = off_footprint.groupby('unit')
    for tier in NO_ROAD_TIERS:
        counts['t_' + tier] = by_unit['tier'].apply(lambda tiers, tier=tier: int((tiers == tier).sum()))
    for access_class in ('walk_in', 'air_only', 'detour', 'no_road_pre'):
        counts['a_' + access_class] = by_unit['access'].apply(
            lambda classes, access_class=access_class: int((classes == access_class).sum())
        )
    counts['trail_cut'] = by_unit['trail_cut'].sum()
    counts['n_unver'] = by_unit['unverified'].sum()
    counts = counts.fillna(0)
    no_road = off_footprint[off_footprint['tier'].isin(NO_ROAD_TIERS)]
    walk_in = no_road[no_road['access'] == 'walk_in']
    if len(walk_in):
        counts = counts.join(
            walk_in.groupby('unit').agg(
                walk_h=('walk_h', 'median'),
                walk_km=('walk_km', 'median'),
                roadhead=('roadhead', lambda nodes: int(nodes.mode().iloc[0])),
            )
        )
    return counts


def _optional(counts, key, cast, default=None):
    return cast(counts[key]) if counts is not None and key in counts and pd.notna(counts[key]) else default


def _count(counts, key):
    return _optional(counts, key, int, 0)


def _unit_row(uid, counts, settlements, xy, base):
    """One unit; `counts` is its row of `_building_counts` (or None), `base` the position of its settlement point."""
    if base is None:
        x, y = counts['x'], counts['y']
        label, place, point_status, near = 'Buildings away from any named settlement', 'outlying', None, None
    else:
        point = settlements.iloc[base]
        near = point['label']
        if uid[0] == 'U':
            x, y = xy[base, 0], xy[base, 1]
            label, place, point_status = point['label'], str(point.get('place')), point['status']
        else:
            x, y = (xy[base, 0], xy[base, 1]) if counts is None else (counts['x'], counts['y'])
            label, place, point_status = f'Outlying buildings near {point["label"]}', 'outlying', None
    high, possible, bridge = (
        _count(counts, 't_cut_off'),
        _count(counts, 't_possibly_cut_off'),
        _count(counts, 't_bridge_uncertain'),
    )
    no_road = high + possible + bridge
    row = dict(
        uid=uid,
        sid=uid[1:] if base is not None else None,
        label=label,
        near=near,
        place=place,
        point_status=point_status,
        x=float(x),
        y=float(y),
        n_bld=_count(counts, 'n_bld'),
        n_fp=_count(counts, 'n_fp'),
        n_fp_possible=_count(counts, 'n_fpp'),
        n_noroad=no_road,
        n_noroad_high=high,
        n_noroad_possible=possible,
        n_noroad_bridge=bridge,
        n_walk=_count(counts, 'a_walk_in'),
        n_air=_count(counts, 'a_air_only'),
        n_detour=_count(counts, 'a_detour'),
        n_no_road_pre=_count(counts, 'a_no_road_pre'),
        n_trail_cut=_count(counts, 'trail_cut'),
        n_unverified=_count(counts, 'n_unver'),
        walk_h=_optional(counts, 'walk_h', float),
        walk_km=_optional(counts, 'walk_km', float),
        roadhead_node=_optional(counts, 'roadhead', int, -1),
    )
    point_without_road = point_status in NO_ROAD_TIERS
    if no_road:
        row['access'] = 'air_only' if row['n_air'] > row['n_walk'] else 'walk_in'
        row['conf'] = 'high' if high * 2 >= no_road else ('possible' if (high + possible) * 2 >= no_road else 'bridge')
    elif point_without_road and row['n_bld'] == 0:
        point = settlements.iloc[base]  # without mapped buildings the settlement point decides
        row['access'] = point['access']
        row['conf'] = {'cut_off': 'high', 'possibly_cut_off': 'possible'}.get(point_status, 'bridge')
        row['walk_h'] = None if pd.isna(point['walk_h']) else float(point['walk_h'])
        row['walk_km'] = None if pd.isna(point['walk_km']) else float(point['walk_km'])
        row['roadhead_node'] = int(point['roadhead_node'])
    else:
        row['access'] = None
        row['conf'] = None
    # A settlement point within the flood extent does not establish that the other buildings have lost access.
    n_off_footprint = row['n_bld'] - row['n_fp']
    by_buildings = bool(no_road >= 0.5 * n_off_footprint) if n_off_footprint > 0 else False
    by_point = bool(
        point_without_road and not bool(settlements['pt_fp'].iloc[base]) and (no_road > 0 or row['n_bld'] == 0)
    )
    row['noroad_unit'] = by_buildings or by_point or bool(row['n_bld'] == 0 and point_without_road)
    row['counted_by'] = 'buildings' if by_buildings else ('point' if row['noroad_unit'] else None)
    if base is not None:
        row['adm_muni'] = settlements['adm_muni'].iloc[base]
        row['adm_ward'] = settlements['adm_ward'].iloc[base]
    row['n_unnamed_places'] = int(((settlements['unit'] == uid) & ~settlements['named']).sum())
    return row


def _unit_table(settlements, xy, counts, named_index):
    """Settlement units: each named place (U), its outlying buildings (O) and 2 km cells of other buildings (G)."""
    position_of_sid = {sid: i for i, sid in enumerate(settlements['sid'])}
    unit_ids = dict.fromkeys(
        [*([] if counts is None else counts.index), *('U' + sid for sid in settlements['sid'].values[named_index])]
    )
    rows = []
    for uid in unit_ids:
        unit_counts = counts.loc[uid] if counts is not None and uid in counts.index else None
        base = position_of_sid.get(uid[1:]) if uid[0] in 'UO' else None
        rows.append(_unit_row(uid, unit_counts, settlements, xy, base))
    return pd.DataFrame(rows)


def _add_roadhead_names(units, network, place_index):
    roadhead_node = units['roadhead_node'].values.astype(int)
    units['roadhead'] = [None] * len(units)
    has_roadhead = roadhead_node >= 0
    if has_roadhead.any():
        units.loc[has_roadhead, 'roadhead'] = _nearest_names(place_index, network.nodes[roadhead_node[has_roadhead]])


def _repair_weights(buildings, settlements, area_mask):
    """Vehicle nodes a repair should reconnect: cut-off buildings outside the footprint, else cut-off settlements."""
    if buildings is not None:
        weighted = buildings[
            ~buildings['cross_border']
            & ~buildings['in_footprint']
            & buildings['tier'].isin(CUT_OFF_TIERS)
            & (buildings['vehicle_node'] >= 0)
        ]
        return weighted['vehicle_node'].values.astype(int), 'buildings'
    weighted = settlements[area_mask & settlements['tier'].isin(CUT_OFF_TIERS)]
    return weighted['veh_node'].values.astype(int), 'settlements'


def _cut_off_units(settlements, area_mask, likely):
    """(label, component in the likely scenario) of the named cut-off settlements."""
    return [
        (settlements['label'].iloc[i], int(likely.component[settlements['veh_node'].iloc[i]]))
        for i in range(len(settlements))
        if area_mask[i]
        and settlements['named'].iloc[i]
        and settlements['tier'].iloc[i] in CUT_OFF_TIERS
        and settlements['veh_node'].iloc[i] >= 0
    ]


def _route_settlements(settlements, area_mask):
    """(sid, label, status, vehicle node) of the named settlements that lost road access or need a long detour."""
    return [
        (
            settlements['sid'].iloc[i],
            settlements['label'].iloc[i],
            settlements['status'].iloc[i],
            int(settlements['veh_node'].iloc[i]),
        )
        for i in range(len(settlements))
        if area_mask[i] and settlements['named'].iloc[i] and settlements['status'].iloc[i] in LISTED_TIERS
    ]


def _run_table(network, run, n_runs, segments, breaks, gain_alone, place_index):
    """One row per break run: length, bridge, evidence, road, repair gain and the settlements whose route crosses it."""
    if not n_runs:
        return None
    road_names, refs = segments['name'].values, segments['ref'].values
    settlements_of_run = defaultdict(set)
    if breaks is not None:
        for _, brk in breaks.iterrows():
            for run_id in str(brk['run']).split(','):
                if run_id:
                    settlements_of_run[int(run_id)].add(brk['settlement'])
    edges = np.where(run >= 0)[0]
    edges = edges[np.argsort(run[edges], kind='stable')]
    cuts = np.r_[0, np.where(np.diff(run[edges]) != 0)[0] + 1, len(edges)]
    geoms = segments.geometry.values
    rows, middles = [], []
    for first, stop in pairwise(cuts):
        run_edges = edges[first:stop]
        run_id = int(run[run_edges[0]])
        longest = run_edges[np.argmax(network.length_m[run_edges])]
        middle = geoms[run_edges[len(run_edges) // 2]].interpolate(0.5, normalized=True)
        middles.append((middle.x, middle.y))
        roads = [
            r for r in dict.fromkeys([_text_or_none(refs[e]) or _text_or_none(road_names[e]) for e in run_edges]) if r
        ]
        crossing = settlements_of_run.get(run_id, ())
        rows.append(
            dict(
                run=run_id,
                length_m=round(float(network.length_m[run_edges].sum()), 1),
                n_segments=len(run_edges),
                bridge=bool(network.is_bridge[run_edges].any()),
                strong=bool(network.closed['strong'][run_edges].all()),
                strong_any=bool(network.closed['strong'][run_edges].any()),
                highway=str(network.highway[longest]),
                road=', '.join(roads[:2]) or None,
                gain_alone=float(gain_alone[run_id]),
                n_settlements_route=len(crossing),
                settlements_route=', '.join(sorted(crossing)[:8]),
                geometry=middle,
            )
        )
    for row, name in zip(rows, _nearest_names(place_index, np.array(middles)), strict=True):
        row['near'] = name
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=segments.crs)


def _repair_rows(runs_table, picks, gain_unit):
    if runs_table is None:
        return []
    run_rows = {int(row['run']): row for _, row in runs_table.iterrows()}
    rows = []
    for pick in picks:
        runs = [run_rows[run_id] for run_id in pick['runs'] if run_id in run_rows]
        lead = max(runs, key=lambda row: row['length_m'])
        rows.append(
            dict(
                rank=pick['rank'],
                gain=round(pick['gain']),
                gain_unit=gain_unit,
                n_settlements=len(pick['settlements']),
                settlements=', '.join(pick['settlements'][:8]),
                n_runs=len(runs),
                runs=','.join(str(row['run']) for row in runs),
                length_m=round(sum(row['length_m'] for row in runs), 1),
                bridge=any(row['bridge'] for row in runs),
                strong=all(row['strong'] for row in runs),
                highway=lead['highway'],
                road=lead['road'],
                near=lead['near'],
                geometry=lead['geometry'],
            )
        )
    return rows


def _health_status(facilities, targets, network, access, rules, aoi_poly, crs):
    """Status of each health facility: in the footprint, cut off, without road before the event, or ok."""
    if not len(facilities):
        return None
    xy = facilities[['x', 'y']].values.astype(float)
    snapped = targets[targets['hid'].notna() & (targets['node'] >= 0)]
    own_pre = facilities['hid'].isin(set(snapped['hid'])).astype(float).values
    own_post = facilities['hid'].isin(set(snapped.loc[snapped['post_ok'].astype(bool), 'hid'])).astype(float).values
    result = classify_access(network, access, xy, rules, own_pre, own_post)
    status = np.full(len(facilities), 'ok', dtype=object)
    status[result['tier'] == 'no_road_access_pre_event'] = 'no_road_pre'
    status[facilities['fp_possible'].values] = 'possibly_in_footprint'
    status[np.isin(result['tier'], ('possibly_cut_off', 'bridge_uncertain'))] = 'possibly_cut_off'
    status[result['tier'] == 'cut_off'] = 'cut_off'
    status[facilities['fp_likely'].values] = 'in_footprint'
    status[facilities['cross_border'].values] = 'cross_border'
    return gpd.GeoDataFrame(
        facilities.drop(columns=['x', 'y']).assign(
            status=status,
            tier=result['tier'],
            access=result['access'],
            in_aoi=shapely.contains_xy(aoi_poly, xy[:, 0], xy[:, 1]),
        ),
        geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]),
        crs=crs,
    )


def _exit_points(network, target_names, crs):
    """One point per 2 km cell where a main road leaves the analysed network."""
    if not len(network.exits):
        return None
    xy = network.nodes[network.exits]
    first_in_cell = np.unique(np.floor(xy / 2000.0), axis=0, return_index=True)[1]
    return gpd.GeoDataFrame(
        dict(name=[target_names.get(int(network.exits[i]), 'road out of the area') for i in first_in_cell]),
        geometry=gpd.points_from_xy(xy[first_in_cell, 0], xy[first_in_cell, 1]),
        crs=crs,
    )


def _building_access_layer(buildings, crs):
    """Buildings without road access, on a long detour, with a lost trail or behind unobserved roads."""
    if buildings is None:
        return None
    listed = buildings[buildings['tier'].isin(LISTED_TIERS) | buildings['trail_cut'] | buildings['unverified']]
    if not len(listed):
        return None
    columns = ['tier', 'access', 'walk_h', 'walk_km', 'detour_km', 'trail_cut', 'in_footprint', 'unit', 'unverified']
    return gpd.GeoDataFrame(listed[columns], geometry=gpd.points_from_xy(listed['x'], listed['y']), crs=crs)


def _rounded(value, digits=1):
    if value is None or not np.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _settlement_summary(settlements, area_mask, n_localities, units):
    counted = settlements[area_mask]
    named = counted[counted['named']]
    lost_exit = counted['lost_exit_only'] & counted['named']
    summary = dict(
        by_status=_value_counts(counted['status']),
        named_by_status=_value_counts(named['status']),
        unnamed_places=int((~counted['named']).sum()),
        cross_border=int((settlements['cross_border'] & settlements['in_aoi']).sum()),
        localities_dropped=int(n_localities),
        lost_exit_only=int(lost_exit.sum()),
        lost_exit_only_names=counted[lost_exit]['label'].astype(str).tolist()[:12],
        access_named=_value_counts(counted[counted['named'] & counted['tier'].isin(NO_ROAD_TIERS)]['access']),
        unverified_named=int(named['unverified'].astype(bool).sum()),
        unverified_names=named[named['unverified'].astype(bool)]['label'].astype(str).tolist()[:12],
        via_roadhead_named=0,
    )
    if len(units):
        named_units = units[units['uid'].str.startswith('U')]
        summary['named_noroad_units'] = _value_counts(named_units[named_units['noroad_unit']]['conf'])
    return summary


def _building_summary(buildings):
    counted = buildings[~buildings['cross_border']]
    tier, access_class = counted['tier'], counted['access']
    hit = counted['in_footprint']
    no_road = tier.isin(NO_ROAD_TIERS) & ~hit
    return dict(
        total=len(counted),
        cross_border=int(buildings['cross_border'].sum()),
        no_road_high=int(((tier == 'cut_off') & ~hit).sum()),
        no_road_possible=int(((tier == 'possibly_cut_off') & ~hit).sum()),
        no_road_bridge=int(((tier == 'bridge_uncertain') & ~hit).sum()),
        walk_in=int((no_road & (access_class == 'walk_in')).sum()),
        air_only=int((no_road & (access_class == 'air_only')).sum()),
        air_only_high=int(((tier == 'cut_off') & (access_class == 'air_only') & ~hit).sum()),
        detour=int(((access_class == 'detour') & ~hit).sum()),
        no_road_pre=int(((access_class == 'no_road_pre') & ~hit).sum()),
        trail_cut=int((counted['trail_cut'] & ~hit).sum()),
        in_footprint=int(hit.sum()),
        fp_class=True,
        in_footprint_no_road=int((tier.isin(NO_ROAD_TIERS) & hit).sum()),
        via_roadhead=0,
        unverified=int((counted['unverified'] & ~hit).sum()),
    )


def _priority_row(unit, routes_by_sid):
    route = routes_by_sid.get(unit['sid']) if unit['sid'] is not None and str(unit['uid']).startswith('U') else None
    return dict(
        uid=unit['uid'],
        label=unit['label'],
        near=unit['near'],
        place=unit['place'],
        muni=unit.get('adm_muni'),
        ward=unit.get('adm_ward'),
        lat=None,
        lon=None,
        x=round(unit['x'], 1),
        y=round(unit['y'], 1),
        n_bld=int(unit['n_bld']),
        n_fp=int(unit['n_fp']),
        n_noroad=int(unit['n_noroad']),
        n_high=int(unit['n_noroad_high']),
        n_possible=int(unit['n_noroad_possible']),
        n_bridge=int(unit['n_noroad_bridge']),
        n_air=int(unit['n_air']),
        n_walk=int(unit['n_walk']),
        access=unit['access'],
        conf=unit['conf'],
        walk_km=_rounded(unit['walk_km']),
        walk_h=_rounded(unit['walk_h']),
        roadhead=unit['roadhead'],
        status=unit['point_status'],
        counted=bool(unit['noroad_unit']),
        n_unverified=int(unit['n_unverified']),
        target=None if route is None else route['target'],
        first_break_km=None if route is None else _rounded(route['first_breach_km'], 2),
        n_breaks=None if route is None else int(route['n_breaches']),
        spof=bool(route is not None and int(route['n_breaches']) == 1 and int(route['n_strong']) == 0),
    )


def _hardest_hit_row(unit):
    return dict(
        uid=unit['uid'],
        label=unit['label'],
        near=unit['near'],
        place=unit['place'],
        muni=unit.get('adm_muni'),
        x=round(unit['x'], 1),
        y=round(unit['y'], 1),
        n_bld=int(unit['n_bld']),
        n_fp=int(unit['n_fp']),
        n_fp_possible=int(unit['n_fp_possible']),
        status=unit['point_status'],
        access=unit['access'],
    )


def _unit_summary(units, routes_by_sid):
    """Units without road access, the priority table and the hardest-hit units."""
    is_named = units['uid'].str.startswith('U')
    named = units[is_named]
    named_no_road = named[named['noroad_unit']]
    counts = dict(
        n=len(units),
        named=len(named),
        rule='settlement point (outside the footprint) or >= 50% of the buildings outside the footprint',
        noroad_high=int((named_no_road['conf'] == 'high').sum()),
        noroad_any=len(named_no_road),
        noroad_any_all=int(units['noroad_unit'].sum()),
        air_only=int((named_no_road['access'] == 'air_only').sum()),
        outlying_noroad_bld=int(units.loc[~is_named, 'n_noroad'].sum()),
        partial=int(((named['n_noroad'] > 0) & ~named['noroad_unit']).sum()),
    )
    priority = units[units['noroad_unit'] | (units['n_noroad'] >= PRIORITY_MIN_BUILDINGS)].copy()
    counts['n_priority'] = len(priority)
    priority = priority.sort_values(['n_noroad', 'n_noroad_high', 'n_bld'], ascending=False)
    hit_share = (units['n_fp'] >= 3) & (units['n_fp'] >= 0.3 * units['n_bld'].clip(lower=1))
    hardest = units[(units['n_fp'] >= 10) | hit_share].sort_values('n_fp', ascending=False)
    return dict(
        units=counts,
        priority=[_priority_row(unit, routes_by_sid) for _, unit in priority.head(25).iterrows()],
        hardest_hit=[_hardest_hit_row(unit) for _, unit in hardest.head(10).iterrows()],
    )


def _labels_with_status(health_status, status):
    return health_status[health_status['status'] == status]['label'].astype(str).tolist()[:15]


def _health_summary(health_status):
    in_area = health_status[health_status['in_aoi'] & (health_status['status'] != 'cross_border')]
    return dict(
        total=len(in_area),
        by_status=_value_counts(in_area['status']),
        in_footprint=_labels_with_status(in_area, 'in_footprint'),
        cut_off=_labels_with_status(in_area, 'cut_off'),
        possibly_cut_off=_labels_with_status(in_area, 'possibly_cut_off'),
        hospitals_in_aoi=int((in_area['kind'] == 'hospital').sum()),
    )


def _network_info(network, net_source, context):
    return dict(
        network=net_source,
        n_nodes=network.n_nodes,
        n_edges=len(network.start),
        n_vehicle_edges=int(network.is_vehicle.sum()),
        n_exits=len(network.exits),
        exit_classes=network.exit_classes,
        n_border_closed_edges=int(network.crosses_border.sum()),
        country=context.get('country_name'),
        border_known=context.get('main_country') is not None,
    )


def _warnings(assessed, n_services, n_exits, net_source):
    warnings = []
    if not assessed:
        warnings.append(
            'No town, hospital or main road leaving the area was found in the pre-event road network, so road access '
            'cannot be assessed (cut-off results are not computed).'
        )
    elif n_services == 0:
        warnings.append(
            'No town or hospital in the analysed road network: "cut off" here means the road connection out of '
            'the area (main roads leaving the analysed network) is lost.'
        )
    if not n_exits:
        warnings.append(
            'No main road leaves the analysed road network: only towns and hospitals inside it are used as '
            'destinations.'
        )
    if net_source != 'buffered':
        warnings.append(
            'The road network was analysed only inside the area (the buffered network was not available): '
            'settlements near the edge may be wrongly shown as cut off or as having no road.'
        )
    return warnings


def road_access(  # noqa: PLR0913
    segments,
    places,
    *,
    net_places=None,
    health=None,
    bld=None,
    hazard,
    transform,
    aoi_poly,
    net_poly,
    admin=None,
    snap_veh=500.0,
    snap_foot=500.0,
    unit_m=1500.0,
    detour_factor=1.5,
    n_repairs=5,
    net_source='aoi',
):
    """Post-event road access of settlements, buildings and health facilities, with routes, breaks and repairs."""
    crs = segments.crs
    rules = AccessRules(snap_veh, snap_foot, detour_factor)
    context = admin_context(admin, aoi_poly, net_poly)
    country = context.get('main_country')
    network = build_network(segments, net_poly, context.get('border'), country)
    likely_near = ndimage.binary_dilation(hazard == LIKELY, iterations=1).astype('uint8')
    possible_near = ndimage.binary_dilation(hazard >= POSSIBLE, iterations=2).astype('uint8')
    facilities = _health_facilities(health, likely_near, possible_near, transform, country)
    has_net_places = net_places is not None and len(net_places) > 0
    targets, target_info = _targets(facilities, net_places if has_net_places else places, network, country, snap_veh)
    access = solve_access(network, targets['node'].values, targets['post_ok'].values.astype(bool))
    info = {**_network_info(network, net_source, context), **target_info, 'n_targets': access.n_sources_pre}
    assessed = access.n_sources_pre > 0
    target_names = _target_names(targets, network, segments)
    settlements, xy, n_localities = _settlement_points(places, crs, aoi_poly, likely_near, transform, country)
    _add_place_access(settlements, xy, targets, network, access, rules, target_names, assessed)
    area_mask = settlements['in_aoi'].values.astype(bool) & ~settlements['cross_border'].values.astype(bool)
    named_index = np.where(area_mask & settlements['named'].values)[0]
    buildings = (
        _building_access(bld, network, access, rules, country, assessed) if bld is not None and len(bld) else None
    )
    _assign_units(settlements, xy, buildings, named_index, unit_m)
    settlements['adm_muni'] = admin_names(context, xy, context.get('muni_level'))
    settlements['adm_ward'] = admin_names(context, xy, context.get('ward_level'))
    place_index = _named_place_index(net_places if has_net_places else settlements)
    units = _unit_table(settlements, xy, _building_counts(buildings) if buildings is not None else None, named_index)
    if len(units):
        _add_roadhead_names(units, network, place_index)
    run, n_runs = break_runs(network)
    routes, breaks = (
        usual_routes(network, access, segments, _route_settlements(settlements, area_mask), run, target_names)
        if assessed
        else (None, None)
    )
    routes_by_sid = {} if routes is None else {route['sid']: route for _, route in routes.iterrows()}
    route_runsets = [
        [int(x) for x in str(route['runs']).split(',') if x]
        for route in routes_by_sid.values()
        if route['status'] in NO_ROAD_TIERS
    ]
    weight_nodes, gain_unit = _repair_weights(buildings, settlements, area_mask)
    likely = access.post['likely']
    picks, gain_alone = (
        greedy_repairs(
            network,
            likely,
            run,
            n_runs,
            weight_nodes,
            _cut_off_units(settlements, area_mask, likely),
            route_runsets,
            n_repairs,
        )
        if assessed and n_runs
        else ([], np.zeros(n_runs))
    )
    runs_table = _run_table(network, run, n_runs, segments, breaks, gain_alone, place_index)
    repair_rows = _repair_rows(runs_table, picks, gain_unit)
    health_status = _health_status(facilities, targets, network, access, rules, aoi_poly, crs)
    info.update(
        n_dangle_links=0,
        unobserved_in_grid_veh_km=round(
            float(network.length_m[network.is_vehicle & network.is_unverified].sum() / 1000), 2
        ),
        q3_options=dict(ACCESS_RULES),
    )
    summary = dict(
        version=2,
        assessed=bool(assessed),
        net=info,
        warnings=_warnings(assessed, info['n_services'], len(network.exits), net_source),
        settlements=_settlement_summary(settlements, area_mask, n_localities, units),
    )
    if buildings is not None:
        summary['buildings'] = _building_summary(buildings)
    if len(units):
        summary.update(_unit_summary(units, routes_by_sid))
    summary['repairs'] = [{k: v for k, v in row.items() if k != 'geometry'} for row in repair_rows]
    summary['n_break_runs'] = int(n_runs)
    if health_status is not None:
        summary['health'] = _health_summary(health_status)
    if context:
        summary['areas'] = dict(
            country=context.get('country_name'),
            districts=area_names(context, context.get('district_level'), aoi_poly),
            municipalities=area_names(context, context.get('muni_level'), aoi_poly, 0.03, 6),
        )
    return dict(
        settlements=settlements,
        units=gpd.GeoDataFrame(units, geometry=gpd.points_from_xy(units['x'], units['y']), crs=crs)
        if len(units)
        else None,
        routes=routes,
        breaks=breaks,
        runs=runs_table,
        repairs=gpd.GeoDataFrame(repair_rows, geometry='geometry', crs=crs) if repair_rows else None,
        health=health_status,
        exits=_exit_points(network, target_names, crs),
        bld_access=_building_access_layer(buildings, crs),
        summary=summary,
    )


def _json_scalar(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is None or isinstance(value, (int, str)):
        return value
    return None if pd.api.types.is_scalar(value) and pd.isna(value) else str(value)


def jsonable(value):
    """Plain JSON values from numpy / pandas values; NaN and inf become None."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return _json_scalar(value)
