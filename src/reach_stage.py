"""Estimate the flood level along river reaches and fill cloud-covered valley ground."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from rasterio.warp import Resampling, reproject

from fusion import area_km2, wide_water_mask

D8_OFFSETS = {  # pysheds D8 code -> (row, col) step
    64: (-1, 0),
    128: (-1, 1),
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
}
STAGE_LEVELS_M = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60], 'float32')

DEM_CELL_M = 30.0
CHANNEL_MIN_CELLS = 2000
REACH_LENGTH_M = 300.0
MAX_POINTER_JUMPS = 64

EVIDENCE_TEMPERATURE = 10.0
FLOOD_COST = 0.5
STAGE_STEP_COST = 1.0
EM_ITERATIONS = 4
PRIOR_WEIGHT_PX = 2000.0
DETECTION_RATE_LIMITS = (0.15, 0.85)
MEASURED_FALSE_ALARM_LIMITS = (0.005, 0.3)
UNMEASURED_FALSE_ALARM_FLOOR = 0.003
MIN_HIGH_GROUND_PX = 1000
MAX_HIGH_GROUND_FALSE_ALARM = 0.15  # a source that fires this often on ground that cannot flood is not used

S1_DETECTION_RATE = 0.35
S1_FALSE_ALARM_RATE = 0.03
S2_DETECTION_RATES = (0.7, 0.3)  # vegetated, bare ground
S2_FALSE_ALARM_RATE = 0.02
VEGETATED_NDVI = 0.3


@dataclass(frozen=True)
class Evidence:
    """Detections, observed pixels and initial detection and false-alarm rates for one source."""

    name: str
    detected: np.ndarray
    observed: np.ndarray
    stratum: np.ndarray
    detection_priors: tuple[float, ...]
    false_alarm_prior: float


@dataclass
class RateFit:
    """Detection and false-alarm rates fitted to the pixels draining to each reach."""

    name: str
    detected: np.ndarray
    observed: np.ndarray
    stratum: np.ndarray
    detection_priors: list[float]
    detection_rates: list[float]
    false_alarm_start: float
    false_alarm_rate: float
    false_alarm_floor: float
    high_ground_rate: float | None


def follow_pointers(pointer: np.ndarray) -> np.ndarray:
    """Map every cell to the end of its pointer chain (a self-pointing cell)."""
    for _ in range(MAX_POINTER_JUMPS):
        jumped = pointer[pointer]
        if np.array_equal(jumped, pointer):
            return pointer
        pointer = jumped
    return pointer


def d8_downstream(fdir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat index of each cell's D8 neighbour downstream (the cell itself at sinks and edges) and the step length."""
    height, width = fdir.shape
    codes = np.nan_to_num(np.asarray(fdir, 'float64'), nan=0).astype(np.int64).ravel()
    cell = np.arange(height * width, dtype=np.int64)
    row, col = np.divmod(cell, width)
    downstream = cell.copy()
    step_m: np.ndarray = np.zeros(cell.size, 'float32')
    for code, (d_row, d_col) in D8_OFFSETS.items():
        index = np.flatnonzero(codes == code)
        next_row = row[index] + d_row
        next_col = col[index] + d_col
        inside = (next_row >= 0) & (next_row < height) & (next_col >= 0) & (next_col < width)
        downstream[index] = np.where(inside, next_row * width + next_col, index)
        step_m[index] = np.where(inside, DEM_CELL_M * np.hypot(d_row, d_col), 0.0)
    return downstream, step_m


def distance_to_outlet(step_m: np.ndarray, next_channel: np.ndarray) -> np.ndarray:
    """Sum the steps along each channel path to its outlet by pointer jumping."""
    dist_m = step_m
    pointer = next_channel
    for _ in range(MAX_POINTER_JUMPS):
        dist_m = dist_m + dist_m[pointer]
        jumped = pointer[pointer]
        if np.array_equal(jumped, pointer):
            break
        pointer = jumped
    return dist_m


def build_network(fdir: np.ndarray, acc_log: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (reach each cell drains to, -1 for none; downstream reach, -1 at outlets; reach distance to outlet)."""
    downstream, step_m = d8_downstream(fdir)
    cell: np.ndarray = np.arange(downstream.size, dtype=np.int64)
    acc = np.asarray(acc_log, 'float64').ravel()
    is_channel = np.isfinite(acc) & (acc > np.log10(CHANNEL_MIN_CELLS + 1.0))
    pointer = downstream.copy()
    pointer[is_channel] = cell[is_channel]
    first_channel = follow_pointers(pointer)  # a dead end where no channel lies downstream
    drains = is_channel[first_channel]

    channel_cell = cell[is_channel]
    own = np.arange(channel_cell.size)
    channel_index: np.ndarray = np.full(cell.size, -1, np.int64)
    channel_index[channel_cell] = own
    next_channel = channel_index[downstream[channel_cell]]
    is_outlet = (next_channel < 0) | (next_channel == own)
    next_channel = np.where(is_outlet, own, next_channel)
    dist_m = distance_to_outlet(np.where(is_outlet, 0.0, step_m[channel_cell]).astype('float64'), next_channel)

    # A reach is the run of channel cells in one REACH_LENGTH_M distance bin; its anchor is the run's last cell.
    distance_bin = np.floor(dist_m / REACH_LENGTH_M).astype(np.int64)
    anchor = follow_pointers(np.where((distance_bin != distance_bin[next_channel]) | is_outlet, own, next_channel))
    anchors, channel_reach = np.unique(anchor, return_inverse=True)
    down = np.where(is_outlet[anchors], -1, channel_reach[next_channel[anchors]]).astype(np.int64)
    reach: np.ndarray = np.full(cell.size, -1, np.int32)
    reach[drains] = channel_reach[channel_index[first_channel[drains]]].astype(np.int32)
    return reach.reshape(fdir.shape), down, dist_m[anchors]


def hand_bin(hand: np.ndarray) -> np.ndarray:
    """Index of the highest stage level at or below each pixel's HAND; -1 where HAND is missing or negative."""
    hand = np.asarray(hand, 'float32')
    level = np.searchsorted(STAGE_LEVELS_M, hand, side='right') - 1
    return np.where(np.isfinite(hand) & (hand >= 0), level, -1).astype(np.int16)


def cumulative_below(per_level: np.ndarray) -> np.ndarray:
    """Sum of `per_level` over the levels below each level; stage k floods the HAND bins below k."""
    total = np.zeros_like(per_level)
    total[:, 1:] = np.cumsum(per_level, 1)[:, :-1]
    return total


def map_stage(log_likelihood: np.ndarray, down: np.ndarray, dist_m: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Most probable stage level per active reach, by max-product dynamic programming down the river tree."""
    # Adjacent reaches pay a penalty for different flood levels.
    n_reach, n_levels = log_likelihood.shape
    total = log_likelihood / EVIDENCE_TEMPERATURE
    total[:, 1:] -= FLOOD_COST
    levels = np.arange(n_levels)
    step_cost = -STAGE_STEP_COST * np.abs(levels[:, None] - levels[None, :])
    best_level_given_down = np.zeros((n_reach, n_levels), np.int8)
    order = [r for r in np.argsort(-dist_m, kind='stable') if active[r]]  # upstream reaches first
    for r in order:
        d = down[r]
        if d >= 0 and active[d]:
            joint = total[r][:, None] + step_cost
            best_level_given_down[r] = np.argmax(joint, 0)
            total[d] += joint.max(0)
    stage = np.zeros(n_reach, np.int64)
    for r in reversed(order):
        d = down[r]
        stage[r] = best_level_given_down[r][stage[d]] if (d >= 0 and active[d]) else int(np.argmax(total[r]))
    return stage


def start_fit(evidence: Evidence, pixels: np.ndarray, high_ground: np.ndarray) -> RateFit:
    """Restrict `evidence` to `pixels`; measure the false-alarm rate on high ground where enough of it was observed."""
    observed = np.asarray(evidence.observed).ravel()[pixels]
    detected = np.asarray(evidence.detected).ravel()[pixels] & observed
    seen_high = observed & high_ground
    high_ground_rate = float(detected[seen_high].mean()) if seen_high.sum() > MIN_HIGH_GROUND_PX else None
    if high_ground_rate is None:
        false_alarm = float(evidence.false_alarm_prior)
        floor = UNMEASURED_FALSE_ALARM_FLOOR
    else:
        false_alarm = float(np.clip(high_ground_rate, *MEASURED_FALSE_ALARM_LIMITS))
        floor = false_alarm
    priors = [float(v) for v in evidence.detection_priors]
    return RateFit(
        name=evidence.name,
        detected=detected,
        observed=observed,
        stratum=np.asarray(evidence.stratum).ravel()[pixels].astype(np.int64),
        detection_priors=priors,
        detection_rates=list(priors),
        false_alarm_start=false_alarm,
        false_alarm_rate=false_alarm,
        false_alarm_floor=floor,
        high_ground_rate=high_ground_rate,
    )


def pixel_log_likelihood(fit: RateFit) -> np.ndarray:
    """Log-likelihood ratio flooded against dry per pixel; 0 where the source saw nothing."""
    detection = np.asarray(fit.detection_rates)[fit.stratum]
    ratio = np.where(
        fit.detected,
        np.log(detection / fit.false_alarm_rate),
        np.log((1 - detection) / (1 - fit.false_alarm_rate)),
    )
    return np.where(fit.observed, ratio, 0.0)


def update_rates(fit: RateFit, flooded: np.ndarray, valley: np.ndarray) -> None:
    """Re-estimate the rates from the flooded pixels, shrunk towards the start values by PRIOR_WEIGHT_PX pixels."""
    seen = fit.observed & valley
    for j, prior in enumerate(fit.detection_priors):
        in_flood = seen & flooded & (fit.stratum == j)
        low = max(DETECTION_RATE_LIMITS[0], prior)  # Lower fitted detection rates would expand the flood extent.
        rate = (fit.detected[in_flood].sum() + PRIOR_WEIGHT_PX * prior) / (in_flood.sum() + PRIOR_WEIGHT_PX)
        fit.detection_rates[j] = float(np.clip(rate, low, max(low, DETECTION_RATE_LIMITS[1])))
    dry = seen & ~flooded
    rate = (fit.detected[dry].sum() + PRIOR_WEIGHT_PX * fit.false_alarm_start) / (dry.sum() + PRIOR_WEIGHT_PX)
    fit.false_alarm_rate = float(np.clip(rate, fit.false_alarm_floor, MEASURED_FALSE_ALARM_LIMITS[1]))


def solve(
    reach: np.ndarray, hand: np.ndarray, down: np.ndarray, dist_m: np.ndarray, sources: list[Evidence]
) -> tuple[np.ndarray, dict[str, dict[str, Any]], dict[str, float]]:
    """Fit the stage per reach and the rates of each source by hard EM; return (stage_m raster, rates, stats)."""
    n_levels = STAGE_LEVELS_M.size
    n_reach = down.size
    bins = hand_bin(hand)
    pixels = np.flatnonzero(((reach >= 0) & (bins >= 0)).ravel())
    pixel_reach = reach.ravel()[pixels].astype(np.int64)
    pixel_bin = bins.ravel()[pixels].astype(np.int64)
    reach_level = pixel_reach * n_levels + pixel_bin
    active = np.bincount(pixel_reach, minlength=n_reach) > 0
    high_ground = pixel_bin == n_levels - 1  # HAND at or above the top level cannot flood
    valley = pixel_bin < n_levels - 1
    fits = [start_fit(source, pixels, high_ground) for source in sources]

    def most_probable_stage() -> np.ndarray:
        per_level = np.zeros((n_reach, n_levels))
        for fit in fits:
            weights = pixel_log_likelihood(fit)
            per_level += np.bincount(reach_level, weights=weights, minlength=n_reach * n_levels).reshape(
                n_reach, n_levels
            )
        return map_stage(cumulative_below(per_level), down, dist_m, active)

    for _ in range(EM_ITERATIONS):
        flooded = pixel_bin < most_probable_stage()[pixel_reach]
        for fit in fits:
            update_rates(fit, flooded, valley)
    stage = most_probable_stage()

    reach_stage_m = STAGE_LEVELS_M[stage]
    stage_m = np.zeros(reach.shape, 'float32')
    in_reach = reach >= 0
    stage_m[in_reach] = reach_stage_m[reach[in_reach]]
    rates: dict[str, dict[str, Any]] = {
        fit.name: dict(
            pd=[round(v, 3) for v in fit.detection_rates],
            pfa=round(fit.false_alarm_rate, 4),
            pfa_high_ground=round(fit.high_ground_rate, 4) if fit.high_ground_rate is not None else None,
        )
        for fit in fits
    }
    is_flooded = stage > 0
    stats = dict(
        n_reaches=int(active.sum()),
        n_flooded=int((is_flooded & active).sum()),
        max_stage_m=float(reach_stage_m.max()) if n_reach else 0.0,
        mean_stage_m=float(reach_stage_m[is_flooded].mean()) if is_flooded.any() else 0.0,
    )
    return stage_m, rates, stats


def evidence_sources(
    layers: Mapping[str, np.ndarray], masks: Mapping[str, np.ndarray], s1_observed: np.ndarray, s2_observed: np.ndarray
) -> list[Evidence]:
    shape = layers['hand'].shape
    sources = [
        Evidence('s1', masks['s1'], s1_observed, np.zeros(shape, np.int8), (S1_DETECTION_RATE,), S1_FALSE_ALARM_RATE)
    ]
    if s2_observed.any():
        if 'pre_ndvi' in layers:
            vegetated = np.nan_to_num(layers['pre_ndvi'], nan=0.0) >= VEGETATED_NDVI
        else:
            vegetated = np.ones(shape, bool)
        bare = (~vegetated).astype(np.int8)
        sources.append(Evidence('s2', masks['s2'], s2_observed, bare, S2_DETECTION_RATES, S2_FALSE_ALARM_RATE))
    return sources


def gap_fill(  # noqa: PLR0913
    layers: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    s1_observed: np.ndarray,
    fdir_30m: np.ndarray,
    acc_log_30m: np.ndarray,
    transform_30m: object,
    crs_30m: object,
    transform: object,
    crs: object,
    likely: np.ndarray,
    flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return cloud-covered valley pixels below the estimated flood level, the level raster and fit statistics."""
    hand = layers['hand']
    reach_30m, down, dist_m = build_network(fdir_30m, acc_log_30m)
    reach = np.full(hand.shape, -1, np.int32)
    reproject(
        reach_30m,
        reach,
        src_transform=transform_30m,
        src_crs=crs_30m,
        dst_transform=transform,
        dst_crs=crs,
        resampling=Resampling.nearest,
        src_nodata=-1,
        dst_nodata=-1,
    )
    # Standing water is excluded from the flood-level fit.
    stable_water = (layers['stable_wat'] == 1) if 'stable_wat' in layers else np.zeros(hand.shape, bool)
    s2_seen = (layers['s2p'] != 255) | (layers['s2e'] != 255)
    sources = evidence_sources(layers, masks, s1_observed & ~stable_water, s2_seen & ~stable_water)
    stage_m, rates, stats = solve(reach, hand, down, dist_m, sources)
    info: dict[str, object] = {'fit': rates, **stats}
    unreliable = [
        name
        for name, rate in rates.items()
        if rate['pfa_high_ground'] is not None and rate['pfa_high_ground'] > MAX_HIGH_GROUND_FALSE_ALARM
    ]
    if unreliable:
        info['unreliable_sources'] = unreliable
        sources = [source for source in sources if source.name not in unreliable]
        if not sources:
            info.update(n_flooded=0, max_stage_m=0.0, mean_stage_m=0.0, filled_km2=0.0)
            return np.zeros(hand.shape, bool), np.zeros(hand.shape, 'float32'), info
        stage_m, refit_rates, stats = solve(reach, hand, down, dist_m, sources)
        info.update(fit=dict(refit_rates, **{name: rates[name] for name in unreliable}), **stats)
    staged = (stage_m > 0) & np.isfinite(hand) & (hand < stage_m) & ~wide_water_mask(stable_water)
    fill = staged & ~flat & ~s2_seen & ~likely
    info['filled_km2'] = area_km2(fill)
    return fill, stage_m, info
