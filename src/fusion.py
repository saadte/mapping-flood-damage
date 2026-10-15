"""Combine radar change, optical change and water predictions into flood masks."""

from collections.abc import Mapping

import numpy as np
from scipy import ndimage as ndi

PIXELS_PER_KM2 = 1e4  # 10 m analysis grid
DEM_CELL_M2 = 900.0  # flow accumulation counts cells of the 30 m DEM
MIN_COMPONENT_PX = 10

FLAT_SLOPE_DEG = 6.0
FLAT_WINDOW_PX = 31
FLAT_BUFFER_PX = 3

# Maximum height above drainage for radar detections and early new water.
LOW_GROUND_HAND_M = 40.0
VALLEY_HAND_M = 80.0  # optical change counts below this height
FLAT_MAX_HAND_M = 1e6  # Flat ground requires finite HAND but no upper height limit.

DL_THRESHOLD = 0.7  # radar water model flood probability on flat ground
DL_STEEP_THRESHOLD = 0.8
DL_WEAK_THRESHOLD = 0.3
BARE_NDVI = 0.3  # Bare ground uses calibrated change because vegetation loss cannot be measured there.

AGREEMENT_RADIUS_PX = 2
SUPPORT_RADIUS_PX = 3
CHANNEL_BUFFER_PX = 5
RIVER_CATCHMENT_KM2 = 20
FLAT_CATCHMENT_KM2 = 5

PLAIN_MAX_HAND_M = 10.0
PLAIN_MIN_AREA_KM2 = 100.0
PLAIN_AI_THRESHOLD = 0.5

WIDE_WATER_RADIUS_PX = 8
CHANNEL_FILL_PX = 8

# These parameter names are also used by the road-break evidence calculation.
PARAMS = dict(
    variant='terrain2_dlsup',
    conn_km2=RIVER_CATCHMENT_KM2,
    dl_thr=DL_THRESHOLD,
    dl_hand=int(LOW_GROUND_HAND_M),
    flat_deg=FLAT_SLOPE_DEG,
    grow_k=0,
    grow_hand=80.0,
    grow_cand='any',
    s2_mode='rule_barecal',
    dl_far=0.0,
    s1_mode='joint',
    s2_time='both',
    flat_hand=FLAT_MAX_HAND_M,
    hi_mode='s1s2+two_conn',
    early_water='h40',
    dl_thr_mtn=DL_STEEP_THRESHOLD,
    mmu_px=0,
    wide_water_r=WIDE_WATER_RADIUS_PX,
    s1_hand=LOW_GROUND_HAND_M,
    s2_hand=VALLEY_HAND_M,
    agree_px=AGREEMENT_RADIUS_PX,
    flat_conn_km2=FLAT_CATCHMENT_KM2,
    flat_conn_dil=CHANNEL_BUFFER_PX,
    chan_fill=CHANNEL_FILL_PX,
    possible_conn=True,
    plain_km2=PLAIN_MIN_AREA_KM2,
    plain_hand=PLAIN_MAX_HAND_M,
    plain_dl=True,
    plain_s2=True,
    plain_ai=True,
)


def area_km2(mask: np.ndarray) -> float:
    return float(mask.sum() / PIXELS_PER_KM2)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Close one-pixel gaps, then drop components smaller than MIN_COMPONENT_PX."""
    labels, _ = ndi.label(ndi.binary_closing(mask, iterations=1))
    keep = np.bincount(labels.ravel()) >= MIN_COMPONENT_PX
    keep[0] = False
    return keep[labels]


def components_touching(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Connected components of `mask` that overlap `seeds`."""
    labels, _ = ndi.label(mask)
    return np.isin(labels, np.unique(labels[seeds & (labels > 0)]))


def channel_buffer(acc_log: np.ndarray, catchment_km2: float) -> np.ndarray:
    """Pixels within CHANNEL_BUFFER_PX of a channel draining at least `catchment_km2`; acc_log = log10(cells + 1)."""
    channel = acc_log >= np.log10(catchment_km2 * 1e6 / DEM_CELL_M2 + 1)
    return ndi.binary_dilation(channel, iterations=CHANNEL_BUFFER_PX)


def flat_mask(slope_deg: np.ndarray, max_slope_deg: float = FLAT_SLOPE_DEG) -> np.ndarray:
    """Pixels whose mean slope over FLAT_WINDOW_PX is below `max_slope_deg`; missing slope counts as steep."""
    slope_filled = np.nan_to_num(slope_deg, nan=90.0).astype('float32')
    return ndi.uniform_filter(slope_filled, FLAT_WINDOW_PX) < max_slope_deg


def plain_mask(flat: np.ndarray, hand: np.ndarray) -> np.ndarray:
    """Connected flat ground below PLAIN_MAX_HAND_M that covers at least PLAIN_MIN_AREA_KM2."""
    labels, _ = ndi.label(flat & (hand < PLAIN_MAX_HAND_M))
    component_km2 = np.bincount(labels.ravel()) / PIXELS_PER_KM2
    component_km2[0] = 0
    return (component_km2 >= PLAIN_MIN_AREA_KM2)[labels]


def wide_water_mask(stable_water: np.ndarray) -> np.ndarray:
    """Stable water wide enough to hold a disc of radius WIDE_WATER_RADIUS_PX, with a one-pixel rim."""
    radius = WIDE_WATER_RADIUS_PX
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disc = (x * x + y * y) <= radius * radius
    core = ndi.binary_opening(stable_water, structure=disc)
    return ndi.binary_dilation(core, iterations=1) & stable_water


def optical_change(layers: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return (persistent change, change in the first clear image) from the Sentinel-2 evidence."""
    later = layers['s2p'] == 1
    early = layers['s2e'] == 1
    if 's2cal_p' in layers:
        calibrated_later = layers['s2cal_p'] == 1
        calibrated_early = layers['s2cal_e'] == 1
        if 'pre_ndvi' in layers:
            bare = np.nan_to_num(layers['pre_ndvi'], nan=1.0) < BARE_NDVI
            calibrated_later = bare & calibrated_later
            calibrated_early = bare & calibrated_early
        later = later | calibrated_later
        early = early | calibrated_early
    early_seen = layers['s2e'] != 255
    later_seen = layers['s2p'] != 255
    persistent = np.where(early_seen & later_seen, later & early, np.where(early_seen, early, later))
    return persistent, early


def two_source_agreement(s1: np.ndarray, s2: np.ndarray, dl: np.ndarray) -> np.ndarray:
    """Union of the three masks where at least two of them lie within AGREEMENT_RADIUS_PX."""
    near_s1, near_s2, near_dl = (
        ndi.binary_dilation(mask, iterations=AGREEMENT_RADIUS_PX).astype('uint8') for mask in (s1, s2, dl)
    )
    return (s1 | s2 | dl) & (near_s1 + near_s2 + near_dl >= 2)


def plain_ai_water(layers: Mapping[str, np.ndarray], dl: np.ndarray) -> np.ndarray:
    """Radar water model above PLAIN_AI_THRESHOLD or optical water model new open water."""
    water = clean_mask(dl > PLAIN_AI_THRESHOLD)
    if 'ai_any' in layers:
        water = water | clean_mask(layers['ai_any'])
    return water


def high_ground_agreement(
    s1: np.ndarray, s2: np.ndarray, dl: np.ndarray, low_ground: np.ndarray, likely: np.ndarray
) -> np.ndarray:
    """Keep radar with optical change above the HAND limits, or two masks connected to the likely extent."""
    high_ground = ~low_ground
    s1_and_s2 = clean_mask(s1 & s2 & high_ground & ~likely)
    two_sources = clean_mask((((s1.astype('uint8') + s2 + (dl > DL_STEEP_THRESHOLD)) >= 2) & high_ground) & ~likely)
    connected = components_touching(two_sources | likely, likely) & two_sources
    return s1_and_s2 | connected


def channel_fill(channel_water: np.ndarray, likely: np.ndarray) -> np.ndarray:
    """Channel water within CHANNEL_FILL_PX of `likely` and connected to it."""
    near_likely = channel_water & ndi.binary_dilation(likely, iterations=CHANNEL_FILL_PX)
    return components_touching(near_likely | likely, likely) & near_likely & ~likely


def fuse(layers: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Return (likely, possible, source_bits, info)."""
    hand = layers['hand']
    acc_log = layers['acc']
    dl = layers['dl']
    s1 = layers['s1j'] if 's1j' in layers else layers['s1']
    s2, s2_early = optical_change(layers)
    low_ground = hand < LOW_GROUND_HAND_M
    valley = hand < VALLEY_HAND_M
    near_river = channel_buffer(acc_log, RIVER_CATCHMENT_KM2)
    s1_strong = s1 & low_ground
    s2_strong = s2 & valley
    dl_strong = (dl > DL_STEEP_THRESHOLD) & low_ground
    strong = clean_mask(s1_strong | s2_strong | dl_strong)
    info: dict[str, object] = {}

    # Flat ground: two sources must agree, and the evidence must reach a drainage line unless it lies on a plain, where
    # the 30 m DEM misplaces the rivers.
    flat = flat_mask(layers['slope'])
    info['flat_frac'] = float(flat.mean())
    plain = plain_mask(flat, hand)
    has_plain = bool(plain.any())
    if has_plain:
        info['plain_area_km2'] = area_km2(plain)
    has_hand = hand < FLAT_MAX_HAND_M
    dl_flat = (dl > DL_THRESHOLD) & has_hand
    flat_likely = clean_mask(two_source_agreement(s1 & has_hand, s2 & has_hand, dl_flat))
    flat_likely = flat_likely & ndi.binary_dilation(flat, iterations=FLAT_BUFFER_PX)
    plausible = components_touching(flat_likely, channel_buffer(acc_log, FLAT_CATCHMENT_KM2) | plain)
    info['flat_implausible_removed_km2'] = area_km2(flat_likely & ~plausible)
    flat_likely = plausible

    # Water already present in the pre-event images can be missed by change detection on plains.
    if has_plain:
        info['plain_dl_added_km2'] = area_km2(clean_mask(dl_flat) & plain & ~flat_likely)  # reported only
        ai_water = plain_ai_water(layers, dl)
        info['plain_ai_km2'] = area_km2(ai_water & plain)
        flat_likely = (flat_likely & ~plain) | (ai_water & plain)

    # Mountain detections must reach a river; radar water model also requires nearby measured change.
    support = ndi.binary_dilation((s1 | s2 | s2_early) & valley, iterations=SUPPORT_RADIUS_PX)
    river_likely = components_touching(clean_mask(s1_strong | s2_strong | (dl_strong & support)), near_river) & ~plain
    likely = clean_mask(flat_likely | river_likely)

    added = high_ground_agreement(s1, s2, dl, low_ground, likely)
    info['hi_added_km2'] = area_km2(added)
    likely = likely | added

    # Water in the first clear image may have receded by the later composite.
    if 's2e_wat' in layers:
        early_water = clean_mask((layers['s2e_wat'] == 1) & low_ground & ~likely) & ~plain
        info['early_water_km2'] = area_km2(early_water)
        likely = likely | early_water
        s2 = s2 | early_water

    wide_water = np.zeros_like(likely)
    if 'stable_wat' in layers:
        stable_water = layers['stable_wat'] == 1
        wide_water = wide_water_mask(stable_water)
        info['wide_water_removed_km2'] = area_km2(likely & wide_water)
        likely = likely & ~wide_water
        fill = channel_fill(stable_water & ~wide_water, likely) & ~plain
        info['chan_fill_km2'] = area_km2(fill)
        likely = likely | fill

    weak = clean_mask(((s1 | s2 | s2_early) & valley) | ((dl > DL_WEAK_THRESHOLD) & low_ground))
    possible = components_touching(strong | weak, near_river) & ~likely & ~wide_water
    dl_detected = dl > DL_THRESHOLD
    # bit 1: radar change, bit 2: optical change, bit 4: radar water model
    source_bits = (
        s1.astype('uint8') | ((s2 | s2_early).astype('uint8') << 1) | (dl_detected.astype('uint8') << 2)
    ) * likely
    info['_masks'] = dict(s1=s1, s2=s2, s2e=s2_early, dl=dl_detected)
    return likely, possible, source_bits.astype('uint8'), info
