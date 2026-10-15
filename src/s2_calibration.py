"""Sentinel-2 change detection with thresholds calibrated on the change between two pre-event composites."""

import numpy as np
import rasterio

BANDS = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
NODATA = -9999
EPS = 1e-6
HAND_MAX_M = 40.0
BARE_NDVI_MAX = 0.3  # strata: bare ground below this pre-event NDVI, vegetation above
MIN_STRATUM_PX = 2000
# Minimum feature change prevents noise from being flagged when pre-event variation is small.
NULL_FLOORS = {'mndwi': 0.08, 'ndvi': 0.08, 'bsi': 0.04, 'nbr2': 0.03, 'sam': 0.05}


def _mndwi(bands):
    return (bands['B03'] - bands['B11']) / (bands['B03'] + bands['B11'] + EPS)


def _ndvi(bands):
    return (bands['B08'] - bands['B04']) / (bands['B08'] + bands['B04'] + EPS)


def _bsi(bands):
    """Bare soil index."""
    return ((bands['B11'] + bands['B04']) - (bands['B08'] + bands['B02'])) / (
        (bands['B11'] + bands['B04']) + (bands['B08'] + bands['B02']) + EPS
    )


def _nbr2(bands):
    return (bands['B11'] - bands['B12']) / (bands['B11'] + bands['B12'] + EPS)


# index -> sign of the change that indicates flooding or debris
CHANGE_INDICES = {
    'mndwi': (_mndwi, 1),  # new water or wet surface
    'ndvi': (_ndvi, -1),  # vegetation stripped
    'bsi': (_bsi, 1),  # fresh bare sediment
    'nbr2': (_nbr2, -1),  # SWIR moisture or mineral change
}


def read_bands(path):
    """Reflectance per band from a pipeline composite (int16 reflectance x 1e4)."""
    with rasterio.open(path) as src:
        stack = src.read().astype('float32')
    stack[stack == NODATA] = np.nan
    return {band: stack[i] / 1e4 for i, band in enumerate(BANDS)}


def _spectral_angle(before, after):
    before_stack = np.stack([before[band] for band in BANDS])
    after_stack = np.stack([after[band] for band in BANDS])
    cos = (before_stack * after_stack).sum(0) / (np.sqrt((before_stack**2).sum(0) * (after_stack**2).sum(0)) + EPS)
    return np.arccos(np.clip(cos, -1, 1))


def change_features(before, after):
    """Change per feature, signed so that larger values look more like flooding or debris."""
    features = {name: sign * (index(after) - index(before)) for name, (index, sign) in CHANGE_INDICES.items()}
    features['sam'] = _spectral_angle(before, after)
    return features


def _max_scaled(features, mask, scales):
    """Per masked pixel, the largest feature value divided by its scale (NaN counts as -9)."""
    return np.max(np.stack([np.nan_to_num(features[name][mask], nan=-9) / scales[name] for name in features]), 0)


def detect(pre, post, pre_older, pre_newer, hand_m, far=0.01):
    """Change mask (1 flagged, 0 not, 255 unobserved) at false-alarm rate `far`, and the thresholds per stratum."""
    change = change_features(pre, post)
    null = change_features(pre_older, pre_newer)  # natural variability between the two halves of the pre-event dates
    pre_ndvi = _ndvi(pre)
    low_mask = hand_m < HAND_MAX_M
    observed = np.isfinite(change['sam']) & np.isfinite(pre_ndvi)
    flagged = np.zeros(hand_m.shape, 'uint8')
    info = {}
    for stratum, stratum_mask in (('bare', pre_ndvi < BARE_NDVI_MAX), ('veg', pre_ndvi >= BARE_NDVI_MAX)):
        calibration_mask = low_mask & stratum_mask & observed & np.isfinite(null['sam'])
        if calibration_mask.sum() < MIN_STRATUM_PX:
            continue
        scales = {}
        for name in change:
            null_values = null[name][calibration_mask]
            null_values = null_values[np.isfinite(null_values)]
            scales[name] = max(float(np.quantile(null_values, 1 - far)), NULL_FLOORS[name])
        # the max over features is itself calibrated on the null
        threshold = max(float(np.quantile(_max_scaled(null, calibration_mask, scales), 1 - far)), 1.0)
        target = stratum_mask & observed
        score = np.full(hand_m.shape, -9.0, 'float32')
        score[target] = _max_scaled(change, target, scales)
        detected = (score >= threshold) & target & low_mask
        flagged |= detected.astype('uint8')
        info[stratum] = dict(
            q={name: round(scale, 4) for name, scale in scales.items()},
            T=round(threshold, 3),
            frac_flagged_low=round(float(detected.sum() / max((stratum_mask & observed & low_mask).sum(), 1)), 4),
        )
    return np.where(observed, flagged, 255).astype('uint8'), info
