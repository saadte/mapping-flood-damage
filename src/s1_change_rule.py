"""Sentinel-1 change rule with thresholds calibrated on each track's own pre-event dates."""

import warnings

import numpy as np
from scipy import ndimage as ndi

MAD_TO_SIGMA = 1.4826
LEE_LOOKS = 4.4  # equivalent number of looks of a GRD scene, sets the speckle variance of the Lee filter
NULL_FLOOR_DB = 0.5  # Minimum change required before a pixel is flagged.
SPREAD_FLOOR_DB = 0.6  # keeps temporally stable pixels from turning tiny changes into large z-scores
Z_MIN = 2.0  # a change must also exceed this many pre-date spreads


def to_db(intensity):
    """Linear intensity in dB; values at or below 1e-6 become NaN."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10 * np.log10(np.where(intensity > 1e-6, intensity, np.nan))


def boxcar_db(intensity, window_px=5):
    """Mean of linear intensity over the finite pixels of a square window, in dB."""
    values = np.nan_to_num(intensity, nan=0.0)
    weights = np.isfinite(intensity).astype('float32')
    total = ndi.uniform_filter(values, window_px)
    count = ndi.uniform_filter(weights, window_px)
    with np.errstate(invalid='ignore', divide='ignore'):
        return to_db(total / count)


def lee_db(intensity, window_px=7):
    """Lee speckle filter of linear intensity, in dB."""
    values = np.nan_to_num(intensity, nan=0.0)
    weights = np.isfinite(intensity).astype('float32')
    count = np.maximum(ndi.uniform_filter(weights, window_px), 1e-6)
    mean = ndi.uniform_filter(values, window_px) / count
    mean_square = ndi.uniform_filter(values * values, window_px) / count
    variance = np.maximum(mean_square - mean * mean, 0)
    speckle_cv2 = 1.0 / LEE_LOOKS
    local_cv2 = variance / np.maximum(mean * mean, 1e-12)
    weight = np.clip(1 - speckle_cv2 / np.maximum(local_cv2, 1e-12), 0, 1)
    filtered = mean + weight * (values - mean)
    filtered[weights == 0] = np.nan
    return to_db(np.maximum(filtered, 1e-6))


def stack_nanmedian(stack):
    """np.nanmedian(stack, 0), with the faster np.median on pixels that are finite on every date."""
    finite = np.isfinite(stack)
    any_finite = finite.any(0)
    out = np.median(np.where(finite, stack, 0), 0).astype(stack.dtype)
    partial = ~finite.all(0) & any_finite
    out[~any_finite] = np.nan
    if partial.any():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            out[partial] = np.nanmedian(stack[:, partial], 0)
    return out


def median_and_spread(dates_db):
    """Per-pixel median over dates and the MAD scaled to a standard deviation."""
    stack = np.stack(dates_db)
    median = stack_nanmedian(stack)
    return median, stack_nanmedian(np.abs(stack - median)) * MAD_TO_SIGMA


def null_thresholds_db(pre_db, valley_mask, far):
    """Darkening and brightening thresholds at false-alarm rate far, from the last pre date against the earlier ones."""
    reference = stack_nanmedian(np.stack(pre_db[:-1])) if len(pre_db) >= 3 else pre_db[0]
    null_change = pre_db[-1] - reference
    values = null_change[valley_mask & np.isfinite(null_change)]
    low, high = np.nanquantile(values, far), np.nanquantile(values, 1 - far)
    return float(min(low, -NULL_FLOOR_DB)), float(max(high, NULL_FLOOR_DB))


def detect_track(pre, post1, valley_mask, post2=None, far=0.005):
    """Darkening (dec) and brightening (inc) masks of one track; pre holds chronological dB dates per polarisation."""
    out = {}
    masks = {'dec': [], 'inc': [], 'dec_persist': [], 'inc_persist': []}
    for pol in pre:
        median_db, spread_db = median_and_spread(pre[pol])
        spread_db = np.maximum(spread_db, SPREAD_FLOOR_DB)
        low_db, high_db = null_thresholds_db(pre[pol], valley_mask, far)
        change_db = post1[pol] - median_db
        z = change_db / spread_db
        dec = (change_db <= low_db) & (z <= -Z_MIN)
        inc = (change_db >= high_db) & (z >= Z_MIN)
        masks['dec'].append(dec)
        masks['inc'].append(inc)
        if post2 is None:
            masks['dec_persist'].append(dec)
            masks['inc_persist'].append(inc)
        else:  # persistence: the second post date shows at least half the threshold change
            second_change_db = post2[pol] - median_db
            masks['dec_persist'].append(dec & (second_change_db <= low_db / 2))
            masks['inc_persist'].append(inc & (second_change_db >= high_db / 2))
        out[f'{pol}_thr'] = (low_db, high_db)
        out[f'{pol}_z1'] = z
    out.update({name: np.logical_or.reduce(pol_masks) for name, pol_masks in masks.items()})
    return out
