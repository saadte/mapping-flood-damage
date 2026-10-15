"""Two-polarisation, multi-date Sentinel-1 change test calibrated on each track's pre-event dates (s1x)."""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.special import digamma, polygamma

POLS = ('VV', 'VH')
SLOPE_EDGES_DEG = (5.0, 15.0, 30.0)
STRATUM_COUNT = 24  # 4 slope classes x 3 brightness terciles x layover/shadow flag
MIN_STRATUM_PX = 4000
CALIBRATION_HAND_MAX_M = 80
CALIBRATION_SAMPLE_PX = 3_000_000
MAD_TO_SIGMA = 1.4826
TAIL_QUANTILE = 0.99
LOG10_2 = float(np.log10(2.0))
ROW_BLOCK = 256
CGROUP_CPU_MAX = '/sys/fs/cgroup/cpu.max'


def boxcar_mean(intensity, window_px):
    """Mean of linear intensity over the finite pixels of a square window; NaN where the mean is not positive."""
    values = ndi.uniform_filter(np.nan_to_num(intensity, nan=0.0).astype('float32'), window_px)
    weights = ndi.uniform_filter(np.isfinite(intensity).astype('float32'), window_px)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = values / weights
    out[~(weights > 1e-3) | ~(out > 1e-6)] = np.nan
    return out


def power_db(intensity):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (10.0 * np.log10(intensity)).astype('float32')


def mean_intensity(dates):
    """Mean of linear intensity over dates, summed in float32 in date order."""
    out = np.array(dates[0], dtype='float32', copy=True)
    for date in dates[1:]:
        out += date
    out /= len(dates)
    return out


def db_variance(dates):
    """Sample variance over dates of the intensity in dB; None for fewer than two dates."""
    if len(dates) < 2:
        return None
    dates_db = [power_db(date) for date in dates]
    mean = dates_db[0].copy()
    for date_db in dates_db[1:]:
        mean += date_db
    mean /= len(dates)
    sum_squares = np.zeros_like(mean)
    for date_db in dates_db:
        sum_squares += (date_db - mean) ** 2
    return sum_squares / (len(dates) - 1)


def inverse_trigamma(x):
    """Solution y of trigamma(y) = x, by Newton iteration."""
    x = float(x)
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(60):
        trigamma = float(polygamma(1, y))
        step = trigamma * (1 - trigamma / x) / float(polygamma(2, y))
        y += step
        if -step / y < 1e-8:
            break
    return y


def variance_prior(s2, df):
    """Scaled inverse-chi2 prior (s0^2, d0) fitted by moments to sample variances with df degrees of freedom."""
    s2 = np.asarray(s2, 'float64')
    s2 = s2[np.isfinite(s2) & (s2 > 1e-8)]
    if s2.size < 200 or df < 1:
        return (float(np.median(s2)) if s2.size else 1.0), float('inf')
    log_s2 = np.log(s2) - digamma(df / 2.0) + np.log(df / 2.0)
    log_mean = float(log_s2.mean())
    log_excess_var = float(log_s2.var(ddof=1) - polygamma(1, df / 2.0))
    if log_excess_var > 0:
        d0 = 2.0 * inverse_trigamma(log_excess_var)
        return float(np.exp(log_mean + digamma(d0 / 2.0) - np.log(d0 / 2.0))), d0
    return float(np.exp(log_mean)), float('inf')


def central_mode(z):
    """Centre of the unchanged pixels: medians iterated over the values within 1 of the previous median."""
    # the iteration moves towards the mode, which a large one-sided changed fraction does not shift
    z = z[np.isfinite(z)]
    if z.size < 100:
        return 0.0
    centre = float(np.median(z))
    for _ in range(6):
        near = z[np.abs(z - centre) < 1.0]
        if near.size < 50:
            break
        new_centre = float(np.median(near))
        if abs(new_centre - centre) < 1e-3:
            return new_centre
        centre = new_centre
    return centre


def scaled_mad(x):
    return MAD_TO_SIGMA * float(np.median(np.abs(x - np.median(x))))


def robust_null_covariance(z_vv, z_vh):
    """MAD scales of two z-scores and their correlation from the MADs of the scaled sum and difference."""
    finite_mask = np.isfinite(z_vv) & np.isfinite(z_vh)
    a, b = z_vv[finite_mask].astype('float64'), z_vh[finite_mask].astype('float64')
    if a.size < 200:
        return 1.0, 1.0, 0.0
    scale_a, scale_b = max(scaled_mad(a), 1e-6), max(scaled_mad(b), 1e-6)
    sum_var, difference_var = scaled_mad(a / scale_a + b / scale_b) ** 2, scaled_mad(a / scale_a - b / scale_b) ** 2
    correlation = float(np.clip((sum_var - difference_var) / max(sum_var + difference_var, 1e-12), -0.95, 0.95))
    return scale_a, scale_b, correlation


def t2_statistic(z_vv, z_vh, scale_vv, scale_vh, correlation):
    """Squared Mahalanobis distance of (z_vv, z_vh) under the null covariance."""
    a = z_vv / scale_vv
    b = z_vh / scale_vh
    return ((a * a - 2 * correlation * a * b + b * b) / (1 - correlation * correlation)).astype('float32')


class NullCalibration:
    """Empirical survival function of null statistics with an exponential tail beyond the 99th percentile."""

    def __init__(self, values):
        values = np.sort(np.asarray(values, 'float64')[np.isfinite(values)])
        if values.size < 50:
            values = np.array([0.0, 1.0])
        self.values = values
        self.count = int(values.size)
        self.tail_start = float(values[int(TAIL_QUANTILE * (self.count - 1))])
        tail_count = self.count - np.searchsorted(values, self.tail_start, side='left')
        self.tail_probability = float(tail_count / self.count)
        excess = values[values > self.tail_start] - self.tail_start
        self.tail_scale = float(excess.mean()) if excess.size >= 10 else float(max(np.std(values), 1e-6))

    def nlp(self, statistic):
        """-log10 p of statistic values."""
        statistic = np.asarray(statistic, dtype='float64')
        exceeding = self.count - np.searchsorted(self.values, statistic, side='left')
        p = (exceeding + 1.0) / (self.count + 1.0)
        with np.errstate(over='ignore', invalid='ignore'):
            tail_p = self.tail_probability * np.exp(-(statistic - self.tail_start) / self.tail_scale)
            p = np.where(statistic > self.tail_start, tail_p, p)
            out = -np.log10(np.clip(p, 1e-300, 1.0))
        out[~np.isfinite(statistic)] = np.nan
        return out.astype('float32')


def stratum_parts(stratum):
    """(slope class, brightness tercile, layover/shadow flag) of a stratum id."""
    return stratum // 6, (stratum % 6) // 2, stratum % 2


def merged_key(stratum, counts):
    """Key of a stratum after merging small strata into (slope, brightness), then (slope), then all; -1 = merged."""
    slope_class, bright_class, geometry = stratum_parts(stratum)
    pair_start = slope_class * 6 + bright_class * 2
    if counts[stratum] >= MIN_STRATUM_PX:
        return slope_class, bright_class, geometry
    if counts[pair_start : pair_start + 2].sum() >= MIN_STRATUM_PX:
        return slope_class, bright_class, -1
    if counts[slope_class * 6 : slope_class * 6 + 6].sum() >= MIN_STRATUM_PX:
        return slope_class, -1, -1
    return -1, -1, -1


def key_members(key):
    """Strata that a merged key covers."""
    return np.array(
        [
            stratum
            for stratum in range(STRATUM_COUNT)
            if all(level < 0 or part == level for part, level in zip(stratum_parts(stratum), key, strict=True))
        ]
    )


class Strata:
    """Slope class x pre-brightness tercile x layover/shadow flag, with small strata merged into coarser ones."""

    def __init__(self, slope_deg, bright_db, geometry_mask, sample):
        slope_class = np.digitize(np.nan_to_num(slope_deg, nan=90.0), SLOPE_EDGES_DEG).astype('int8')
        sample_bright = bright_db.ravel()[sample]
        finite = np.isfinite(sample_bright)
        if finite.sum() > 100:
            terciles = [float(x) for x in np.quantile(sample_bright[finite], [1 / 3, 2 / 3])]
        else:
            terciles = [-1e9, 1e9]
        bright_class = np.digitize(np.nan_to_num(bright_db, nan=-999.0), terciles).astype('int8')
        stratum = (slope_class * 6 + bright_class * 2 + geometry_mask.astype('int8')).astype('int8')
        stratum[~np.isfinite(bright_db) | ~np.isfinite(slope_deg)] = -1
        sample_stratum = stratum.ravel()[sample]
        counts = np.bincount(sample_stratum[sample_stratum >= 0].astype('int64'), minlength=STRATUM_COUNT)
        keys = [merged_key(s, counts) for s in range(STRATUM_COUNT)]
        self.keys = sorted(set(keys))
        group_of_stratum = np.array([*(self.keys.index(key) for key in keys), -1], 'int16')
        self.group = group_of_stratum[stratum.astype('int16')]  # stratum -1 reads the trailing -1
        self.sample_group = self.group.ravel()[sample]
        # a merged group calibrates on all its strata, including those that kept a group of their own
        self.sample_pools = [np.isin(sample_stratum, key_members(key)) for key in self.keys]


def stratum_lut(values):
    """float32 look-up table by group; group -1 reads NaN."""
    return np.array([*values, np.nan], 'float32')


def stratum_centres(z, pools):
    return stratum_lut([central_mode(z[pool]) for pool in pools])


def moderated_variance(s2, ref_count, prior_s2, prior_df):
    """Empirical-Bayes moderated variance (Smyth 2004); the prior alone without a sample variance."""
    df = ref_count - 1
    if s2 is None or df <= 0:
        return prior_s2
    with np.errstate(invalid='ignore'):
        weight = np.where(np.isfinite(prior_df), prior_df / (prior_df + df), 1.0).astype('float32')
    return weight * prior_s2 + (1 - weight) * s2


def z_pair(post_mean, ref_mean, ref_s2, ref_count, post_count, priors, group):
    """Moderated z-scores (VV, VH) of the dB change from the reference dates to the post dates."""
    out = []
    for pol in POLS:
        change_db = power_db(post_mean[pol]) - power_db(ref_mean[pol])
        prior_s2, prior_df = priors[pol]
        s2 = moderated_variance(ref_s2[pol], ref_count, prior_s2[group], prior_df[group])
        with np.errstate(invalid='ignore', divide='ignore'):
            out.append((change_db / np.sqrt(s2 * (1.0 / post_count + 1.0 / ref_count))).astype('float32'))
    return out


def dates_mean(dates):
    return {pol: mean_intensity([date[pol] for date in dates]) for pol in POLS}


def dates_db_variance(dates):
    return {pol: db_variance([date[pol] for date in dates]) for pol in POLS}


class ChangeTestCalibration(NamedTuple):
    """Look-up tables by group of one T^2 test: z centres, null covariance and null distribution."""

    centre_vv: np.ndarray
    centre_vh: np.ndarray
    scale_vv: np.ndarray
    scale_vh: np.ndarray
    correlation: np.ndarray
    nulls: list


def null_z_pairs(sample_pre, post_count, priors, pools, sample_group):
    """Centred z-scores of every pre-only realisation: post_count consecutive pre dates against the other pre dates."""
    pairs = []
    for first in range(len(sample_pre) - post_count + 1):
        fake_post = range(first, first + post_count)
        reference = [date for j, date in enumerate(sample_pre) if j not in fake_post]
        z_vv, z_vh = z_pair(
            dates_mean([sample_pre[j] for j in fake_post]),
            dates_mean(reference),
            dates_db_variance(reference),
            len(reference),
            post_count,
            priors,
            sample_group,
        )
        pairs.append(
            (z_vv - stratum_centres(z_vv, pools)[sample_group], z_vh - stratum_centres(z_vh, pools)[sample_group])
        )
    return pairs


def calibrate_test(sample_pre, sample_post, sample_ref_s2, post_count, priors, strata):
    """Calibration of the test of the first post_count post dates on the calibration sample."""
    pools, group = strata.sample_pools, strata.sample_group
    pairs = null_z_pairs(sample_pre, post_count, priors, pools, group)
    covariances = [
        robust_null_covariance(
            np.concatenate([z_vv[pool] for z_vv, _ in pairs]), np.concatenate([z_vh[pool] for _, z_vh in pairs])
        )
        for pool in pools
    ]
    scale_vv, scale_vh, correlation = (stratum_lut([covariance[k] for covariance in covariances]) for k in range(3))
    null_t2 = [t2_statistic(z_vv, z_vh, scale_vv[group], scale_vh[group], correlation[group]) for z_vv, z_vh in pairs]
    nulls = [NullCalibration(np.concatenate([t2[pool] for t2 in null_t2])) for pool in pools]
    event_vv, event_vh = z_pair(
        dates_mean(sample_post[:post_count]),
        dates_mean(sample_pre),
        sample_ref_s2,
        len(sample_pre),
        post_count,
        priors,
        group,
    )
    return ChangeTestCalibration(
        stratum_centres(event_vv, pools), stratum_centres(event_vh, pools), scale_vv, scale_vh, correlation, nulls
    )


def calibrated_nlp(statistic, group, nulls):
    """-log10 p of a statistic under the null distribution of each pixel's group; NaN for group -1."""
    out = np.full(statistic.shape, np.nan, 'float32')
    for i, null in enumerate(nulls):
        group_mask = group == i
        out[group_mask] = null.nlp(statistic[group_mask])
    return out


def change_test_nlp(calibration, z_vv, z_vh, group):
    """-log10 p of the T^2 of centred (z_vv, z_vh)."""
    z_vv = z_vv - calibration.centre_vv[group]
    z_vh = z_vh - calibration.centre_vh[group]
    t2 = t2_statistic(
        z_vv, z_vh, calibration.scale_vv[group], calibration.scale_vh[group], calibration.correlation[group]
    )
    return calibrated_nlp(t2, group, calibration.nulls)


def track_nlp(pre, post, strata, sample, threads):
    """-log10 p maps keyed by the number of post dates tested (1: first date, 2: their mean), and the observed mask."""
    post_counts = (1, 2) if len(post) >= 2 else (1,)
    sample_pre = [{pol: date[pol].ravel()[sample] for pol in POLS} for date in pre]
    sample_post = [{pol: date[pol].ravel()[sample] for pol in POLS} for date in post]
    sample_ref_s2 = dates_db_variance(sample_pre)
    priors = {}
    for pol in POLS:
        fits = [variance_prior(sample_ref_s2[pol][pool], len(pre) - 1) for pool in strata.sample_pools]
        priors[pol] = (stratum_lut([s0 for s0, _ in fits]), stratum_lut([d0 for _, d0 in fits]))
    calibrations = {
        post_count: calibrate_test(sample_pre, sample_post, sample_ref_s2, post_count, priors, strata)
        for post_count in post_counts
    }
    shape = post[0]['VV'].shape
    nlp = {post_count: np.full(shape, np.nan, 'float32') for post_count in post_counts}
    observed_mask = np.zeros(shape, bool)

    def fill_rows(start):
        rows = slice(start, min(start + ROW_BLOCK, shape[0]))
        group = strata.group[rows]
        block_pre = [{pol: date[pol][rows] for pol in POLS} for date in pre]
        block_post = [{pol: date[pol][rows] for pol in POLS} for date in post]
        block_observed_mask = np.isfinite(block_post[0]['VV']) & np.isfinite(block_pre[-1]['VV'])
        ref_mean = dates_mean(block_pre)
        ref_s2 = dates_db_variance(block_pre)
        for post_count, calibration in calibrations.items():
            post_mean = dates_mean(block_post[:post_count])
            z_vv, z_vh = z_pair(post_mean, ref_mean, ref_s2, len(pre), post_count, priors, group)
            block_nlp = change_test_nlp(calibration, z_vv, z_vh, group)
            block_nlp[~block_observed_mask] = np.nan
            nlp[post_count][rows] = block_nlp
        observed_mask[rows] = block_observed_mask

    # every per-group quantity comes from the calibration sample, so row blocks are independent
    with ThreadPoolExecutor(threads) as executor:
        list(executor.map(fill_rows, range(0, shape[0], ROW_BLOCK)))
    return nlp, observed_mask


def bonferroni_pair(nlp_a, nlp_b):
    """-log10 of min(1, 2 min(p_a, p_b)); nlp_a alone when nlp_b is None."""
    if nlp_b is None:
        return nlp_a
    with np.errstate(invalid='ignore'):
        best = np.fmax(nlp_a, nlp_b)
    return np.maximum(best - LOG10_2, 0.0).astype('float32')


def available_cpus():
    """CPUs this process may use: the cgroup v2 quota when set, else the affinity mask, else os.cpu_count()."""
    if os.path.exists(CGROUP_CPU_MAX):
        with open(CGROUP_CPU_MAX) as f:
            quota, period = f.read().split()[:2]
        if quota != 'max':
            return max(1, int(int(quota) / int(period)))
    if hasattr(os, 'sched_getaffinity'):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def calibration_sample(hand_m):
    """Sorted flat indices of up to 3 million random low-ground pixels (finite HAND below 80 m)."""
    candidates = np.flatnonzero((np.isfinite(hand_m) & (hand_m < CALIBRATION_HAND_MAX_M)).ravel())
    if candidates.size <= CALIBRATION_SAMPLE_PX:
        return candidates
    return np.sort(np.random.default_rng(0).choice(candidates, CALIBRATION_SAMPLE_PX, replace=False))


def read_boxcar_means(paths, window_px, threads):
    """Boxcar means of bands 1 (VV) and 2 (VH) of linear sigma0 GeoTIFFs, as one dict per file."""

    def filtered_band(job):
        path, band = job
        with rasterio.open(path) as src:
            return boxcar_mean(src.read(band), window_px)

    jobs = [(path, band) for path in paths for band in (1, 2)]
    with ThreadPoolExecutor(threads) as executor:
        bands = list(executor.map(filtered_band, jobs))
    return [{'VV': bands[2 * i], 'VH': bands[2 * i + 1]} for i in range(len(paths))]


def s1x_track(pre, post, slope_deg, geometry_mask, sample, threads):
    """-log10 p of one track (Bonferroni of the first-post and post-mean tests), the observed mask and the strata."""
    strata = Strata(slope_deg, power_db(mean_intensity([date['VV'] for date in pre])), geometry_mask, sample)
    nlp, observed_mask = track_nlp(pre, post, strata, sample, threads)
    return bonferroni_pair(nlp[1], nlp.get(2)), observed_mask, dict(strata=strata.keys)
