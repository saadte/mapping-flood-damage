"""Score flood maps of Trishuli 2026 against the Copernicus EMS EMSR927 grading maps (© European Union)."""

import glob

import geopandas as gpd
import numpy as np
from rasterio import windows

import reference_scoring

EMSR927_DIR = f'{reference_scoring.REFERENCE_DIR}/emsr927/raw'
PRODUCTS = {
    'AOI01': 'EMSR927_AOI01_GRA_PRODUCT_v1',
    'AOI02': 'EMSR927_AOI02_GRA_PRODUCT_v2',
    'AOI03': 'EMSR927_AOI03_GRA_MONIT01_v1',
    'AOI05': 'EMSR927_AOI05_GRA_MONIT01_v2',
}
LAYERS = [
    'areaOfInterestA',
    'observedEventA',
    'notAnalysedA',
    'builtUpP',
    'builtUpA',
    'transportationL',
    'transportationP',
]


def load_ems(crs=reference_scoring.TRISHULI_CRS):
    """EMS layers by kind for each area of interest, without empty geometries."""
    ref = {}
    for aoi, product in PRODUCTS.items():
        layers = {}
        for kind in LAYERS:
            paths = glob.glob(f'{EMSR927_DIR}/{product}/*_{kind}_*.json')
            if paths:
                layer = gpd.read_file(paths[0])
                layer = layer[layer.geometry.notna() & ~layer.geometry.is_empty]
                layers[kind] = layer.to_crs(crs)
        ref[aoi] = layers
    return ref


def aoi_window(ref_aoi, transform, width, height):
    """Grid window of the area of interest, clipped to the grid."""
    area = ref_aoi['areaOfInterestA'].geometry.union_all()
    window = windows.from_bounds(*area.bounds, transform).round_offsets().round_lengths()
    return window.intersection(windows.Window(0, 0, width, height))


def masks_for(ref_aoi, window, transform):
    """(analysed area, observed flood inside it) on the window."""
    window_transform = windows.transform(window, transform)
    shape = (int(window.height), int(window.width))
    analysed = reference_scoring.rasterize_mask(ref_aoi['areaOfInterestA'], window_transform, shape)
    not_analysed = ref_aoi.get('notAnalysedA')
    if not_analysed is not None and len(not_analysed):
        analysed &= ~reference_scoring.rasterize_mask(not_analysed, window_transform, shape)
    gt = reference_scoring.rasterize_mask(ref_aoi['observedEventA'], window_transform, shape) & analysed
    return analysed, gt


def score(pred, gt, valid):
    """Pixel counts, IoU, precision, recall and F1 of pred against gt over the valid pixels; areas for 10 m pixels."""
    pred = pred & valid
    gt = gt & valid
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        iou=tp / max(tp + fp + fn, 1),
        prec=tp / max(tp + fp, 1),
        rec=tp / max(tp + fn, 1),
        f1=2 * tp / max(2 * tp + fp + fn, 1),
        gt_ha=gt.sum() / 100,
        pred_ha=pred.sum() / 100,
    )


def pooled(rows):
    """Scores from the pixel counts summed over several score() results."""
    tp = sum(r['tp'] for r in rows)
    fp = sum(r['fp'] for r in rows)
    fn = sum(r['fn'] for r in rows)
    return dict(
        iou=tp / max(tp + fp + fn, 1),
        prec=tp / max(tp + fp, 1),
        rec=tp / max(tp + fn, 1),
        f1=2 * tp / max(2 * tp + fp + fn, 1),
    )


def evaluate_raster(pred, transform, ref=None, observable=None):
    """Scores of a boolean map per area of interest and pooled; with observable, also inside the visible pixels."""
    if ref is None:
        ref = load_ems()
    height, width = pred.shape
    out = {}
    rows_all = []
    rows_observable = []
    for aoi, layers in ref.items():
        window = aoi_window(layers, transform, width, height)
        analysed, gt = masks_for(layers, window, transform)
        rows = slice(int(window.row_off), int(window.row_off + window.height))
        cols = slice(int(window.col_off), int(window.col_off + window.width))
        aoi_pred = pred[rows, cols]
        score_all = score(aoi_pred, gt, analysed)
        rows_all.append(score_all)
        record = dict(all=score_all)
        if observable is not None:
            seen = observable[rows, cols] & analysed
            score_seen = score(aoi_pred, gt, seen)
            rows_observable.append(score_seen)
            record['observable'] = score_seen
            record['observable_frac_of_gt'] = float((gt & seen).sum() / max(gt.sum(), 1))
        out[aoi] = record
    out['pooled_all'] = pooled(rows_all)
    out['mean_f1_all'] = float(np.mean([r['f1'] for r in rows_all]))
    if rows_observable:
        out['pooled_observable'] = pooled(rows_observable)
    return out
