"""Score boolean flood maps against reference polygons inside the reference analysis extent."""

import os

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy import ndimage

REFERENCE_DIR = os.environ.get('TRI_REFERENCE_DIR', '/content/tri/validation_only')
UNOSAT_DIR = f'{REFERENCE_DIR}/unosat'
TRISHULI_CRS = 'EPSG:32645'
# UNOSAT FL20260826NPL layers of Trishuli 2026
UNOSAT_FILES = dict(
    extent='UNOSAT_Multisensors_20260826_20260828_AnalysisExtent.gpkg',
    gt='UNOSAT_Multisensor_20260826_20260828_FloodExtent.gpkg',
    buildings='UNOSAT_Multisensor_20260826_20260828_AffectedBuildings.gpkg',
)


def load_unosat(crs=TRISHULI_CRS):
    return {name: gpd.read_file(f'{UNOSAT_DIR}/{file}').to_crs(crs) for name, file in UNOSAT_FILES.items()}


def rasterize_mask(layer, transform, shape):
    """Pixels whose centre lies inside a geometry of a GeoDataFrame or GeoSeries."""
    shapes = [(g, 1) for g in layer.geometry if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(shape, dtype=bool)
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0).astype(bool)


def rasterize_ref(ref, transform, shape):
    """(extent mask, reference flood mask inside the extent) on a grid."""
    extent = rasterize_mask(ref['extent'], transform, shape)
    return extent, rasterize_mask(ref['gt'], transform, shape) & extent


def score(pred, gt, valid):
    """IoU, precision, recall and F1 of pred against gt over the valid pixels; areas for 10 m pixels."""
    pred = pred & valid
    gt = gt & valid
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return dict(
        iou=tp / max(tp + fp + fn, 1),
        prec=tp / max(tp + fp, 1),
        rec=tp / max(tp + fn, 1),
        f1=2 * tp / max(2 * tp + fp + fn, 1),
        gt_km2=gt.sum() / 1e4,
        pred_km2=pred.sum() / 1e4,
    )


def tolerant_score(pred, gt, valid, tolerance_px=2):
    """Precision, recall and F1 that accept a boundary offset of up to tolerance_px between map and reference."""
    # the references trace boundaries on very-high-resolution images, our maps on a 10 m grid
    pred = pred & valid
    gt = gt & valid
    gt_dilated = ndimage.binary_dilation(gt, iterations=tolerance_px)
    pred_dilated = ndimage.binary_dilation(pred, iterations=tolerance_px)
    precision = (pred & gt_dilated).sum() / max(pred.sum(), 1)
    recall = (gt & pred_dilated).sum() / max(gt.sum(), 1)
    return dict(
        prec_tol=float(precision),
        rec_tol=float(recall),
        f1_tol=float(2 * precision * recall / max(precision + recall, 1e-9)),
    )
