"""Predict open water in Sentinel-2 images with the Sen1Floods11 U-Net."""

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch

BANDS = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
S1F11_IDX = [1, 2, 3, 7, 11, 12]  # positions of BANDS in the 13-band Sen1Floods11 S2Hand chips
MEAN = np.array([0.05, 0.07, 0.07, 0.24, 0.18, 0.11], 'float32')
STD = np.array([0.05, 0.05, 0.06, 0.11, 0.09, 0.08], 'float32')
# Visible-band offsets reduce the difference between training and input reflectance.
DOS_CAP = np.array([0.12, 0.08, 0.06, 0.0, 0.0, 0.0], 'float32')
DOS_FLOOR = 0.01
INDEX_EPS = 0.02  # keeps the ratios of dark pixels (shadow, deep water) stable
TILE_PX = 512
OVERLAP_PX = 64
BATCH = 8
SIZE_MULTIPLE_PX = 32  # the U-Net halves the resolution five times


def _dark_object_offsets(bands):
    """Per-band offset from the 1st percentile of the valid pixels, on the visible bands only."""
    offsets = np.zeros(6, 'float32')
    for b in range(3):
        values = bands[b][np.isfinite(bands[b])]
        if values.size > 100:
            subsample = values[:: max(1, values.size // 200000)]
            offsets[b] = float(np.clip(np.percentile(subsample, 1.0) - DOS_FLOOR, 0.0, DOS_CAP[b]))
    return offsets


def features(bands):
    """Model input (9, H, W) float32 with no-data pixels at 0, and the valid mask, from (6, H, W) reflectance."""
    bands = np.asarray(bands, 'float32')
    valid = np.isfinite(bands).all(0)
    bands = np.clip(bands - _dark_object_offsets(bands)[:, None, None], -0.05, 1.5)
    green, red, nir, swir1 = bands[1], bands[2], bands[3], bands[4]
    mndwi = (green - swir1) / (np.abs(green) + np.abs(swir1) + INDEX_EPS)
    ndwi = (green - nir) / (np.abs(green) + np.abs(nir) + INDEX_EPS)
    ndvi = (nir - red) / (np.abs(nir) + np.abs(red) + INDEX_EPS)
    standardized = (bands - MEAN[:, None, None]) / STD[:, None, None]
    channels = np.concatenate([standardized, np.stack([mndwi, ndwi, ndvi])]).astype('float32')
    channels[:, ~valid] = 0.0
    return channels, valid


def build_model(pretrained=True):
    return smp.Unet('resnet34', encoder_weights='imagenet' if pretrained else None, in_channels=9, classes=2)


def load_model(path, device):
    model = build_model(pretrained=False)
    state = torch.load(path, map_location='cpu')
    model.load_state_dict({name: tensor.float() for name, tensor in state.items()})  # stored as float16
    return model.to(device).eval()


def _padding_px(size_px):
    """Bottom or right padding that makes size_px a multiple of SIZE_MULTIPLE_PX and at least one tile."""
    padding_px = max((-size_px) % SIZE_MULTIPLE_PX, TILE_PX - size_px)
    return padding_px + (-(size_px + padding_px)) % SIZE_MULTIPLE_PX


def _tile_origins(size_px):
    last = max(size_px - TILE_PX, 0)
    return sorted({*range(0, last + 1, TILE_PX - OVERLAP_PX), last})


def _blend_weights():
    """Tile weights that rise linearly over the overlap from each edge."""
    ramp = np.minimum(np.arange(1, TILE_PX + 1), np.arange(TILE_PX, 0, -1)).astype('float32')
    ramp = np.minimum(ramp, OVERLAP_PX) / OVERLAP_PX
    return np.outer(ramp, ramp)


def _water_probability(model, batch, tta):
    """Water probability of a batch of tiles, averaged with its left-right and up-down flips when tta."""
    probability = torch.softmax(model(batch), 1)[:, 1]
    if not tta:
        return probability
    flipped_x = torch.softmax(model(batch.flip(-1)), 1)[:, 1].flip(-1)
    flipped_y = torch.softmax(model(batch.flip(-2)), 1)[:, 1].flip(-2)
    return (probability + flipped_x + flipped_y) / 3


def predict(model, bands, device, tta=False, keep=None):
    """Water probability (H, W) float32, NaN where no data; tiles without a `keep` pixel are skipped and get 0."""
    inputs, valid = features(bands)
    height, width = valid.shape
    inputs = np.pad(inputs, ((0, 0), (0, _padding_px(height)), (0, _padding_px(width))))
    padded_height, padded_width = inputs.shape[1:]
    x_origins = _tile_origins(padded_width)
    origins = [
        (y, x)
        for y in _tile_origins(padded_height)
        for x in x_origins
        if inputs[:, y : y + TILE_PX, x : x + TILE_PX].any()
        and (keep is None or keep[y : y + TILE_PX, x : x + TILE_PX].any())
    ]
    weights = _blend_weights()
    total = np.zeros((padded_height, padded_width), 'float32')
    weight_sum = np.zeros((padded_height, padded_width), 'float32')
    with torch.no_grad():
        for i in range(0, len(origins), BATCH):
            batch_origins = origins[i : i + BATCH]
            tiles = np.stack([inputs[:, y : y + TILE_PX, x : x + TILE_PX] for y, x in batch_origins])
            probabilities = _water_probability(model, torch.from_numpy(tiles).to(device), tta).float().cpu().numpy()
            for (y, x), probability in zip(batch_origins, probabilities, strict=True):
                total[y : y + TILE_PX, x : x + TILE_PX] += probability * weights
                weight_sum[y : y + TILE_PX, x : x + TILE_PX] += weights
    water = np.where(weight_sum > 0, total / np.maximum(weight_sum, 1e-6), 0.0)[:height, :width]
    return np.where(valid, water, np.nan).astype('float32')


def read_bands(path):
    """(6, H, W) float32 reflectance with NaN for no data, and the raster profile, from a pipeline composite."""
    with rasterio.open(path) as src:
        stack = src.read().astype('float32')
        profile = src.profile
    return np.where(stack <= -9990, np.nan, stack / 1e4).astype('float32'), profile
