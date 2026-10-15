"""Flood class probabilities from a Kuro Siwo-trained radar U-Net on one Sentinel-1 track."""

import json
import os

import numpy as np
import torch
from torch import nn

import s1_change_rule
import segmentation_model

CLIP_DB = (-34.833, 7.5)  # input range of every model in training
TILE_PX = 512
OVERLAP_PX = 128
CLASS_COUNT = 3
ADABN_TILES = 256
ADABN_BATCH = 16
ADABN_TILE_PX = 256
ADABN_GRID_PX = 16  # spacing of the candidate tile origins
ADABN_MIN_CANDIDATES = 50
ADABN_MIN_VALID_FRACTION = 0.9


def build_input(lin, inc_deg=None, filt='lee7'):
    """Model input (6, H, W) in dB, NaN = no data, from linear (VV, VH) pairs named 'post', 'pre1' (latest), 'pre2'."""
    # gamma0 x cos(ellipsoid incidence) is the sigma0 the models were trained on (exact on flat ground)
    cos_incidence = np.cos(np.radians(inc_deg)).astype('float32') if inc_deg is not None else 1.0
    channels = []
    for name in ('post', 'pre1', 'pre2'):
        for band in lin[name]:
            linear = band * cos_incidence
            if filt == 'lee7':
                channels.append(s1_change_rule.lee_db(linear, 7))
            elif filt == 'box5':
                channels.append(s1_change_rule.boxcar_db(linear, 5))
            else:
                channels.append(s1_change_rule.to_db(linear))
    return np.stack(channels).astype('float32')


def _valid_fraction_per_tile(valid, tile_px):
    """Valid fraction of each tile whose origin lies on the ADABN_GRID_PX grid (exact, from an integral image)."""
    height, width = valid.shape
    integral = np.zeros((height + 1, width + 1), 'int64')
    integral[1:, 1:] = np.cumsum(np.cumsum(valid.astype('int32'), 0), 1)
    y0 = np.arange(0, max(height - tile_px, 0) + 1, ADABN_GRID_PX)
    x0 = np.arange(0, max(width - tile_px, 0) + 1, ADABN_GRID_PX)
    y1 = np.minimum(y0 + tile_px, height)
    x1 = np.minimum(x0 + tile_px, width)
    counts = integral[y1][:, x1] - integral[y0][:, x1] - integral[y1][:, x0] + integral[y0][:, x0]
    return counts / float(tile_px * tile_px)


def _well_covered(centre_mask, valid_fraction):
    """Select tiles centred in the requested mask, lowering the valid-pixel requirement if few qualify."""
    valid_fraction = valid_fraction[: centre_mask.shape[0], : centre_mask.shape[1]]
    centre_mask = centre_mask[: valid_fraction.shape[0], : valid_fraction.shape[1]]
    for min_fraction in (ADABN_MIN_VALID_FRACTION, 0.5):
        ys, xs = np.nonzero(centre_mask & (valid_fraction >= min_fraction))
        if len(ys) > ADABN_MIN_CANDIDATES:
            return ys, xs
    return np.nonzero(valid_fraction >= 0.5)  # almost no valid data inside the mask: any well-covered tile


def _adabn_origins(mask, valid, height, width):
    """Origins (ys, xs) of the tiles AdaBN may sample, or None to sample anywhere."""
    if mask is None and valid is None:
        return None
    if mask is None:
        mask = np.ones((height, width), bool)
    half = ADABN_TILE_PX // 2
    centre_mask = mask[half : height - half : ADABN_GRID_PX, half : width - half : ADABN_GRID_PX]
    if valid is None:
        ys, xs = np.nonzero(centre_mask)
    else:
        ys, xs = _well_covered(centre_mask, _valid_fraction_per_tile(valid, ADABN_TILE_PX))
    if len(ys) <= ADABN_MIN_CANDIDATES:
        return None
    return ys * ADABN_GRID_PX, xs * ADABN_GRID_PX


def _adapt_batch_norm(model, inputs, device, mask, valid):
    """Estimate batch-normalisation statistics from this scene without labels or gradient updates."""
    rng = np.random.RandomState(0)
    _, height, width = inputs.shape
    origins = _adabn_origins(mask, valid, height, width)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            module.momentum = None  # plain average over all sampled tiles
            module.train()
    with torch.no_grad():
        for _ in range(ADABN_TILES // ADABN_BATCH):
            batch = []
            for _ in range(ADABN_BATCH):
                if origins is None:
                    y0 = rng.randint(0, max(height - ADABN_TILE_PX, 1))
                    x0 = rng.randint(0, max(width - ADABN_TILE_PX, 1))
                else:
                    k = rng.randint(len(origins[0]))
                    y0, x0 = int(origins[0][k]), int(origins[1][k])
                batch.append(inputs[:, y0 : y0 + ADABN_TILE_PX, x0 : x0 + ADABN_TILE_PX])
            model(torch.from_numpy(np.stack(batch)).to(device))
    model.eval()


def _normalized_input(x_db, db_mean, db_std, diff, terrain):
    mean = np.array([db_mean['VV'], db_mean['VH']] * 3, 'float32')[:, None, None]
    std = np.array([db_std['VV'], db_std['VH']] * 3, 'float32')[:, None, None]
    median = np.nanmedian(x_db.reshape(x_db.shape[0], -1), axis=1)[:, None, None]
    filled = np.where(np.isfinite(x_db), x_db, median)
    inputs = (filled - mean) / std
    if diff:
        change = filled[0:2] - 0.5 * (filled[2:4] + filled[4:6])  # post minus the mean of the two pre images
        inputs = np.concatenate([inputs, change / 3.0], 0)
    if terrain is not None:
        inputs = np.concatenate([inputs, terrain.astype('float32')], 0)
    return inputs.astype('float32')


def _blend_window():
    """Hann window over the tile, floored at 1e-3 so that edge pixels keep some weight."""
    hann = np.hanning(TILE_PX).astype('float32')
    hann = np.maximum(hann, 1e-3)
    return np.outer(hann, hann)


def _tile_origins(size_px):
    origins = list(range(0, max(size_px - TILE_PX, 0) + 1, TILE_PX - OVERLAP_PX))
    if origins[-1] + TILE_PX < size_px:
        origins.append(size_px - TILE_PX)
    return origins


def _tile_probabilities(model, tile, device, tta):
    """Class probabilities of one tile, averaged over its 8 rotations and flips when tta."""
    variants = [(k, flip) for k in range(4) for flip in (False, True)] if tta else [(0, False)]
    total = 0
    # bf16 autocast only on GPU: CPUs without native bf16 emulate it about 100x slower
    with torch.autocast(device.split(':')[0], dtype=torch.bfloat16, enabled=device.startswith('cuda')):
        for k, flip in variants:
            variant = torch.rot90(tile, k, (2, 3))
            if flip:
                variant = variant.flip(3)
            output = model(variant).float().softmax(1)
            if flip:
                output = output.flip(3)
            total = total + torch.rot90(output, -k, (2, 3))
    return total / len(variants)


def _tiled_probabilities(model, inputs, device, tta):
    _, height, width = inputs.shape
    window = _blend_window()
    probabilities = np.zeros((CLASS_COUNT, height, width), 'float32')
    weight_sum = np.zeros((height, width), 'float32')
    for y0 in _tile_origins(height):
        for x0 in _tile_origins(width):
            tile = torch.from_numpy(inputs[:, y0 : y0 + TILE_PX, x0 : x0 + TILE_PX]).unsqueeze(0).to(device)
            pad_y, pad_x = TILE_PX - tile.shape[2], TILE_PX - tile.shape[3]
            if pad_y or pad_x:  # scene smaller than a tile
                tile = torch.nn.functional.pad(tile, (0, pad_x, 0, pad_y), mode='reflect')
            rows, columns = TILE_PX - pad_y, TILE_PX - pad_x
            tile_probabilities = _tile_probabilities(model, tile, device, tta)[0].cpu().numpy()[:, :rows, :columns]
            weights = window[:rows, :columns]
            probabilities[:, y0 : y0 + rows, x0 : x0 + columns] += tile_probabilities * weights
            weight_sum[y0 : y0 + rows, x0 : x0 + columns] += weights
    probabilities /= np.maximum(weight_sum, 1e-6)
    return probabilities


@torch.no_grad()
def predict(  # noqa: PLR0913
    model,
    x_db,
    db_mean,
    db_std,
    device='cuda',
    tta=True,
    diff=False,
    adabn=False,
    adabn_mask=None,
    terrain=None,
    clip_db=None,
    adabn_valid=False,
):
    """Class probabilities (3, H, W), NaN where any input channel is missing."""
    if clip_db is not None:
        x_db = np.clip(x_db, clip_db[0], clip_db[1])  # keeps radar shadow and layover in the training range
    valid = np.isfinite(x_db).all(0)
    inputs = _normalized_input(x_db, db_mean, db_std, diff, terrain)
    if adabn:
        # adabn_valid: partial-coverage tracks would otherwise feed median-filled areas into the statistics
        _adapt_batch_norm(model, inputs, device, adabn_mask, valid if adabn_valid else None)
    model.eval()
    probabilities = _tiled_probabilities(model, inputs, device, tta)
    probabilities[:, ~valid] = np.nan
    return probabilities


def load_model(checkpoint_path, config_path, device):
    """Model in eval mode on device, and its training config."""
    if not str(device).startswith('cuda'):
        torch.set_num_threads(os.cpu_count() or 4)
    with open(config_path) as f:
        config = json.load(f)
    in_channels = (8 if config.get('diff') else 6) + (2 if config.get('terrain') else 0)
    model = segmentation_model.build(config.get('arch', 'Unet'), config['encoder'], in_channels, CLASS_COUNT)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return model.to(device).eval(), config
