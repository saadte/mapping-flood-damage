"""Train the optical water segmentation U-Net on Sen1Floods11 hand-labelled S2 chips; score test and Bolivia once."""

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.nn import functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import optical_ai

BUCKET_URL = 'https://storage.googleapis.com/sen1floods11/v1.1'
CHIP_URL = f'{BUCKET_URL}/data/flood_events/HandLabeled'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DOWNLOAD_THREADS = 8
CHIP_PX = 512
CROP_PX = 256
EPOCHS = 40
BATCH_SIZE = 16
CROPS_PER_CHIP = 4  # training crops per chip and epoch
ENCODER_LR = 3e-4
DECODER_LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_SHARE = 0.1
CLASS_WEIGHTS = [1.0, 2.0]  # no water, water
NO_DATA = -1
BLUR_P = 0.7
SHADE_P = 0.5


def download(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path + '.part')
        os.replace(path + '.part', path)
    return path


def split_chips(name):
    with urllib.request.urlopen(f'{BUCKET_URL}/splits/flood_handlabeled/flood_{name}_data.csv') as response:
        lines = response.read().decode().splitlines()
    return [line.split(',')[0].replace('_S1Hand.tif', '') for line in lines if line.strip()]


def load_chip(chip, chips_dir):
    """(six S2 bands as uint16 [6, 512, 512], labels int8 [512, 512] with -1 no data, 0 no water, 1 water)."""
    bands_path = download(f'{CHIP_URL}/S2Hand/{chip}_S2Hand.tif', os.path.join(chips_dir, f'{chip}_S2Hand.tif'))
    labels_path = download(
        f'{CHIP_URL}/LabelHand/{chip}_LabelHand.tif', os.path.join(chips_dir, f'{chip}_LabelHand.tif')
    )
    with rasterio.open(bands_path) as src:
        bands = src.read([i + 1 for i in optical_ai.S1F11_IDX]).astype('uint16')
    with rasterio.open(labels_path) as src:
        labels = src.read(1).astype('int8')
    return bands, labels


def load_split(name, chips_dir):
    with ThreadPoolExecutor(DOWNLOAD_THREADS) as executor:
        chips = list(executor.map(lambda chip: load_chip(chip, chips_dir), split_chips(name)))
    bands = np.stack([b for b, _ in chips])
    labels = np.stack([y for _, y in chips])
    water_share = (labels == 1).sum() / max((labels >= 0).sum(), 1)
    print(name, bands.shape, 'water share of labelled px', round(float(water_share), 3), flush=True)
    return bands, labels


def blur20(x):
    """Average 2 x 2 pixels, then interpolate back to represent a 20 m composite on the 10 m grid."""
    t = torch.from_numpy(x)[None]
    t = functional.interpolate(functional.avg_pool2d(t, 2), scale_factor=2, mode='bilinear', align_corners=False)
    return t[0].numpy()


def blurred_with_gaps(reflectance):
    """blur20 of the valid data, with no-data pixels kept as NaN."""
    no_data = ~np.isfinite(reflectance).all(0)
    reflectance = blur20(np.nan_to_num(reflectance, nan=0.0))
    reflectance[:, no_data] = np.nan
    return reflectance


def to_reflectance(bands):
    reflectance = bands.astype('float32') / 1e4
    reflectance[:, (bands == 0).all(0)] = np.nan
    return reflectance


def shading_field(rng, height, width, low=0.45, high=1.15):
    """Smooth multiplicative illumination field: terrain shading darkens every band and keeps band ratios."""
    coarse = rng.uniform(low, high, size=(1, 1, 4, 4)).astype('float32')
    field = functional.interpolate(torch.from_numpy(coarse), size=(height, width), mode='bicubic', align_corners=False)
    return field[0, 0].clamp(0.3, 1.3).numpy()


def training_sample(bands, labels, rng):
    """Random crop with 20 m blur, per-band gain and offset, shading and D4 orientation; (features, labels)."""
    y0 = rng.integers(0, CHIP_PX - CROP_PX + 1)
    x0 = rng.integers(0, CHIP_PX - CROP_PX + 1)
    x = to_reflectance(bands[:, y0 : y0 + CROP_PX, x0 : x0 + CROP_PX])
    y = labels[y0 : y0 + CROP_PX, x0 : x0 + CROP_PX].copy()
    if rng.random() < BLUR_P:
        x = blurred_with_gaps(x)
    gain = rng.uniform(0.85, 1.15, size=(6, 1, 1)).astype('float32') * rng.uniform(0.85, 1.15)
    offset = np.concatenate([rng.uniform(-0.01, 0.02, 3), rng.uniform(-0.01, 0.01, 3)]).astype('float32')
    x = x * gain + offset[:, None, None]
    if rng.random() < SHADE_P:
        x = x * shading_field(rng, CROP_PX, CROP_PX)[None]
    features, valid = optical_ai.features(x)
    y = np.where(valid, y, NO_DATA)
    k = rng.integers(0, 4)
    features = np.rot90(features, k, (1, 2))
    y = np.rot90(y, k)
    if rng.random() < 0.5:
        features = features[:, :, ::-1]
        y = y[:, ::-1]
    return np.ascontiguousarray(features), np.ascontiguousarray(y)


def evaluate(model, bands, labels, blurred):
    """Water IoU, F1, precision and recall pooled over a split, and the mean IoU of chips that contain water."""
    tp = fp = fn = 0
    chip_iou = []
    for i in range(len(bands)):
        x = to_reflectance(bands[i])
        if blurred:
            x = blurred_with_gaps(x)
        prob = optical_ai.predict(model, x, device=DEVICE)
        scored = (labels[i] >= 0) & np.isfinite(prob)
        pred = (prob > 0.5) & scored
        water = (labels[i] == 1) & scored
        chip_tp, chip_fp, chip_fn = int((pred & water).sum()), int((pred & ~water).sum()), int((~pred & water).sum())
        tp += chip_tp
        fp += chip_fp
        fn += chip_fn
        if water.sum() > 0:
            chip_iou.append(chip_tp / max(chip_tp + chip_fp + chip_fn, 1))
    return dict(
        iou=round(tp / max(tp + fp + fn, 1), 4),
        f1=round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        prec=round(tp / max(tp + fp, 1), 4),
        rec=round(tp / max(tp + fn, 1), 4),
        mean_chip_iou=round(float(np.mean(chip_iou)) if chip_iou else float('nan'), 4),
        n=len(bands),
    )


def scored_split(model, bands, labels):
    return dict(native=evaluate(model, bands, labels, False), blur20=evaluate(model, bands, labels, True))


def loss_on_batch(model, x, y, class_weights):
    """Weighted cross-entropy plus soft Dice on water."""
    logits = model(x)
    cross_entropy = functional.cross_entropy(logits, y, ignore_index=NO_DATA, weight=class_weights)
    prob = torch.softmax(logits.float(), 1)[:, 1]
    labelled = (y >= 0).float()
    water = (y == 1).float()
    dice = 1 - (2 * (prob * water * labelled).sum() + 1) / ((prob * labelled).sum() + (water * labelled).sum() + 1)
    return cross_entropy + dice


def train(model, bands, labels, valid_bands, valid_labels, rng):
    """Train for EPOCHS epochs; returns the half-precision state of the epoch with the best valid IoU, and the log."""
    encoder = [p for name, p in model.named_parameters() if name.startswith('encoder')]
    decoder = [p for name, p in model.named_parameters() if not name.startswith('encoder')]
    optimizer = torch.optim.AdamW(
        [dict(params=encoder, lr=ENCODER_LR), dict(params=decoder, lr=DECODER_LR)], weight_decay=WEIGHT_DECAY
    )
    steps = max(1, (len(bands) * CROPS_PER_CHIP) // BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[ENCODER_LR, DECODER_LR], total_steps=EPOCHS * steps, pct_start=WARMUP_SHARE
    )
    use_amp = DEVICE == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)
    class_weights = torch.tensor(CLASS_WEIGHTS, device=DEVICE)
    best_score, best_state, best_epoch = -1, None, None
    log = []
    for epoch in range(EPOCHS):
        model.train()
        start = time.time()
        loss_sum = 0
        for _ in range(steps):
            samples = [training_sample(bands[i], labels[i], rng) for i in rng.integers(0, len(bands), BATCH_SIZE)]
            x = torch.from_numpy(np.stack([f for f, _ in samples])).to(DEVICE)
            y = torch.from_numpy(np.stack([t for _, t in samples])).long().to(DEVICE)
            with torch.autocast(device_type=DEVICE, enabled=use_amp):
                loss = loss_on_batch(model, x, y, class_weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss)
        model.eval()
        valid = scored_split(model, valid_bands, valid_labels)
        log.append(
            dict(
                epoch=epoch,
                loss=round(loss_sum / steps, 4),
                valid_native=valid['native'],
                valid_blur20=valid['blur20'],
                secs=round(time.time() - start),
            )
        )
        print(json.dumps(log[-1]), flush=True)
        # composites are 20 m data on a 10 m grid, so selection weighs the blurred chips as much as the native ones
        score = 0.5 * (valid['native']['iou'] + valid['blur20']['iou'])
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {
                k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                for k, v in model.state_dict().items()
            }
    return best_state, best_score, best_epoch, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('chips_dir', help='download cache for the Sen1Floods11 chips')
    parser.add_argument('out_dir', help='output directory for best.pt, config.json and final_metrics.json')
    args = parser.parse_args()
    os.makedirs(args.chips_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    bands, labels = load_split('train', args.chips_dir)
    valid_bands, valid_labels = load_split('valid', args.chips_dir)
    model = optical_ai.build_model(pretrained=True).to(DEVICE)
    best_state, best_score, best_epoch, log = train(model, bands, labels, valid_bands, valid_labels, rng)
    model_path = os.path.join(args.out_dir, 'best.pt')
    torch.save(best_state, model_path)
    model = optical_ai.load_model(model_path, DEVICE)
    final = dict(
        run=os.path.basename(os.path.normpath(args.out_dir)),
        best_epoch=best_epoch,
        valid_sel=round(best_score, 4),
        log=log,
        n_train=len(bands),
        n_valid=len(valid_bands),
        encoder='resnet34, ImageNet initialisation',
        data='Sen1Floods11 hand-labelled (S2Hand, LabelHand), official splits',
    )
    # test and Bolivia are scored once, for the checkpoint selected on valid
    for name in ('test', 'bolivia'):
        final[name] = scored_split(model, *load_split(name, args.chips_dir))
        print('held-out', name, json.dumps(final[name]), flush=True)
    with open(os.path.join(args.out_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final, f, indent=1)
    config = dict(
        bands=optical_ai.BANDS,
        mean=optical_ai.MEAN.tolist(),
        std=optical_ai.STD.tolist(),
        arch='unet_resnet34',
        in_channels=9,
        classes=2,
        dtype='float16',
    )
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(config, f)


if __name__ == '__main__':
    main()
