"""Train the radar flood segmentation U-Net on Kuro Siwo, optionally with Sen1Floods11 tiles in every batch."""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import segmentation_model

# Kuro Siwo BlackBench event-disjoint split ("RandomEvents"), by activation id
TRAIN_EVENTS = [
    '130',
    '470',
    '555',
    '118',
    '174',
    '324',
    '421',
    '554',
    '427',
    '518',
    '502',
    '498',
    '497',
    '496',
    '492',
    '147',
    '267',
    '273',
    '275',
    '417',
    '567',
    '1111011',
    '1111004',
    '1111009',
    '1111010',
    '1111006',
    '1111005',
]
VAL_EVENTS = ['514', '559', '279', '520', '437', '1111003', '1111008']
TEST_EVENTS = ['321', '561', '445', '562', '411', '1111002', '277', '1111007', '205', '1111013']

DEVICE = 'cuda'
ARCH = 'Unet'
IN_CHANNELS = 6  # post VV, post VH, pre1 VV, pre1 VH, pre2 VV, pre2 VH
CLASSES = 3  # no water, permanent water, flood
CLASS_NAMES = ['nowater', 'perm', 'flood']
FLOOD = 2
IGNORE = 255
EPOCHS = 30
STEPS_PER_EPOCH = 800
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
WARMUP_STEPS = 500
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CLASS_WEIGHTS = (0.5, 1.0, 2.0)
POSITIVE_FRACTION = 0.7  # share of each batch drawn from training tiles that contain flood
EXTRA_FRACTION = 0.25  # share of each batch drawn from the extra training tiles
DB_STATS_TILES = 2000
FLOOD_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

GAIN_SD_DB = 1.0
POL_OFFSET_SD_DB = 0.4
SPECKLE_P = 0.3
BLUR_P = 0.2
PRE_SWAP_P = 0.5
PRE_DROP_P = 0.1


class ConcatenatedArray:
    """Read-only concatenation of arrays (memmaps) along axis 0, without copying."""

    def __init__(self, parts):
        self.parts = parts
        self.offsets = np.cumsum([0, *[len(p) for p in parts]])
        self.shape = (int(self.offsets[-1]), *parts[0].shape[1:])
        self.dtype = parts[0].dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self[index[0]][(slice(None), *index[1:])]
        if np.isscalar(index):
            part = int(np.searchsorted(self.offsets, index, side='right') - 1)
            return self.parts[part][index - self.offsets[part]]
        index = np.asarray(index)
        out = np.empty((len(index), *self.shape[1:]), self.dtype)
        for part, array in enumerate(self.parts):
            in_part = (index >= self.offsets[part]) & (index < self.offsets[part + 1])
            if in_part.any():
                local = index[in_part] - self.offsets[part]
                order = np.argsort(local, kind='stable')  # memmaps read fastest in ascending order
                rows = np.empty((len(local), *self.shape[1:]), self.dtype)
                rows[order] = array[local[order]]
                out[in_part] = rows
        return out


@dataclass
class Tiles:
    codes: np.ndarray | ConcatenatedArray  # [N, 6, 224, 224] uint8 dB codes
    labels: np.ndarray | ConcatenatedArray  # [N, 224, 224] uint8
    events: np.ndarray
    train_idx: np.ndarray
    positive_idx: np.ndarray  # training tiles that contain flood
    negative_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    extra_train_idx: np.ndarray
    extra_val_idx: np.ndarray


def load_extra(paths):
    """Tiles in the Kuro Siwo compact format whose metadata carry a split (train or valid)."""
    codes, labels, meta = [], [], []
    for path in paths:
        with np.load(path) as arrays:
            codes.append(arrays['X'])
            labels.append(arrays['Y'])
            meta += json.loads(str(arrays['meta']))
    return np.concatenate(codes), np.concatenate(labels), meta


def load_tiles(memmap_dir, extra_paths):
    codes = np.load(os.path.join(memmap_dir, 'X.npy'), mmap_mode='r')
    labels = np.load(os.path.join(memmap_dir, 'Y.npy'), mmap_mode='r')
    with open(os.path.join(memmap_dir, 'meta.json')) as f:
        meta = json.load(f)
    events = np.array([m['actid'] for m in meta])
    is_train = np.isin(events, TRAIN_EVENTS)
    is_val = np.isin(events, VAL_EVENTS)
    is_test = np.isin(events, TEST_EVENTS)
    is_extra_train = np.zeros(len(events), bool)
    is_extra_val = np.zeros(len(events), bool)
    if extra_paths:
        extra_codes, extra_labels, extra_meta = load_extra(extra_paths)
        split = np.array([m['split'] for m in extra_meta])
        codes = ConcatenatedArray([codes, extra_codes])
        labels = ConcatenatedArray([labels, extra_labels])
        events = np.concatenate([events, np.array([m['actid'] for m in extra_meta])])
        padding = np.zeros(len(extra_meta), bool)
        is_train, is_val, is_test = (np.concatenate([mask, padding]) for mask in (is_train, is_val, is_test))
        is_extra_train = np.concatenate([is_extra_train, split == 'train'])
        is_extra_val = np.concatenate([is_extra_val, split == 'valid'])
        print('extra tiles train/valid', int(is_extra_train.sum()), int(is_extra_val.sum()), flush=True)
    print('samples train/val/test', is_train.sum(), is_val.sum(), is_test.sum(), flush=True)
    train_idx = np.where(is_train)[0]
    has_flood = np.array([(labels[i] == FLOOD).any() for i in train_idx])
    print('train with flood px', has_flood.mean(), flush=True)
    return Tiles(
        codes=codes,
        labels=labels,
        events=events,
        train_idx=train_idx,
        positive_idx=train_idx[has_flood],
        negative_idx=train_idx[~has_flood],
        val_idx=np.where(is_val)[0],
        test_idx=np.where(is_test)[0],
        extra_train_idx=np.where(is_extra_train)[0],
        extra_val_idx=np.where(is_extra_val)[0],
    )


def db_stats(codes, train_idx):
    """Mean and standard deviation (dB) of post-event VV and VH over a fixed sample of training tiles."""
    sample = np.random.RandomState(0).choice(train_idx, min(DB_STATS_TILES, len(train_idx)), replace=False)
    post = codes[sample, :2].astype('float32')
    post[post == 0] = np.nan
    post = post / 6.0 - 35.0
    mean_db = {pol: float(np.nanmean(post[:, i])) for i, pol in enumerate(['VV', 'VH'])}
    std_db = {pol: float(np.nanstd(post[:, i])) for i, pol in enumerate(['VV', 'VH'])}
    return mean_db, std_db


def db_scale(mean_db, std_db):
    """Per-channel (mean, std) tensors [1, 6, 1, 1] for the three dates."""
    mean = torch.tensor([mean_db['VV'], mean_db['VH']] * 3, device=DEVICE)[None, :, None, None]
    std = torch.tensor([std_db['VV'], std_db['VH']] * 3, device=DEVICE)[None, :, None, None]
    return mean, std


def codes_to_db(codes):
    """uint8 codes to dB; code 0 (no data) becomes NaN."""
    db = codes.float() / 6.0 - 35.0
    return torch.where(codes == 0, torch.full_like(db, float('nan')), db)


def filled_db(codes):
    """dB with no-data pixels set to the median of their tile and channel (-15 dB if the channel is empty)."""
    db = codes_to_db(codes)
    median = torch.nanmedian(db.flatten(2), dim=2).values[..., None, None]
    db = torch.where(torch.isnan(db), median.expand_as(db), db)
    return torch.nan_to_num(db, nan=-15.0)


def normalised(db, scale):
    mean, std = scale
    return (db - mean) / std


def augment(db, labels):
    """Random D4 orientation, then per-tile gain, speckle, blur and pre-image swap or drop on the dB channels."""
    batch = db.shape[0]
    k = np.random.randint(4)
    db = torch.rot90(db, k, (2, 3))
    labels = torch.rot90(labels, k, (1, 2))
    if np.random.rand() < 0.5:
        db = db.flip(3)
        labels = labels.flip(2)
    device = db.device
    # one gain per date shared by VV and VH, plus one offset per polarisation
    gain = torch.randn(batch, 3, 1, 1, 1, device=device) * GAIN_SD_DB
    pol_offset = torch.randn(batch, 1, 2, 1, 1, device=device) * POL_OFFSET_SD_DB
    db = (db.reshape(batch, 3, 2, *db.shape[2:]) + gain + pol_offset).reshape(batch, 6, *db.shape[2:])
    if np.random.rand() < SPECKLE_P:
        # multiplicative gamma speckle in linear power is additive in dB
        looks = torch.tensor(float(np.random.uniform(4, 10)), device=device)
        gamma = torch.distributions.Gamma(looks, looks).sample(db.shape)
        db = db + 10 * torch.log10(gamma.clamp_min(1e-3))
    if np.random.rand() < BLUR_P:
        kernel = torch.ones(db.shape[1], 1, 3, 3, device=device) / 9.0
        db = functional.conv2d(functional.pad(db, (1, 1, 1, 1), mode='reflect'), kernel, groups=db.shape[1])
    swap = torch.rand(batch, device=device) < PRE_SWAP_P
    if swap.any():
        db[swap] = db[swap][:, [0, 1, 4, 5, 2, 3]]
    drop = torch.rand(batch, device=device) < PRE_DROP_P
    if drop.any():
        db[drop, 4:6] = db[drop, 2:4]
    return db, labels


def dice_loss(logits, labels):
    """Soft Dice loss averaged over the permanent-water and flood classes."""
    valid = labels != IGNORE
    prob = logits.softmax(1)
    loss = 0
    for c in (1, 2):
        p = prob[:, c][valid]
        t = (labels[valid] == c).float()
        intersection = (p * t).sum()
        total = p.sum() + t.sum()
        loss = loss + 1 - (2 * intersection + 1) / (total + 1)
    return loss / 2


def lr_factor(step):
    """Linear warm-up, then cosine decay to zero at the last step."""
    return min(1, (step + 1) / WARMUP_STEPS) * 0.5 * (1 + math.cos(math.pi * min(step, TOTAL_STEPS) / TOTAL_STEPS))


@torch.no_grad()
def confusion_matrix(pred, labels):
    valid = labels != IGNORE
    index = labels[valid].long() * CLASSES + pred[valid].long()
    return torch.bincount(index, minlength=CLASSES * CLASSES).view(CLASSES, CLASSES).cpu().numpy()


def class_metrics(confusion):
    """IoU and F1 per class, mean IoU, and IoU and F1 of water (permanent + flood)."""
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        out[f'iou_{name}'] = tp / max(tp + fp + fn, 1)
        out[f'f1_{name}'] = 2 * tp / max(2 * tp + fp + fn, 1)
    out['miou'] = np.mean([out[f'iou_{name}'] for name in CLASS_NAMES])
    water_tp = confusion[1:, 1:].sum()
    water_fp = confusion[0, 1:].sum()
    water_fn = confusion[1:, 0].sum()
    out['f1_water'] = 2 * water_tp / max(2 * water_tp + water_fp + water_fn, 1)
    out['iou_water'] = water_tp / max(water_tp + water_fp + water_fn, 1)
    return {k: float(v) for k, v in out.items()}


def selection_score(metrics):
    return metrics['iou_flood'] + 0.5 * metrics['iou_perm']


def eval_batches(tiles, idx):
    for start in range(0, len(idx), EVAL_BATCH_SIZE):
        batch = idx[start : start + EVAL_BATCH_SIZE]
        codes = torch.from_numpy(tiles.codes[batch]).to(DEVICE)
        labels = torch.from_numpy(tiles.labels[batch]).to(DEVICE)
        yield filled_db(codes), labels


def tta_probability_sum(model, x):
    """Sum of class probabilities over the 8 rotations and flips, in the input orientation."""
    total = 0
    for k in range(4):
        for flip in (False, True):
            rotated = torch.rot90(x, k, (2, 3))
            rotated = rotated.flip(3) if flip else rotated
            prob = model(rotated).float().softmax(1)
            prob = prob.flip(3) if flip else prob
            total = total + torch.rot90(prob, -k, (2, 3))
    return total


@torch.no_grad()
def evaluate(model, tiles, idx, scale, use_tta=False):
    """Class metrics and confusion matrix pooled over the tiles idx."""
    model.eval()
    confusion = np.zeros((CLASSES, CLASSES), np.int64)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for db, labels in eval_batches(tiles, idx):
            x = normalised(db, scale)
            pred = tta_probability_sum(model, x).argmax(1) if use_tta else model(x).float().argmax(1)
            confusion += confusion_matrix(pred, labels)
    model.train()
    return class_metrics(confusion), confusion


@torch.no_grad()
def threshold_metrics(model, tiles, idx, scale):
    """Class metrics (with TTA) when flood is predicted wherever p_flood > t, for each t in FLOOD_THRESHOLDS."""
    model.eval()
    confusions = {t: np.zeros((CLASSES, CLASSES), np.int64) for t in FLOOD_THRESHOLDS}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for db, labels in eval_batches(tiles, idx):
            prob = tta_probability_sum(model, normalised(db, scale)) / 8.0
            not_flood = prob[:, :2].argmax(1)
            for t in FLOOD_THRESHOLDS:
                pred = torch.where(prob[:, FLOOD] > t, torch.full_like(not_flood, FLOOD), not_flood)
                confusions[t] += confusion_matrix(pred, labels)
    model.train()
    return {t: class_metrics(confusion) for t, confusion in confusions.items()}


def sample_batch(tiles):
    positive_count = round(BATCH_SIZE * POSITIVE_FRACTION)
    batch = np.concatenate(
        [
            np.random.choice(tiles.positive_idx, positive_count),
            np.random.choice(tiles.negative_idx, BATCH_SIZE - positive_count),
        ]
    )
    if len(tiles.extra_train_idx):
        extra_count = round(BATCH_SIZE * EXTRA_FRACTION)
        batch[np.random.choice(BATCH_SIZE, extra_count, replace=False)] = np.random.choice(
            tiles.extra_train_idx, extra_count
        )
    batch.sort()
    return batch


def train_epoch(model, optimizer, scheduler, tiles, scale):
    """One epoch of training; returns the mean loss."""
    class_weights = torch.tensor(CLASS_WEIGHTS, device=DEVICE)
    loss_sum = 0
    for _ in range(STEPS_PER_EPOCH):
        batch = sample_batch(tiles)
        codes = torch.from_numpy(tiles.codes[batch]).to(DEVICE, non_blocking=True)
        labels = torch.from_numpy(tiles.labels[batch]).to(DEVICE).long()
        db, labels = augment(filled_db(codes), labels)
        x = normalised(db, scale).contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        logits = logits.float()
        cross_entropy = functional.cross_entropy(logits, labels, weight=class_weights, ignore_index=IGNORE)
        loss = cross_entropy + dice_loss(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_sum += loss.item()
    return loss_sum / STEPS_PER_EPOCH


def validation(model, tiles, scale):
    """Kuro Siwo val metrics, and the log record of the epoch with its checkpoint selection score."""
    val_metrics, _ = evaluate(model, tiles, tiles.val_idx, scale)
    record = {f'val_{k}': v for k, v in val_metrics.items()}
    val_events = tiles.events[tiles.val_idx]
    record['val_ev_flood'] = {
        str(e): evaluate(model, tiles, tiles.val_idx[val_events == e], scale)[0]['iou_flood']
        for e in sorted(set(val_events))
    }
    record['val_evmean_flood'] = float(np.mean(list(record['val_ev_flood'].values())))
    score = selection_score(val_metrics)
    if len(tiles.extra_val_idx):
        extra_metrics, _ = evaluate(model, tiles, tiles.extra_val_idx, scale)
        record.update({f'xval_{k}': v for k, v in extra_metrics.items()})
        score = 0.5 * (score + selection_score(extra_metrics))
    record['score'] = score
    return val_metrics, record


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=1)


def train(model, tiles, scale, out_dir):
    """Train for EPOCHS epochs and keep the epoch with the best selection score as best.pt."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    best_score = -1
    with open(os.path.join(out_dir, 'log.jsonl'), 'a') as log:
        for epoch in range(EPOCHS):
            start = time.time()
            loss = train_epoch(model, optimizer, scheduler, tiles, scale)
            val_metrics, validation_record = validation(model, tiles, scale)
            record = dict(
                epoch=epoch,
                loss=loss,
                secs=round(time.time() - start),
                lr=scheduler.get_last_lr()[0],
                **validation_record,
            )
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + '\n')
            log.flush()
            torch.save(model.state_dict(), os.path.join(out_dir, 'last.pt'))
            if record['score'] > best_score:
                best_score = record['score']
                torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
                write_json(dict(epoch=epoch, **val_metrics), os.path.join(out_dir, 'best_val.json'))


def final_metrics(model, tiles, scale, eval_test):
    """Plain and TTA metrics per split, and plain metrics per event."""
    splits = {'val': tiles.val_idx}
    if eval_test:
        splits['test'] = tiles.test_idx
    if len(tiles.extra_val_idx):
        splits['s1f11_valid'] = tiles.extra_val_idx
    results = {}
    for name, idx in splits.items():
        plain, _ = evaluate(model, tiles, idx, scale)
        tta, tta_confusion = evaluate(model, tiles, idx, scale, use_tta=True)
        results[name] = dict(plain=plain, tta=tta, cm_tta=tta_confusion.tolist())
        events = tiles.events[idx]
        results[f'{name}_per_event'] = {
            e: evaluate(model, tiles, idx[events == e], scale)[0] for e in sorted(set(events))
        }
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mm', required=True, help='directory with the Kuro Siwo X.npy, Y.npy and meta.json')
    parser.add_argument('--out', required=True, help='output directory')
    parser.add_argument('--encoder', default='resnet34')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--extra', default='', help='comma-separated npz files of extra tiles (Sen1Floods11)')
    parser.add_argument('--eval_test', action='store_true', help='also score the held-out Kuro Siwo test events')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    tiles = load_tiles(args.mm, args.extra.split(',') if args.extra else [])
    mean_db, std_db = db_stats(tiles.codes, tiles.train_idx)
    print('dB stats', mean_db, std_db, flush=True)
    scale = db_scale(mean_db, std_db)
    model = (
        segmentation_model.build(ARCH, args.encoder, IN_CHANNELS, CLASSES)
        .to(DEVICE)
        .to(memory_format=torch.channels_last)
    )
    config = dict(
        vars(args),
        arch=ARCH,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        bs=BATCH_SIZE,
        lr=LEARNING_RATE,
        wd=WEIGHT_DECAY,
        cw=list(CLASS_WEIGHTS),
        pos_frac=POSITIVE_FRACTION,
        extra_frac=EXTRA_FRACTION,
        db_mean=mean_db,
        db_std=std_db,
        n_train=len(tiles.train_idx),
        n_val=len(tiles.val_idx),
        n_test=len(tiles.test_idx),
        n_extra_train=len(tiles.extra_train_idx),
        n_extra_val=len(tiles.extra_val_idx),
    )
    config_path = os.path.join(args.out, 'config.json')
    write_json(config, config_path)
    train(model, tiles, scale, args.out)

    model.load_state_dict(torch.load(os.path.join(args.out, 'best.pt'), map_location=DEVICE))
    results = final_metrics(model, tiles, scale, args.eval_test)
    # the flood probability threshold with the best pooled flood IoU on Kuro Siwo val is the DL-only operating point
    by_threshold = threshold_metrics(model, tiles, tiles.val_idx, scale)
    best_threshold = max(FLOOD_THRESHOLDS, key=lambda t: by_threshold[t]['iou_flood'])
    results['calib_val'] = dict(
        best_thr=best_threshold,
        by_thr={
            str(t): {key: by_threshold[t][key] for key in ('iou_flood', 'f1_flood', 'miou')} for t in FLOOD_THRESHOLDS
        },
    )
    config['flood_thr'] = best_threshold
    write_json(config, config_path)
    print('CALIB val flood thr', best_threshold, {t: round(m['iou_flood'], 4) for t, m in by_threshold.items()})
    write_json(results, os.path.join(args.out, 'final_metrics.json'))
    print('FINAL', json.dumps({k: v for k, v in results.items() if not k.endswith('per_event')}), flush=True)


if __name__ == '__main__':
    main()
