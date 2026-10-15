"""Relabel Sen1Floods11 flood pixels by the pre-event images: water before the event becomes permanent water."""

import argparse
import json
import os
import urllib.request

import numpy as np

BUCKET_URL = 'https://storage.googleapis.com/sen1floods11/v1.1'
LOCATION_NAMES = {'Mekong': 'Cambodia'}  # chip ids use Mekong, the metadata use Cambodia
PERMANENT = 1
FLOOD = 2
IGNORE = 255
PRE1_VH = 3
PRE2_VH = 5


def vh_thresholds_db():
    """Sen1Floods11's own per-event VH water threshold (dB), by location."""
    with urllib.request.urlopen(f'{BUCKET_URL}/Sen1Floods11_Metadata.geojson') as response:
        features = json.load(response)['features']
    return {f['properties']['location']: float(f['properties']['VH_thresh']) for f in features}


def relabel(codes, labels, metas, thresholds_db):
    """Flood that is water on both pre dates becomes permanent water; flood that is water on one pre date is ignored."""
    # Sen1Floods11 flood is any water outside JRC permanent water; Kuro Siwo flood is water new since the pre dates
    stats = {}
    for i, meta in enumerate(metas):
        location = meta['actid'].removeprefix('s1f11_')
        threshold_db = thresholds_db[LOCATION_NAMES.get(location, location)]
        pre1_water = (codes[i, PRE1_VH].astype('float32') / 6.0 - 35.0 < threshold_db) & (codes[i, PRE1_VH] > 0)
        pre2_water = (codes[i, PRE2_VH].astype('float32') / 6.0 - 35.0 < threshold_db) & (codes[i, PRE2_VH] > 0)
        flood = labels[i] == FLOOD
        counts = stats.setdefault(location, [0, 0, 0])
        counts[0] += int(flood.sum())
        counts[1] += int((flood & pre1_water & pre2_water).sum())
        counts[2] += int((flood & (pre1_water ^ pre2_water)).sum())
        labels[i][flood & pre1_water & pre2_water] = PERMANENT
        labels[i][flood & (pre1_water ^ pre2_water)] = IGNORE
    for location, (flood_px, pre_existing_px, ambiguous_px) in stats.items():
        print(
            location,
            'flood px',
            flood_px,
            'pre-existing',
            round(pre_existing_px / max(flood_px, 1), 3),
            'ambiguous',
            round(ambiguous_px / max(flood_px, 1), 3),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir', help='directory with s1f11_train.npz and s1f11_valid.npz')
    args = parser.parse_args()
    thresholds_db = vh_thresholds_db()
    for split in ('train', 'valid'):
        with np.load(os.path.join(args.data_dir, f's1f11_{split}.npz')) as tiles:
            codes, labels, metas = tiles['X'], tiles['Y'].copy(), json.loads(str(tiles['meta']))
        relabel(codes, labels, metas, thresholds_db)
        out = os.path.join(args.data_dir, f's1f11_{split}_rl.npz')
        np.savez_compressed(out, X=codes, Y=labels, meta=json.dumps(metas))
        print(
            split, 'tiles with flood after relabelling', int(sum((y == FLOOD).any() for y in labels)), 'of', len(labels)
        )


if __name__ == '__main__':
    main()
