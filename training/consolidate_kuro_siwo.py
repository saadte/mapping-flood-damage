"""Merge Kuro Siwo chunks into memory-mappable X.npy, Y.npy and meta.json, keeping the first copy of each tile."""

import argparse
import glob
import json
import os

import numpy as np

TILE_PX = 224


def unique_tiles(chunk_paths):
    """Per chunk, the indices of tiles not seen in an earlier chunk, and the metadata of all kept tiles."""
    seen = set()
    plan = []
    metas = []
    for path in chunk_paths:
        with np.load(path) as chunk:
            meta = json.loads(str(chunk['meta']))
        keep = []
        for i, m in enumerate(meta):
            tile_id = (m['actid'], m['aoiid'], m['grid_id'], m['flood_date'])
            if tile_id in seen:  # the HF train and test shards share tiles
                continue
            seen.add(tile_id)
            keep.append(i)
        plan.append((path, keep))
        metas += [meta[i] for i in keep]
    return plan, metas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('chunk_dir')
    parser.add_argument('out_dir')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plan, metas = unique_tiles(sorted(glob.glob(os.path.join(args.chunk_dir, '*.npz'))))
    count = len(metas)
    print('unique tiles', count, flush=True)
    codes = np.lib.format.open_memmap(
        os.path.join(args.out_dir, 'X.npy'), mode='w+', dtype=np.uint8, shape=(count, 6, TILE_PX, TILE_PX)
    )
    labels = np.lib.format.open_memmap(
        os.path.join(args.out_dir, 'Y.npy'), mode='w+', dtype=np.uint8, shape=(count, TILE_PX, TILE_PX)
    )
    start = 0
    for path, keep in plan:
        if not keep:
            continue
        with np.load(path) as chunk:
            codes[start : start + len(keep)] = chunk['X'][keep]
            labels[start : start + len(keep)] = chunk['Y'][keep]
        start += len(keep)
    codes.flush()
    labels.flush()
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(metas, f)


if __name__ == '__main__':
    main()
