"""Convert Kuro Siwo labelled GRD webdataset shards into chunks of uint8 dB codes, labels and tile metadata."""

import argparse
import io
import json
import os
import tarfile

import numpy as np

# post-event, most recent pre-event and older pre-event acquisitions
CHANNELS = ['flood_vv', 'flood_vh', 'sec1_vv', 'sec1_vh', 'sec2_vv', 'sec2_vh']
CHUNK_TILES = 1000
INVALID_CLASS = 3  # Kuro Siwo marks unusable pixels with class 3
IGNORE = 255


def db_codes(linear):
    """Linear backscatter to uint8 codes round((dB + 35) * 6) clipped to 1..255; code 0 is no data."""
    linear = linear.astype('float32')
    bad = ~np.isfinite(linear) | (linear <= 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        db = 10 * np.log10(np.where(bad, 1.0, linear))
    codes = np.clip(np.round((db + 35.0) * 6.0), 1, 255).astype('uint8')
    codes[bad] = 0
    return codes


def shard_samples(path):
    """(key, {extension: bytes}) for each sample of a webdataset tar, in archive order."""
    sample_key = None
    members = {}
    with tarfile.open(path, mode='r|*') as archive:
        for member in archive:
            if not member.isfile():
                continue
            key, _, extension = member.name.partition('.')
            if sample_key is not None and key != sample_key:
                yield sample_key, members
                members = {}
            sample_key = key
            members[extension] = archive.extractfile(member).read()
    if members:
        yield sample_key, members


def converted_sample(members, sample_id):
    """(codes [6, H, W], labels [H, W], metadata) of one Kuro Siwo sample."""
    info = json.loads(members['info.json'])
    codes = np.stack([db_codes(np.load(io.BytesIO(members[f'{channel}.npy']))[0]) for channel in CHANNELS])
    mask = np.load(io.BytesIO(members['mask.npy']))[0]
    valid = np.load(io.BytesIO(members['valid_mask.npy']))[0]
    labels = np.where((valid > 0) & (mask < INVALID_CLASS) & np.isfinite(mask), mask, IGNORE).astype('uint8')
    labels[(codes == 0).any(0)] = IGNORE
    meta = dict(
        key=sample_id,
        actid=str(info.get('actid')),
        aoiid=str(info.get('aoiid')),
        grid_id=str(info.get('grid_id')),
        flood_date=info.get('flood_date'),
        pflood=info.get('pflood'),
        pwater=info.get('pwater'),
        pcovered=info.get('pcovered'),
        gvalid=info.get('gvalid'),
        dates={name: source.get('source_date') for name, source in info.get('sources', {}).items()},
    )
    return codes, labels, meta


def write_chunk(samples, path):
    codes = np.stack([codes for codes, _, _ in samples])
    labels = np.stack([labels for _, labels, _ in samples])
    meta = [meta for _, _, meta in samples]
    np.savez_compressed(path, X=codes, Y=labels, meta=np.array(json.dumps(meta)))
    print(os.path.basename(path), len(samples), 'tiles', flush=True)


def convert_shard(path, out_dir):
    """Chunks <split>_<shard>_<chunk>.npz of a shard <split>/shard-<shard>.tar."""
    split = os.path.basename(os.path.dirname(os.path.abspath(path)))
    shard = int(os.path.basename(path).removeprefix('shard-').removesuffix('.tar'))
    chunk_index = 0
    samples = []
    for key, members in shard_samples(path):
        samples.append(converted_sample(members, f'{split}/{shard}/{key}'))
        if len(samples) == CHUNK_TILES:
            write_chunk(samples, os.path.join(out_dir, f'{split}_{shard:02d}_{chunk_index:03d}.npz'))
            chunk_index += 1
            samples = []
    if samples:
        write_chunk(samples, os.path.join(out_dir, f'{split}_{shard:02d}_{chunk_index:03d}.npz'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='output directory for the chunks')
    parser.add_argument('shards', nargs='+', help='train_GRD/shard-NNNNN.tar and test_GRD/shard-NNNNN.tar files')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for path in args.shards:
        convert_shard(path, args.out)


if __name__ == '__main__':
    main()
