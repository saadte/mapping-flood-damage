"""Rebuild the Sen1Floods11 hand-labelled train and valid chips as Kuro Siwo-format tiles from our own S1 processing."""

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import odc.stac
import planetary_computer
import pystac
import pystac_client
import rasterio
from odc.geo.geobox import GeoBox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import s1_change_rule
import s1_grd

BUCKET_URL = 'https://storage.googleapis.com/sen1floods11/v1.1'
CHIP_URL = f'{BUCKET_URL}/data/flood_events/HandLabeled'
STAC_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1'
SPLIT_FILES = {'train': 'flood_train_data.csv', 'valid': 'flood_valid_data.csv'}
LOCATION_NAMES = {'Mekong': 'Cambodia'}  # chip ids use Mekong, the metadata use Cambodia
SEARCH_DAYS_BEFORE = 75
PRE_GAP_DAYS = 12
GEOID_STEP_PX = 32
LOOKUP_STEP_PX = 8
LEE_WINDOW_PX = 7
TILE_PX = 224
MIN_LABELLED_SHARE = 0.3
WORKERS = 8


def event_metadata():
    """Sen1Floods11 event properties (S1 date, relative orbit, pass direction, VH threshold) by location."""
    with urllib.request.urlopen(f'{BUCKET_URL}/Sen1Floods11_Metadata.geojson') as response:
        features = json.load(response)['features']
    return {f['properties']['location']: f['properties'] for f in features}


def split_chips():
    """(split, chip id) for every hand-labelled train and valid chip, in the order of the split files."""
    chips = []
    for split, name in SPLIT_FILES.items():
        with urllib.request.urlopen(f'{BUCKET_URL}/splits/flood_handlabeled/{name}') as response:
            lines = response.read().decode().splitlines()
        chips += [(split, line.split(',')[0].replace('_S1Hand.tif', '')) for line in lines if line.strip()]
    return chips


def download(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path + '.part')
        os.replace(path + '.part', path)
    return path


def chip_file(chips_dir, chip, layer):
    """Local copy of a hand-labelled chip layer (S1Hand, LabelHand or JRCWaterHand)."""
    return download(f'{CHIP_URL}/{layer}/{chip}_{layer}.tif', os.path.join(chips_dir, f'{chip}_{layer}.tif'))


def catalog():
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def acquisition_plan(event, bbox):
    """Same-track S1 GRD items for the post date and the two latest dates at least PRE_GAP_DAYS earlier, or None."""
    event_date = dt.date.fromisoformat(event['s1_date'].replace('/', '-'))
    search = catalog().search(
        collections=['sentinel-1-grd'],
        bbox=bbox,
        datetime=f'{event_date - dt.timedelta(days=SEARCH_DAYS_BEFORE)}/{event_date + dt.timedelta(days=1)}',
    )
    items_by_day = {}
    for item in search.item_collection():
        if (
            item.properties.get('sat:relative_orbit') == int(event['rel_orbit_num'])
            and item.properties['sat:orbit_state'].upper() == event['orbit']
            and {'VV', 'VH'} <= set(item.properties.get('sar:polarizations', []))
        ):
            items_by_day.setdefault(item.properties['datetime'][:10], []).append(item)
    days = sorted(items_by_day)
    post = [d for d in days if abs((dt.date.fromisoformat(d) - event_date).days) <= 1]
    pre = [d for d in days if dt.date.fromisoformat(d) <= event_date - dt.timedelta(days=PRE_GAP_DAYS)]
    if not post or len(pre) < 2:
        return None
    dates = dict(post=post[0], pre1=pre[-1], pre2=pre[-2])
    return dict({role: [item.to_dict() for item in items_by_day[day]] for role, day in dates.items()}, dates=dates)


def mean_sigma0(item_dicts, transform, crs, shape, dem_ellipsoid_m):
    """(VV, VH) linear sigma0 averaged over the items of one date where both are valid, or None without coverage."""
    height, width = shape
    total = {pol: np.zeros(shape) for pol in ('vv', 'vh')}
    count = np.zeros(shape)
    for item_dict in item_dicts:
        item = planetary_computer.sign(pystac.Item.from_dict(item_dict))
        sigma0, _ = s1_grd.geocode_item(
            item, transform, crs, height, width, dem_ellipsoid_m, lookup_step_px=LOOKUP_STEP_PX, log=lambda *_: None
        )
        if sigma0['vv'] is None:
            continue
        valid = np.isfinite(sigma0['vv']) & np.isfinite(sigma0['vh'])
        for pol, pol_total in total.items():
            pol_total[valid] += sigma0[pol][valid]
        count += valid
    if count.max() == 0:
        return None
    return tuple(
        np.where(count > 0, total[pol] / np.maximum(count, 1), np.nan).astype('float32') for pol in ('vv', 'vh')
    )


def build_chip(job):
    """Codes [6, 512, 512], labels and metadata of one chip, cached as <chip>.npz; None if a date has no coverage."""
    split, chip, plan, chips_dir = job
    out = os.path.join(chips_dir, f'{chip}.npz')
    if os.path.exists(out):
        return out
    with rasterio.open(chip_file(chips_dir, chip, 'S1Hand')) as src:
        transform, crs, shape = src.transform, str(src.crs), (src.height, src.width)
    with rasterio.open(chip_file(chips_dir, chip, 'LabelHand')) as src:
        hand_label = src.read(1)
    with rasterio.open(chip_file(chips_dir, chip, 'JRCWaterHand')) as src:
        jrc_water = src.read(1)
    height, width = shape
    dem_items = (
        catalog()
        .search(
            collections=['cop-dem-glo-30'],
            bbox=[transform.c, transform.f + height * transform.e, transform.c + width * transform.a, transform.f],
        )
        .item_collection()
    )
    geobox = GeoBox(shape, transform, crs)
    dem = odc.stac.load(
        dem_items, bands=['data'], geobox=geobox, resampling='bilinear', dtype='float32', nodata=np.nan
    ).data.values[0]
    dem_ellipsoid_m, _ = s1_grd.dem_ellipsoid(dem, transform, crs, geoid_step_px=GEOID_STEP_PX)
    channels = []
    for role in ('post', 'pre1', 'pre2'):
        sigma0 = mean_sigma0(plan[role], transform, crs, shape, dem_ellipsoid_m)
        if sigma0 is None:
            return None
        for linear in sigma0:
            db = s1_change_rule.lee_db(linear, LEE_WINDOW_PX)
            codes = np.clip(np.round((db + 35.0) * 6.0), 1, 255).astype('uint8')
            codes[~np.isfinite(db)] = 0
            channels.append(codes)
    labels = np.full(shape, 255, 'uint8')
    labels[hand_label == 0] = 0
    labels[(hand_label == 1) & (jrc_water == 1)] = 1
    labels[(hand_label == 1) & (jrc_water != 1)] = 2
    meta = dict(split=split, chip=chip, actid='s1f11_' + chip.split('_')[0], dates=plan['dates'])
    np.savez_compressed(out, X=np.stack(channels), Y=labels, meta=json.dumps(meta))
    return out


def chip_jobs(chips, events, chips_dir):
    """Build jobs for the chips of every event with a same-track post and two pre dates."""
    chips_by_location = {}
    for split, chip in chips:
        chips_by_location.setdefault(chip.split('_')[0], []).append((split, chip))
    jobs = []
    for location, location_chips in chips_by_location.items():
        bounds = []
        for _, chip in location_chips:
            with rasterio.open(chip_file(chips_dir, chip, 'S1Hand')) as src:
                bounds.append(src.bounds)
        bbox = [
            min(b.left for b in bounds),
            min(b.bottom for b in bounds),
            max(b.right for b in bounds),
            max(b.top for b in bounds),
        ]
        plan = acquisition_plan(events[LOCATION_NAMES.get(location, location)], bbox)
        print(location, len(location_chips), 'chips, dates', plan and plan['dates'], flush=True)
        if plan:
            jobs += [(split, chip, plan, chips_dir) for split, chip in location_chips]
    return jobs


def write_tiles(chip_paths, split, out_path):
    """Four 224 px tiles per 512 px chip of the split; tiles with less than 30% labelled pixels are skipped."""
    codes, labels, metas = [], [], []
    for path in chip_paths:
        with np.load(path) as chip:
            meta = json.loads(str(chip['meta']))
            if meta['split'] != split:
                continue
            chip_codes, chip_labels = chip['X'], chip['Y']
        for y0 in (0, TILE_PX):
            for x0 in (0, TILE_PX):
                tile_labels = chip_labels[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX]
                if (tile_labels != 255).mean() < MIN_LABELLED_SHARE:
                    continue
                codes.append(chip_codes[:, y0 : y0 + TILE_PX, x0 : x0 + TILE_PX])
                labels.append(tile_labels)
                metas.append(dict(meta, aoiid=meta['chip'], grid_id=f'{y0}_{x0}', flood_date=meta['dates']['post']))
    np.savez_compressed(out_path, X=np.stack(codes), Y=np.stack(labels), meta=json.dumps(metas))
    print(split, 'tiles', len(codes), 'flood px share', float(np.mean([(y == 2).mean() for y in labels])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'out_dir', help='output directory; downloaded and processed chips are cached in <out_dir>/chips'
    )
    args = parser.parse_args()
    chips_dir = os.path.join(args.out_dir, 'chips')
    os.makedirs(chips_dir, exist_ok=True)
    jobs = chip_jobs(split_chips(), event_metadata(), chips_dir)
    start = time.time()
    chip_paths = []
    with ProcessPoolExecutor(min(WORKERS, os.cpu_count() or 4)) as executor:
        for i, path in enumerate(executor.map(build_chip, jobs)):
            if path:
                chip_paths.append(path)
            if i % 20 == 0:
                print('chips', i, 'of', len(jobs), f'{time.time() - start:.0f} s', flush=True)
    for split in SPLIT_FILES:
        write_tiles(chip_paths, split, os.path.join(args.out_dir, f's1f11_{split}.npz'))


if __name__ == '__main__':
    main()
