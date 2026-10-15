"""Reference flood maps of the South Asian hold-out development events on the event grid, for scoring only."""

import glob
import os
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd

import reference_scoring

HOLDOUT_DIR = f'{reference_scoring.REFERENCE_DIR}/new_events'

# bbox and date define the pipeline run; pkg is the extracted reference package that holds the gt and extent layers
EVENTS = {
    'nepal_terai_2021': dict(
        bbox=(84.85, 26.85, 85.45, 27.20),
        date='2021-06-29',
        ref_type='water',
        product='UNOSAT 3235, GLIDE FL20210630NPL',
        pkg='unosat3235_FL20210630NPL_SHP',
        gt=['S1_20210702_FloodExtent_Lumbini_Gandaki_Bagmati_Province2.shp'],
        extent=['S1_20210702_AnalysisExtent_Lumbini_Gandaki_Bagmati_Province2.shp'],
    ),
    'nepal_farwest_2020': dict(
        bbox=(80.85, 28.33, 81.33, 28.68),
        date='2020-07-28',
        ref_type='water',
        product='UNOSAT 2871, GLIDE FL20200708NPL',
        pkg='unosat2871_FL20200708NPL_SHP',
        gt=['ST1_20200729_FloodExtent_Bardiya_Kailali.shp'],
        extent=['ST1_20200729_AnalysisExtent_Bardiya_Kailali.shp'],
    ),
    'sunamganj_2022': dict(
        bbox=(91.61, 24.67, 91.99, 25.11),
        date='2022-06-15',
        ref_type='water (incl. persisting first-wave water)',
        product='UNOSAT 3303, GLIDE FL20220525BGD',
        pkg='unosat3303_FL20220525BGD_SHP',
        gt=['RCM1_20220618_FloodExtent_Sylhet.shp'],
        extent=['RCM1_20220618_AnalysisExtent_Sylhet.shp'],
    ),
    'panjshir_2018': dict(
        bbox=(69.64, 35.37, 69.82, 35.50),
        date='2018-07-12',
        ref_type='mixed (debris flow / landslide + flood water)',
        product='UNOSAT 2665, GLIDE LS20180712AFG',
        pkg='unosat2665_LS20180712AFG_SHP',
        gt=['ST20180713_LandslideExtent_Peshghor.shp', 'ST20180713_WaterExtent_Peshghor.shp'],
        extent=['Analysis_Extent.shp'],
    ),
    'afg_baghlanjadid_2024': dict(
        bbox=(68.62, 36.06, 68.88, 36.26),
        date='2024-05-10',
        ref_type='water (flash-flood water in a Hindu Kush valley town)',
        product='UNOSAT 3849, GLIDE FL20240512AFG',
        pkg='unosat3849_FL20240512AFG_SHP',
        gt=['PHR_20240514_FloodExtent_BanghlanCity.shp', 'PHR_20240514_FloodExtent_SouthBanghlan.shp'],
        extent=['PHR_20240514_AnalysisExtent_BanghlanCity.shp', 'PHR_20240514_AnalysisExtent_SouthBanghlan.shp'],
    ),
}


def extract_zips(ev):
    """Extract each reference zip of the event into a folder of the same name, once."""
    for path in glob.glob(f'{HOLDOUT_DIR}/{ev}/*.zip'):
        if not os.path.isdir(path[:-4]):
            with zipfile.ZipFile(path) as archive:
                archive.extractall(path[:-4])


def file_name(path):
    # zips written on Windows can carry backslashes in their member names
    return os.path.basename(path.replace('\\', '/'))


def find_layer(ev, name, package):
    paths = sorted(
        path
        for path in glob.glob(f'{HOLDOUT_DIR}/{ev}/**/*', recursive=True)
        if os.path.isfile(path) and file_name(path) == name and package in path
    )
    if not paths:
        raise FileNotFoundError(f'{ev}: {name} (package {package!r}) not found under {HOLDOUT_DIR}/{ev}')
    return paths[0]


def read_geometries(path, crs):
    """Non-empty geometries of a layer in crs; layers without a CRS are WGS84."""
    layer = gpd.read_file(path)
    layer = layer[layer.geometry.notna() & ~layer.geometry.is_empty]
    if layer.crs is None:
        layer = layer.set_crs(4326)
    return layer.to_crs(crs).geometry


def joined_layers(ev, names, package, crs):
    return gpd.GeoSeries(pd.concat([read_geometries(find_layer(ev, name, package), crs) for name in names]), crs=crs)


def layers(ev, crs):
    """(reference, extent) geometries of a hold-out event as GeoSeries in crs."""
    extract_zips(ev)
    event = EVENTS[ev]
    return joined_layers(ev, event['gt'], event['pkg'], crs), joined_layers(ev, event['extent'], event['pkg'], crs)


def mask(geometries, transform, shape):
    if geometries.empty:
        return np.zeros(shape, bool)
    return reference_scoring.rasterize_mask(geometries, transform, shape)


def ref_for(ev, transform, crs, shape):
    """(reference mask inside the extent, extent mask) of a hold-out event on its grid."""
    gt, extent = layers(ev, crs)
    extent_mask = mask(extent, transform, shape)
    return mask(gt, transform, shape) & extent_mask, extent_mask
