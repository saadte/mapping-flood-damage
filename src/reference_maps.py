"""Reference flood maps and damaged buildings, roads and bridges of the development events, for scoring only."""

import glob
import os
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine

import emsr927_scoring
import reference_scoring

REFERENCE_DIR = reference_scoring.REFERENCE_DIR
UNOSAT_MELAMCHI_KASKI = f'{REFERENCE_DIR}/unosat_melamchi_3234/**'
UNOSAT_KATHMANDU = f'{REFERENCE_DIR}/unosat_ktm_3996/**'
UNOSAT_BAGHLAN_GDB = f'{REFERENCE_DIR}/unosat_baghlan_3848/**/FL20240512AFG.gdb'
SHOVI_FLOW = f'{REFERENCE_DIR}/unosat_shovi_3653/**/PHR_20230806_LandslideExtent_OniDistrict.shp'
ARSTANBAP_FLOW = f'{REFERENCE_DIR}/unosat_arstanbap_3907/**/WV3_20240727_MudSlideExtent_Arstanbap.shp'
FLOW_BUFFER_M = 1000
# EMS products with one folder per area of interest, shapefiles only
EMS_AOI_FOLDERS = {
    'alex': ['emsr467_aoi04', 'emsr467_aoi05', 'emsr468_aoi03'],
    'slovenia': ['emsr680_aoi05', 'emsr680_aoi06', 'emsr680_aoi07', 'emsr680_aoi14', 'emsr680_aoi15'],
}
# newer EMS products: grading and monitoring products under one root, shapefiles or GeoJSON
EMS_PRODUCT_ROOTS = {
    'jablanica': f'{REFERENCE_DIR}/ems_emsr770_jablanica',
    'ahr': f'{REFERENCE_DIR}/ems_emsr517_ahr',
    'derna': f'{REFERENCE_DIR}/ems_emsr696_derna',
}
EMS_FLOOD_LAYERS = ('areaOfInterestA', 'notAnalysedA', 'observedEventA')
EMS_DAMAGE_LAYERS = ('builtUpP', 'transportationL', 'transportationP')
BRIDGE_LAYERS = ('transportationL', 'transportationP')
DAMAGE_GRADES = (
    'destroyed',
    'damaged',
    'possibly damaged',
    'moderately damaged',
    'highly damaged',
    'completely destroyed',
    'negligible to slight damage',
    'affected',
    'possibly affected',
)


@dataclass
class Grid:
    transform: Affine
    crs: object
    shape: tuple[int, int]

    def mask(self, layer):
        return reference_scoring.rasterize_mask(layer, self.transform, self.shape)

    def file_mask(self, pattern):
        """Mask of the first file (sorted) matching a recursive glob pattern."""
        return self.mask(read_first(pattern, self.crs))


def read_layer(path, crs):
    return gpd.read_file(path).to_crs(crs)


def read_first(pattern, crs):
    """The first file (sorted) matching a recursive glob pattern."""
    return read_layer(sorted(glob.glob(pattern, recursive=True))[0], crs)


def baghlan_gdb():
    return glob.glob(UNOSAT_BAGHLAN_GDB, recursive=True)[0]


def ems_folder_products(folders, kinds):
    """{layer kind: shapefiles} for each area-of-interest folder."""
    return [
        {kind: sorted(glob.glob(f'{REFERENCE_DIR}/{folder}/**/*_{kind}_*.shp', recursive=True)) for kind in kinds}
        for folder in folders
    ]


def ems_product_files(product_dir, kind):
    """Shapefiles of a layer kind in a product folder, else its GeoJSON files."""
    return sorted(glob.glob(f'{product_dir}/*_{kind}_*.shp')) or sorted(glob.glob(f'{product_dir}/*_{kind}_*.json'))


def ems_root_products(root, kinds):
    """{layer kind: files} for each product folder (one with an area of interest) under root."""
    area_files = glob.glob(f'{root}/**/*areaOfInterestA*', recursive=True)
    product_dirs = sorted({os.path.dirname(path) for path in area_files if path.endswith(('.shp', '.json'))})
    return [{kind: ems_product_files(product_dir, kind) for kind in kinds} for product_dir in product_dirs]


def trishuli_flood(grid):
    extent, gt = reference_scoring.rasterize_ref(
        reference_scoring.load_unosat(crs=str(grid.crs)), grid.transform, grid.shape
    )
    return gt, extent


def melamchi_flood(grid):
    extent = grid.file_mask(f'{UNOSAT_MELAMCHI_KASKI}/S2_20210624_AnalysisExtent_Melamchi.shp')
    return grid.file_mask(f'{UNOSAT_MELAMCHI_KASKI}/S2_20210624_LandslideExtent_Melamchi.shp') & extent, extent


def kaski_flood(grid):
    analysed = grid.file_mask(f'{UNOSAT_MELAMCHI_KASKI}/S2_20210704_AnalysisExtent_Kaski_Syangja.shp')
    extent = analysed & ~grid.file_mask(f'{UNOSAT_MELAMCHI_KASKI}/S2_20210704_Cloudobstruction_Kaski_Syangja.shp')
    return grid.file_mask(f'{UNOSAT_MELAMCHI_KASKI}/S2_20210704_FloodExtent_Kaski_Syangja.shp') & extent, extent


def kathmandu_flood(grid):
    extent = grid.file_mask(f'{UNOSAT_KATHMANDU}/Spot6_20240930_AnalysisExtent_Kathmandu.shp')
    return grid.file_mask(f'{UNOSAT_KATHMANDU}/Spot6_20240930_FloodExtent_Kathmandu.shp') & extent, extent


def buffered_flow(grid, pattern):
    """A mapped flow, with a 1 km buffer around it as the extent: no analysis extent was published."""
    flow = read_first(pattern, grid.crs)
    extent = grid.mask(flow.buffer(FLOW_BUFFER_M))
    return grid.mask(flow) & extent, extent


def shovi_flood(grid):
    return buffered_flow(grid, SHOVI_FLOW)


def arstanbap_flood(grid):
    return buffered_flow(grid, ARSTANBAP_FLOW)


def baghlan_flood(grid):
    gdb = baghlan_gdb()
    extent = grid.mask(gpd.read_file(gdb, layer='S2_20240515_AnalysisExtent_BurkaDistrict').to_crs(grid.crs))
    mudflow = grid.mask(gpd.read_file(gdb, layer='S2_20240515_MudflowExtent_BurkaDistrict').to_crs(grid.crs))
    return mudflow & extent, extent


def ems_flood(grid, products):
    """Union over EMS products of the analysed area, and of the observed event inside it."""
    extent = np.zeros(grid.shape, bool)
    gt = np.zeros(grid.shape, bool)
    for files in products:
        analysed = np.zeros(grid.shape, bool)
        for path in files['areaOfInterestA']:
            analysed |= grid.mask(read_layer(path, grid.crs))
        for path in files['notAnalysedA']:
            analysed &= ~grid.mask(read_layer(path, grid.crs))
        for path in files.get('imageFootprintA', []):
            footprints = read_layer(path, grid.crs)
            if 'obj_type' in footprints.columns:
                not_analysed = footprints[footprints['obj_type'].astype(str).str.startswith('Not Analysed')]
                if len(not_analysed):
                    analysed &= ~grid.mask(not_analysed)
        observed = np.zeros(grid.shape, bool)
        for path in files['observedEventA']:
            observed |= grid.mask(read_layer(path, grid.crs))
        extent |= analysed
        gt |= observed & analysed
    return gt, extent


FLOOD_REFERENCES = {
    'trishuli': trishuli_flood,
    'melamchi': melamchi_flood,
    'kaski': kaski_flood,
    'kathmandu': kathmandu_flood,
    'shovi': shovi_flood,
    'baghlan': baghlan_flood,
    'arstanbap': arstanbap_flood,
}


def ref_for(ev, transform, crs, shape):
    """(reference flood mask, analysed extent mask) of a development event on its grid; None for other events."""
    grid = Grid(transform, crs, shape)
    for prefix, reference in FLOOD_REFERENCES.items():
        if ev.startswith(prefix):
            return reference(grid)
    for prefix, folders in EMS_AOI_FOLDERS.items():
        if ev.startswith(prefix):
            return ems_flood(grid, ems_folder_products(folders, EMS_FLOOD_LAYERS))
    for prefix, root in EMS_PRODUCT_ROOTS.items():
        if ev.startswith(prefix):
            return ems_flood(grid, ems_root_products(root, (*EMS_FLOOD_LAYERS, 'imageFootprintA')))
    return None


def damaged_geometries(layer):
    """Geometries of the features graded as damaged or affected; all features if the layer has no damage grade."""
    grade = next((c for c in layer.columns if c.lower().startswith('damage_gr')), None)
    if grade is not None:
        layer = layer[layer[grade].astype(str).str.lower().str.strip().isin(DAMAGE_GRADES)]
    return layer[['geometry']]


def damaged_bridges(layer):
    """Geometries of the bridges (EMS object type 214) graded as damaged, or None."""
    if layer is None or not len(layer):
        return None
    object_type = next((c for c in layer.columns if c.lower() == 'obj_type'), None)
    grade = next((c for c in layer.columns if c.lower().startswith('damage_gr')), None)
    if object_type is None:
        return None
    is_damaged_bridge = layer[object_type].astype(str).str.contains('214|Bridge', case=False, regex=True)
    if grade is not None:
        is_damaged_bridge &= layer[grade].astype(str).str.lower().str.strip().isin(DAMAGE_GRADES)
    return layer[is_damaged_bridge][['geometry']] if is_damaged_bridge.any() else None


def combined(frames, crs, drop_duplicates=False):
    if not frames:
        return None
    frame = gpd.GeoDataFrame(pd.concat(frames), geometry='geometry', crs=crs)
    return frame.drop_duplicates() if drop_duplicates else frame


def trishuli_damage(crs):
    ems = emsr927_scoring.load_ems(crs=str(crs))
    buildings = [damaged_geometries(layers['builtUpP']) for layers in ems.values() if 'builtUpP' in layers]
    roads = [damaged_geometries(layers['transportationL']) for layers in ems.values() if 'transportationL' in layers]
    bridges = (damaged_bridges(layers.get(kind)) for layers in ems.values() for kind in BRIDGE_LAYERS)
    return dict(
        buildings=combined(buildings, crs),
        roads=combined(roads, crs),
        bridges=combined([b for b in bridges if b is not None], crs),
    )


def melamchi_damage(crs):
    return dict(buildings=read_first(f'{UNOSAT_MELAMCHI_KASKI}/S2_20210624_DamagedPoint_Melamchi.shp', crs), roads=None)


def kathmandu_damage(crs):
    buildings = read_first(f'{UNOSAT_KATHMANDU}/Spot6_20240930_PotentiallyAffectedBuilding_Kathmandu.shp', crs)
    return dict(buildings=buildings, roads=None)


def baghlan_damage(crs):
    gdb = baghlan_gdb()
    return dict(
        buildings=gpd.read_file(gdb, layer='PHR_20240514_AffectedStructure_Burka').to_crs(crs),
        roads=gpd.read_file(gdb, layer='PHR_20240514_AffectedRoad_Burka').to_crs(crs),
    )


def ems_damage(products, crs, drop_duplicates):
    """Damaged buildings, roads and bridges from EMS products given as {layer kind: files}."""
    buildings = [damaged_geometries(read_layer(p, crs)) for files in products for p in files['builtUpP']]
    roads = [damaged_geometries(read_layer(p, crs)) for files in products for p in files['transportationL']]
    bridges = (damaged_bridges(read_layer(p, crs)) for files in products for kind in BRIDGE_LAYERS for p in files[kind])
    return dict(
        buildings=combined(buildings, crs, drop_duplicates),
        roads=combined(roads, crs, drop_duplicates),
        bridges=combined([b for b in bridges if b is not None], crs, drop_duplicates),
    )


DAMAGE_REFERENCES = {
    'trishuli': trishuli_damage,
    'melamchi': melamchi_damage,
    'kathmandu': kathmandu_damage,
    'baghlan': baghlan_damage,
}


def damage_refs(ev, crs):
    """Reference damaged buildings, roads and bridges of a development event; {} for events without them."""
    for prefix, reference in DAMAGE_REFERENCES.items():
        if ev.startswith(prefix):
            return reference(crs)
    for prefix, folders in EMS_AOI_FOLDERS.items():
        if ev.startswith(prefix):
            return ems_damage(ems_folder_products(folders, EMS_DAMAGE_LAYERS), crs, drop_duplicates=False)
    for prefix, root in EMS_PRODUCT_ROOTS.items():
        if ev.startswith(prefix):
            return ems_damage(ems_root_products(root, EMS_DAMAGE_LAYERS), crs, drop_duplicates=True)
    return {}
