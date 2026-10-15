"""Sentinel-1 incidence angle and DEM-based layover and shadow masks on the event grid."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import griddata

from s1_grd import fetch_product_annotation


def parse_annotation(xml):
    """Platform heading (deg) and tie points (lon, lat, ellipsoid incidence deg) of a product annotation."""
    root = ET.fromstring(xml)
    heading_deg = float(root.find('.//generalAnnotation/productInformation/platformHeading').text)
    points = [
        [float(point.find(tag).text) for tag in ('longitude', 'latitude', 'incidenceAngle')]
        for point in root.findall('.//geolocationGrid/geolocationGridPointList/geolocationGridPoint')
    ]
    return dict(heading=heading_deg, points=np.array(points))


def fetch_annotation(item):
    """Heading and incidence tie points of a signed sentinel-1-grd item."""
    return parse_annotation(fetch_product_annotation(item, 'vv'))


def incidence_on_grid(annotation, lonlat):
    """Ellipsoid incidence angle (deg) at (lon, lat): linear between tie points, nearest tie point outside them."""
    points = annotation['points']
    incidence_deg = griddata(points[:, :2], points[:, 2], lonlat, method='linear')
    if np.isnan(incidence_deg).any():
        nearest_deg = griddata(points[:, :2], points[:, 2], lonlat, method='nearest')
        incidence_deg = np.where(np.isnan(incidence_deg), nearest_deg, incidence_deg)
    return incidence_deg


def shadow_sweep(dem, incidence_rad, pixel_m):
    """Shadow mask of a DEM whose rows run along ground range, near range first."""
    tan_elevation = np.tan(np.pi / 2 - incidence_rad)
    horizon = np.full(dem.shape[0], -np.inf, dtype='float64')
    shadow = np.zeros(dem.shape, bool)
    previous = dem[:, 0].astype('float64')
    for j in range(1, dem.shape[1]):
        horizon = np.maximum(horizon, previous) - pixel_m * tan_elevation[:, j]
        shadow[:, j] = dem[:, j] < horizon
        previous = dem[:, j]
    return shadow


def layover_sweep(dem, incidence_rad, pixel_m):
    """Layover mask of a DEM whose rows run along ground range: slant range fails to increase strictly."""
    ground_range_m = np.arange(dem.shape[1], dtype='float64')[None, :] * pixel_m
    slant_range_m = ground_range_m * np.sin(incidence_rad) - dem * np.cos(incidence_rad)
    running_max = np.maximum.accumulate(slant_range_m, axis=1)
    running_min_from_far = np.minimum.accumulate(slant_range_m[:, ::-1], axis=1)[:, ::-1]
    layover = np.zeros(dem.shape, bool)
    layover[:, 1:] |= slant_range_m[:, 1:] <= running_max[:, :-1]
    layover[:, :-1] |= slant_range_m[:, :-1] >= running_min_from_far[:, 1:]
    return layover


def rotate_back(mask, angle_deg, shape):
    """Undo the ground-range rotation of a mask and crop it to the original grid shape."""
    rotated = ndi.rotate(mask.astype('float32'), -angle_deg, reshape=True, order=0, mode='constant', cval=0)
    row0 = (rotated.shape[0] - shape[0]) // 2
    col0 = (rotated.shape[1] - shape[1]) // 2
    return rotated[row0 : row0 + shape[0], col0 : col0 + shape[1]] > 0.5


def layover_shadow(dem, pixel_m, heading_deg, incidence_deg):
    """Layover and shadow masks of a north-up DEM grid seen by a right-looking sensor."""
    look_azimuth_deg = heading_deg + 90.0  # ground-range direction of a right-looking sensor
    angle_deg = look_azimuth_deg - 90.0  # rotation of the look direction from east (+x)
    nodata_mask = ~np.isfinite(dem)
    filled = np.where(nodata_mask, np.nanmin(dem), dem).astype('float32')
    incidence = np.broadcast_to(np.asarray(incidence_deg, dtype='float32'), dem.shape)
    rotated_dem = ndi.rotate(filled, angle_deg, reshape=True, order=1, mode='nearest')
    rotated_incidence = ndi.rotate(np.ascontiguousarray(incidence), angle_deg, reshape=True, order=1, mode='nearest')
    valid_mask = ndi.rotate(np.ones_like(filled), angle_deg, reshape=True, order=0, mode='constant', cval=0) > 0.5
    incidence_rad = np.radians(rotated_incidence)
    shadow = shadow_sweep(rotated_dem, incidence_rad, pixel_m) & valid_mask
    layover = layover_sweep(rotated_dem, incidence_rad, pixel_m) & valid_mask
    layover_mask = rotate_back(layover, angle_deg, dem.shape) & ~nodata_mask
    shadow_mask = rotate_back(shadow, angle_deg, dem.shape) & ~nodata_mask
    return layover_mask, shadow_mask
