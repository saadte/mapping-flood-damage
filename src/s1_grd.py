"""Calibration, thermal noise removal and Range-Doppler terrain correction of Sentinel-1 GRD scenes."""

import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pyproj
import rasterio
import requests
from pyproj import Transformer
from pyproj.transformer import TransformerGroup
from rasterio.windows import Window
from scipy import ndimage as ndi

POLS = ('vv', 'vh')
SPEED_OF_LIGHT_M_S = 299792458.0
WGS84_A_M = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
GEOID_GRID = 'us_nga_egm08_25'  # EGM2008 on a 2.5' grid, about 80 MB
TRANSIENT_HTTP_STATUS = (429, 500, 502, 503, 504)
HTTP_ATTEMPTS = 5
ORBIT_DEGREE = 5  # a degree-5 fit over a few minutes of state vectors is accurate to millimetres
LOOKUP_STEP_PX = 16
LEVEL_STEP_M = 400.0
GEOID_STEP_PX = 64
BLOCK_ROWS = 1024


def fetch_text(url):
    """Body of a GET request, retried on connection errors and transient HTTP status codes."""
    for attempt in range(HTTP_ATTEMPTS):
        try:
            response = requests.get(url, timeout=120)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            response = None
        if response is not None and response.status_code not in TRANSIENT_HTTP_STATUS:
            if response.status_code != 200:
                raise RuntimeError(f'GET {url.split("?")[0]} returned HTTP {response.status_code}')
            return response.text
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f'GET {url.split("?")[0]} failed {HTTP_ATTEMPTS} times')


def fetch_product_annotation(item, pol):
    """Product annotation XML of a signed sentinel-1-grd item, which sits next to manifest.safe."""
    base, _, query = item.assets['safe-manifest'].href.partition('?')
    xml = fetch_text(f'{base.rsplit("/", 1)[0]}/annotation/iw-{pol}.xml?{query}')
    if 'geolocationGrid' not in xml:
        raise RuntimeError('no product annotation for ' + item.id)
    return xml


def annotation_time(text):
    return np.datetime64(text.strip()[:26], 'ns')


def float_at(element, path):
    return float(element.find(path).text)


def array_at(element, path, dtype=float):
    return np.array(element.find(path).text.split(), dtype=dtype)


def seconds_since(element, path, start):
    return (annotation_time(element.find(path).text) - start) / np.timedelta64(1, 's')


def parse_product(xml):
    """Image size and timing, orbit state vectors, ground-range polynomials and tie points of a product annotation."""
    root = ET.fromstring(xml)
    image = root.find('.//imageAnnotation/imageInformation')
    first_line_time = annotation_time(image.find('productFirstLineUtcTime').text)
    orbits = root.findall('.//generalAnnotation/orbitList/orbit')
    conversions = root.findall('.//coordinateConversion/coordinateConversionList/coordinateConversion')
    tie_points = root.findall('.//geolocationGrid/geolocationGridPointList/geolocationGridPoint')
    product = dict(
        azimuth_interval_s=float_at(image, 'azimuthTimeInterval'),
        range_spacing_m=float_at(image, 'rangePixelSpacing'),
        line_count=int(image.find('numberOfLines').text),
        pixel_count=int(image.find('numberOfSamples').text),
        orbit_time_s=np.array([seconds_since(orbit, 'time', first_line_time) for orbit in orbits]),
        orbit_position_m=np.array([[float_at(orbit, f'position/{axis}') for axis in 'xyz'] for orbit in orbits]),
        orbit_velocity_m_s=np.array([[float_at(orbit, f'velocity/{axis}') for axis in 'xyz'] for orbit in orbits]),
        srgr_time_s=np.array([seconds_since(record, 'azimuthTime', first_line_time) for record in conversions]),
        srgr_origin_m=np.array([float_at(record, 'sr0') for record in conversions]),
        srgr_coefficients=[array_at(record, 'srgrCoefficients') for record in conversions],
        tie_points=np.array(
            [
                [
                    seconds_since(point, 'azimuthTime', first_line_time),
                    float_at(point, 'slantRangeTime'),
                    float_at(point, 'line'),
                    float_at(point, 'pixel'),
                    float_at(point, 'latitude'),
                    float_at(point, 'longitude'),
                    float_at(point, 'height'),
                ]
                for point in tie_points
            ]
        ),
    )
    product['tie_point_residuals'] = tie_point_residuals(product)
    return product


def parse_calibration(xml):
    root = ET.fromstring(xml)
    vectors = root.findall('.//calibrationVectorList/calibrationVector')
    return dict(
        line=np.array([int(vector.find('line').text) for vector in vectors]),
        pixel=[array_at(vector, 'pixel', int) for vector in vectors],
        sigma=[array_at(vector, 'sigmaNought') for vector in vectors],
    )


def parse_noise(xml):
    """Range noise vectors and azimuth noise blocks of a noise annotation."""
    root = ET.fromstring(xml)
    # products processed before IPF 2.9 have a single noise vector list and no azimuth noise
    range_vectors = root.findall('.//noiseRangeVectorList/noiseRangeVector') or root.findall(
        './/noiseVectorList/noiseVector'
    )
    range_noise = dict(
        line=np.array([int(vector.find('line').text) for vector in range_vectors]),
        pixel=[array_at(vector, 'pixel', int) for vector in range_vectors],
        lut=[
            array_at(vector, 'noiseRangeLut' if vector.find('noiseRangeLut') is not None else 'noiseLut')
            for vector in range_vectors
        ],
    )
    azimuth_blocks = [
        dict(
            first_line=int(vector.find('firstAzimuthLine').text),
            last_line=int(vector.find('lastAzimuthLine').text),
            first_pixel=int(vector.find('firstRangeSample').text),
            last_pixel=int(vector.find('lastRangeSample').text),
            line=array_at(vector, 'line', int),
            lut=array_at(vector, 'noiseAzimuthLut'),
        )
        for vector in root.findall('.//noiseAzimuthVectorList/noiseAzimuthVector')
    ]
    return dict(range=range_noise, azimuth=azimuth_blocks)


def geodetic_to_ecef(lon, lat, height_m):
    lon_rad, lat_rad = np.radians(lon), np.radians(lat)
    n = WGS84_A_M / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)
    return np.stack(
        [
            (n + height_m) * np.cos(lat_rad) * np.cos(lon_rad),
            (n + height_m) * np.cos(lat_rad) * np.sin(lon_rad),
            (n * (1 - WGS84_E2) + height_m) * np.sin(lat_rad),
        ],
        -1,
    )


class Orbit:
    """Polynomial fit of the annotation state vectors; time in s from the first line."""

    def __init__(self, time_s, position_m, velocity_m_s):
        near = (time_s > -60) & (time_s < 90)
        if near.sum() < ORBIT_DEGREE + 2:  # take the 8 state vectors nearest to mid-scene
            near = np.argsort(np.abs(time_s - 15))[:8]
        time_s, position_m, velocity_m_s = time_s[near], position_m[near], velocity_m_s[near]
        self.mean_time_s = time_s.mean()
        offset_s = time_s - self.mean_time_s
        self.position_fit = [np.polyfit(offset_s, position_m[:, k], ORBIT_DEGREE) for k in range(3)]
        self.velocity_fit = [np.polyfit(offset_s, velocity_m_s[:, k], ORBIT_DEGREE) for k in range(3)]
        self.acceleration_fit = [np.polyder(fit) for fit in self.velocity_fit]
        fitted_m = np.stack([np.polyval(self.position_fit[k], offset_s) for k in range(3)], -1)
        self.fit_error_m = float(np.abs(fitted_m - position_m).max())

    def state(self, time_s):
        """Position, velocity and acceleration of the satellite at time_s."""
        offset_s = time_s - self.mean_time_s
        return tuple(
            np.stack([np.polyval(fit, offset_s) for fit in fits], -1)
            for fits in (self.position_fit, self.velocity_fit, self.acceleration_fit)
        )


def zero_doppler(orbit, position_m, initial_time_s):
    """Zero-Doppler time (s from the first line) and slant range (m) of ECEF points, by Newton iteration."""
    time_s = np.full(position_m.shape[:-1], initial_time_s, 'float64')
    for _ in range(8):
        satellite_m, velocity_m_s, acceleration = orbit.state(time_s)
        offset_m = position_m - satellite_m
        doppler = (offset_m * velocity_m_s).sum(-1)
        doppler_rate = -(velocity_m_s * velocity_m_s).sum(-1) + (offset_m * acceleration).sum(-1)
        time_s = time_s - doppler / doppler_rate
    satellite_m = orbit.state(time_s)[0]
    return time_s, np.linalg.norm(position_m - satellite_m, axis=-1)


def srgr_pixel(product, time_s, slant_range_m):
    """Range pixel from the ground-range polynomials, interpolated linearly in azimuth time between records."""
    record_time_s = product['srgr_time_s']
    origin_m = product['srgr_origin_m']
    coefficients = product['srgr_coefficients']
    if len(record_time_s) == 1:
        ground_range_m = np.polynomial.polynomial.polyval(slant_range_m - origin_m[0], coefficients[0])
    else:
        i = np.clip(np.searchsorted(record_time_s, time_s) - 1, 0, len(record_time_s) - 2)
        w = np.clip((time_s - record_time_s[i]) / (record_time_s[i + 1] - record_time_s[i]), 0, 1)
        before = np.zeros_like(slant_range_m)
        after = np.zeros_like(slant_range_m)
        for k in np.unique(i):
            m = i == k
            before[m] = np.polynomial.polynomial.polyval(slant_range_m[m] - origin_m[k], coefficients[k])
            after[m] = np.polynomial.polynomial.polyval(slant_range_m[m] - origin_m[k + 1], coefficients[k + 1])
        ground_range_m = before * (1 - w) + after * w
    return ground_range_m / product['range_spacing_m']


def tie_point_residuals(product):
    """Tie-point lines and, per line, the tie-point pixel minus the srgr pixel as a function of slant range."""
    points = product['tie_points']
    lines = np.unique(points[:, 2])
    residuals = []
    for line in lines:
        row = points[points[:, 2] == line]
        row = row[np.argsort(row[:, 1])]
        slant_range_m = row[:, 1] * SPEED_OF_LIGHT_M_S / 2
        residuals.append((slant_range_m, row[:, 3] - srgr_pixel(product, row[:, 0], slant_range_m)))
    return lines, residuals


def tie_point_correction(product, line, slant_range_m):
    """Pixel correction interpolated between tie-point rows in line and along each row in slant range."""
    # the residuals hold the height steps between IW sub-swaths that one srgr polynomial misses (5 px in the Himalaya)
    lines, residuals = product['tie_point_residuals']
    i = np.clip(np.searchsorted(lines, line) - 1, 0, len(lines) - 2)
    w = np.clip((line - lines[i]) / (lines[i + 1] - lines[i]), 0, 1)
    correction = np.empty_like(slant_range_m)
    for k in np.unique(i):
        m = i == k
        correction[m] = (
            np.interp(slant_range_m[m], *residuals[k]) * (1 - w[m])
            + np.interp(slant_range_m[m], *residuals[k + 1]) * w[m]
        )
    return correction


def range_doppler(product, orbit, lon, lat, height_m):
    """Zero-Doppler time (s), slant range (m) and image line of points given by lon, lat and ellipsoidal height."""
    position_m = geodetic_to_ecef(lon, lat, height_m)
    initial_time_s = 0.5 * product['line_count'] * product['azimuth_interval_s']
    time_s, slant_range_m = zero_doppler(orbit, position_m, initial_time_s)
    return time_s, slant_range_m, time_s / product['azimuth_interval_s']


def radar_coords(product, orbit, lon, lat, height_m):
    """Image (line, pixel) of points given by lon, lat and ellipsoidal height."""
    time_s, slant_range_m, line = range_doppler(product, orbit, lon, lat, height_m)
    pixel = srgr_pixel(product, time_s, slant_range_m) + tie_point_correction(product, line, slant_range_m)
    return line, pixel


def tie_point_errors(product, orbit):
    """Largest line and pixel errors (px) when the annotation tie points are geocoded again."""
    points = product['tie_points']
    time_s, slant_range_m, line = range_doppler(product, orbit, points[:, 5], points[:, 4], points[:, 6])
    srgr = srgr_pixel(product, time_s, slant_range_m)
    pixel = srgr + tie_point_correction(product, line, slant_range_m)
    return dict(
        gcp_pix_err_hybrid=float(np.abs(pixel - points[:, 3]).max()),
        gcp_line_err=float(np.abs(line - points[:, 2]).max()),
        gcp_pix_err=float(np.abs(srgr - points[:, 3]).max()),
    )


def geoid_grid_path():
    """Path of the EGM2008 grid in PROJ's user or data directories, or None."""
    directories = [pyproj.datadir.get_user_data_dir(), *pyproj.datadir.get_data_dir().split(os.pathsep)]
    for directory in directories:
        path = os.path.join(directory, GEOID_GRID + '.tif')
        if os.path.exists(path):
            return path
    return None


def sync_geoid_grid():
    """Download the EGM2008 grid into PROJ's user data directory; True when the grid is then on disk."""
    command = [sys.executable, '-m', 'pyproj', 'sync', '--file', GEOID_GRID]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
    except subprocess.TimeoutExpired:
        print('geoid sync: no result after 900 s', flush=True)
        return False
    print('geoid sync: exit', result.returncode, (result.stdout + result.stderr).strip()[-200:], flush=True)
    if result.returncode != 0 or geoid_grid_path() is None:
        return False
    pyproj.datadir.append_data_dir(pyproj.datadir.get_user_data_dir())  # lets the running PROJ context find the grid
    return True


def egm2008_undulation(lon, lat):
    """EGM2008 undulation (m) and the transformation used, or None when PROJ has no working EGM2008 grid."""
    for transformer in TransformerGroup('EPSG:4326+3855', 'EPSG:4979', always_xy=True).transformers:
        if 'ballpark' in transformer.description.lower():  # a ballpark transformation returns N = 0
            continue
        _, _, undulation_m = transformer.transform(lon, lat, np.zeros_like(lon))
        undulation_m = np.asarray(undulation_m, 'float64')
        if np.isfinite(undulation_m).all() and 0 < np.abs(undulation_m).max() < 150:
            return undulation_m, f'EPSG:4326+3855 ({transformer.description[:60]})'
    return None


def geoid_undulation(lon, lat, sync=True):
    """EGM2008 undulation N (m) with h_ellipsoid = h_CopDEM + N, and its transformation; sync fetches a missing grid."""
    pyproj.network.set_network_enabled(True)
    undulation = egm2008_undulation(lon, lat)
    if undulation is None and sync and geoid_grid_path() is None:
        print('EGM2008 geoid unavailable: downloading the grid', flush=True)
        if sync_geoid_grid():
            undulation = egm2008_undulation(lon, lat)
    if undulation is None:
        raise RuntimeError(f'no EGM2008 geoid in PROJ; run: python -m pyproj sync --file {GEOID_GRID}')
    return undulation


def coarse_axis(size, step):
    """Every step-th index of an axis, plus the last one."""
    return np.unique(np.r_[np.arange(0, size, step), size - 1])


def lonlat_of_pixels(transform, crs, rows, cols):
    """Longitude and latitude of pixel centres of a north-up grid."""
    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e
    return Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(x, y)


def dem_ellipsoid(dem, transform, crs, geoid_step_px=GEOID_STEP_PX):
    """Copernicus DEM (EGM2008 heights) as ellipsoidal heights, with N interpolated from a coarse grid."""
    height, width = dem.shape
    rows = coarse_axis(height, geoid_step_px)
    cols = coarse_axis(width, geoid_step_px)
    col_grid, row_grid = np.meshgrid(cols, rows)
    lon, lat = lonlat_of_pixels(transform, crs, row_grid, col_grid)
    undulation_m, model = geoid_undulation(np.asarray(lon, 'float64'), np.asarray(lat, 'float64'))
    row_index = np.interp(np.arange(height), rows, np.arange(len(rows)))
    col_index = np.interp(np.arange(width), cols, np.arange(len(cols)))
    fine_undulation_m = ndi.map_coordinates(
        undulation_m, np.meshgrid(row_index, col_index, indexing='ij'), order=1, mode='nearest'
    ).astype('float32')
    info = dict(geoid=model, N_min=float(undulation_m.min()), N_max=float(undulation_m.max()))
    return dem + fine_undulation_m, info


def build_lookup(product, orbit, transform, crs, height, width, dem_ellipsoid_m, step_px):
    """Exact (line, pixel) on a coarse grid of rows, columns and height levels spanning the DEM."""
    rows = coarse_axis(height, step_px)
    cols = coarse_axis(width, step_px)
    col_grid, row_grid = np.meshgrid(cols, rows)
    lon, lat = lonlat_of_pixels(transform, crs, row_grid, col_grid)
    low_m, high_m = np.nanmin(dem_ellipsoid_m) - 50, np.nanmax(dem_ellipsoid_m) + 50
    levels_m = np.arange(
        np.floor(low_m / LEVEL_STEP_M) * LEVEL_STEP_M,
        np.ceil(high_m / LEVEL_STEP_M) * LEVEL_STEP_M + LEVEL_STEP_M / 2,
        LEVEL_STEP_M,
    )
    line = np.empty((len(levels_m), *lon.shape), 'float64')
    pixel = np.empty_like(line)
    for k, level_m in enumerate(levels_m):
        line[k], pixel[k] = radar_coords(product, orbit, lon, lat, np.full(lon.shape, level_m))
    return dict(rows=rows, cols=cols, levels_m=levels_m, line=line, pixel=pixel)


def interpolate_lookup(lookup, rows, cols, height_m):
    """Trilinear interpolation of the lookup at fine-grid rows, columns and ellipsoidal heights."""
    row_index = np.interp(rows, lookup['rows'], np.arange(len(lookup['rows'])))
    col_index = np.interp(cols, lookup['cols'], np.arange(len(lookup['cols'])))
    levels_m = lookup['levels_m']
    level_index = (height_m - levels_m[0]) / (levels_m[1] - levels_m[0])
    coords = np.stack([level_index, row_index, col_index])
    return (
        ndi.map_coordinates(lookup['line'], coords, order=1, mode='nearest'),
        ndi.map_coordinates(lookup['pixel'], coords, order=1, mode='nearest'),
    )


def scene_crop(lookup, product):
    """Event-grid rows and columns (row0, row1, col0, col1) of lookup nodes that fall in the scene at any height."""
    node_mask = (
        (lookup['line'] > -2)
        & (lookup['line'] < product['line_count'] + 1)
        & (lookup['pixel'] > -2)
        & (lookup['pixel'] < product['pixel_count'] + 1)
    ).any(0)
    if not node_mask.any():
        return None
    node_rows, node_cols = np.nonzero(node_mask)
    rows, cols = lookup['rows'], lookup['cols']
    return (
        int(rows[max(node_rows.min() - 1, 0)]),
        int(rows[min(node_rows.max() + 1, len(rows) - 1)]) + 1,
        int(cols[max(node_cols.min() - 1, 0)]),
        int(cols[min(node_cols.max() + 1, len(cols) - 1)]) + 1,
    )


def crop_radar_coords(lookup, dem_ellipsoid_m, crop):
    """(line, pixel) of every pixel of the crop; DEM gaps take the median height."""
    row0, row1, col0, col1 = crop
    fill_m = float(np.nanmedian(dem_ellipsoid_m))
    line = np.empty((row1 - row0, col1 - col0), 'float32')
    pixel = np.empty_like(line)
    col_axis = np.arange(col0, col1, dtype='float64')
    for start in range(row0, row1, BLOCK_ROWS):
        stop = min(row1, start + BLOCK_ROWS)
        row_grid, col_grid = np.meshgrid(np.arange(start, stop, dtype='float64'), col_axis, indexing='ij')
        height_m = np.nan_to_num(dem_ellipsoid_m[start:stop, col0:col1], nan=fill_m)
        block = slice(start - row0, stop - row0)
        line[block], pixel[block] = interpolate_lookup(lookup, row_grid, col_grid, height_m)
    return line, pixel


def lookup_errors(product, orbit, transform, crs, dem_ellipsoid_m, crop, line, pixel):
    """Largest lookup errors (px) against exact geocoding at 400 random pixels of the crop."""
    row0, row1, col0, col1 = crop
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(row0, row1, 400), rng.integers(col0, col1, 400)
    height_mask = np.isfinite(dem_ellipsoid_m[rows, cols])
    rows, cols = rows[height_mask], cols[height_mask]
    lon, lat = lonlat_of_pixels(transform, crs, rows, cols)
    height_m = dem_ellipsoid_m[rows, cols].astype('float64')
    exact_line, exact_pixel = radar_coords(product, orbit, np.asarray(lon), np.asarray(lat), height_m)
    return dict(
        lut_line_err=float(np.abs(exact_line - line[rows - row0, cols - col0]).max()),
        lut_pix_err=float(np.abs(exact_pixel - pixel[rows - row0, cols - col0]).max()),
    )


def interpolate_vectors(lines, pixels, values, line_axis, pixel_axis):
    """Bilinear interpolation of annotation vectors (one vector per line) onto a window."""
    rows = np.stack([np.interp(pixel_axis, px, v) for px, v in zip(pixels, values, strict=True)])
    j = np.clip(np.searchsorted(lines, line_axis) - 1, 0, len(lines) - 2)
    w = np.clip((line_axis - lines[j]) / (lines[j + 1] - lines[j]), 0, 1).astype('float32')[:, None]
    return (rows[j] * (1 - w) + rows[j + 1] * w).astype('float32')


def azimuth_noise_scale(blocks, line_axis, line0, pixel0, shape):
    """Azimuth noise factor of a window; 1 outside the annotated blocks."""
    scale = np.ones(shape, 'float32')
    for block in blocks:
        r0, r1 = max(block['first_line'] - line0, 0), min(block['last_line'] - line0 + 1, shape[0])
        c0, c1 = max(block['first_pixel'] - pixel0, 0), min(block['last_pixel'] - pixel0 + 1, shape[1])
        if r0 < r1 and c0 < c1:
            scale[r0:r1, c0:c1] = np.interp(line_axis[r0:r1], block['line'], block['lut'])[:, None]
    return scale


def calibrate(dn, line0, pixel0, calibration, noise):
    """sigma0 (ESA-EOPG-CSCOP-TN-0002) of a DN window starting at (line0, pixel0); NaN where DN is 0."""
    line_axis = np.arange(line0, line0 + dn.shape[0], dtype='float64')
    pixel_axis = np.arange(pixel0, pixel0 + dn.shape[1], dtype='float64')
    gain = interpolate_vectors(
        calibration['line'].astype(float), calibration['pixel'], calibration['sigma'], line_axis, pixel_axis
    )
    range_noise = noise['range']
    noise_power = interpolate_vectors(
        range_noise['line'].astype(float), range_noise['pixel'], range_noise['lut'], line_axis, pixel_axis
    )
    noise_power *= azimuth_noise_scale(noise['azimuth'], line_axis, line0, pixel0, dn.shape)
    amplitude = dn.astype('float32')
    sigma0 = (amplitude * amplitude - noise_power) / (gain * gain)
    sigma0 = np.maximum(sigma0, 1e-5)  # negative values after noise removal go to a -50 dB floor
    sigma0[dn == 0] = np.nan
    return sigma0


def read_sigma0(item, pol, window):
    """Calibrated sigma0 of one polarisation inside the image window (line0, line1, pixel0, pixel1)."""
    line0, line1, pixel0, pixel1 = window
    with rasterio.Env(GDAL_HTTP_MAX_RETRY=6, GDAL_HTTP_RETRY_DELAY=3), rasterio.open(item.assets[pol].href) as src:
        dn = src.read(1, window=Window(pixel0, line0, pixel1 - pixel0, line1 - line0))
    calibration = parse_calibration(fetch_text(item.assets[f'schema-calibration-{pol}'].href))
    noise = parse_noise(fetch_text(item.assets[f'schema-noise-{pol}'].href))
    return calibrate(dn, line0, pixel0, calibration, noise)


def resample(sigma0, window, line, pixel, scene_mask):
    """Bilinear resampling of a sigma0 window at fractional (line, pixel); NaN outside scene_mask."""
    line0, _, pixel0, _ = window
    out = np.full(line.shape, np.nan, 'float32')
    for start in range(0, line.shape[0], BLOCK_ROWS):
        rows = slice(start, min(line.shape[0], start + BLOCK_ROWS))
        coords = np.stack([line[rows] - line0, pixel[rows] - pixel0])
        out[rows] = ndi.map_coordinates(sigma0, coords, order=1, mode='constant', cval=np.nan)
    out[~scene_mask] = np.nan
    return out


def geocode_item(item, transform, crs, height, width, dem_ellipsoid_m, lookup_step_px=LOOKUP_STEP_PX, log=print):
    """Linear sigma0 per polarisation of a signed sentinel-1-grd item on the event grid, and geocoding diagnostics."""
    product = parse_product(fetch_product_annotation(item, 'vv'))
    orbit = Orbit(product['orbit_time_s'], product['orbit_position_m'], product['orbit_velocity_m_s'])
    diagnostics = dict(orbit_fit_m=orbit.fit_error_m, **tie_point_errors(product, orbit))
    sigma0 = dict.fromkeys(POLS)
    lookup = build_lookup(product, orbit, transform, crs, height, width, dem_ellipsoid_m, lookup_step_px)
    crop = scene_crop(lookup, product)
    if crop is None:
        diagnostics['cover'] = 0.0
        return sigma0, diagnostics
    line, pixel = crop_radar_coords(lookup, dem_ellipsoid_m, crop)
    diagnostics.update(lookup_errors(product, orbit, transform, crs, dem_ellipsoid_m, crop, line, pixel))
    scene_mask = (line > -1) & (line < product['line_count']) & (pixel > -1) & (pixel < product['pixel_count'])
    diagnostics['cover'] = float(scene_mask.sum() / (height * width))
    if not scene_mask.any():
        return sigma0, diagnostics
    window = (
        int(max(0, np.floor(line[scene_mask].min()) - 2)),
        int(min(product['line_count'], np.ceil(line[scene_mask].max()) + 3)),
        int(max(0, np.floor(pixel[scene_mask].min()) - 2)),
        int(min(product['pixel_count'], np.ceil(pixel[scene_mask].max()) + 3)),
    )
    row0, row1, col0, col1 = crop
    for pol in POLS:
        grid = np.full((height, width), np.nan, 'float32')
        grid[row0:row1, col0:col1] = resample(read_sigma0(item, pol, window), window, line, pixel, scene_mask)
        sigma0[pol] = grid
    log(
        f'grd {item.id[:60]} tie points line/pix(srgr)/pix max {diagnostics["gcp_line_err"]:.3f}/'
        f'{diagnostics["gcp_pix_err"]:.3f}/{diagnostics["gcp_pix_err_hybrid"]:.3f} px, lookup '
        f'{diagnostics["lut_line_err"]:.3f}/{diagnostics["lut_pix_err"]:.3f} px, orbit fit '
        f'{diagnostics["orbit_fit_m"]:.3f} m, cover {diagnostics["cover"]:.2f}'
    )
    return sigma0, diagnostics
