"""Relief and slope input channels of the flood segmentation model, computed alike in training and inference."""

import numpy as np
import torch
from scipy import ndimage as ndi
from torch.nn import functional

PX = 10.0  # inference grid pixel size (m)
RELIEF_WINDOW_PX = 101  # about 1 km
DEM_NODATA = -32768


def fill_nodata(dem: np.ndarray) -> np.ndarray:
    """DEM as float32 with nodata cells set to the median of the valid cells (all zero when none is valid)."""
    filled: np.ndarray = dem.astype('float32')
    invalid = ~np.isfinite(filled) | (filled <= DEM_NODATA + 1)
    if invalid.all():
        return np.zeros_like(filled)
    if invalid.any():
        filled[invalid] = np.median(filled[~invalid])
    return filled


def channels_np(dem: np.ndarray, smooth: bool) -> np.ndarray:
    """[relief / 50 m, slope / 30 deg] as float32 [2, H, W] from a gap-free DEM on the inference grid."""
    dem = dem.astype('float32')
    relief = dem - ndi.minimum_filter(dem, size=RELIEF_WINDOW_PX, mode='nearest')
    surface = ndi.uniform_filter(dem, size=3, mode='nearest') if smooth else dem
    padded = np.pad(surface, 1, mode='edge')  # replicate padding, as in channels_torch
    dz_dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * PX)
    dz_dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * PX)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    return np.stack([np.clip(relief / 50.0, 0, 6), np.clip(slope_deg / 30.0, 0, 3)]).astype('float32')


def channels_torch(dem: torch.Tensor, px: float | torch.Tensor, smooth: bool) -> torch.Tensor:
    """Batched channels_np: dem [B, H, W] -> [B, 2, H, W]; `px` is a scalar or a [B] tensor of pixel sizes (m)."""
    dem = dem[:, None]
    local_min = -functional.max_pool2d(-dem, RELIEF_WINDOW_PX, stride=1, padding=RELIEF_WINDOW_PX // 2)
    relief = dem - local_min
    surface = dem
    if smooth:
        surface = functional.avg_pool2d(functional.pad(dem, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
    padded = functional.pad(surface, (1, 1, 1, 1), mode='replicate')
    if isinstance(px, torch.Tensor):
        px = px.view(-1, 1, 1, 1)
    dz_dx = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) / (2 * px)
    dz_dy = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) / (2 * px)
    slope_deg = torch.rad2deg(torch.atan(torch.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)))
    return torch.cat([torch.clamp(relief / 50.0, 0, 6), torch.clamp(slope_deg / 30.0, 0, 3)], 1)
