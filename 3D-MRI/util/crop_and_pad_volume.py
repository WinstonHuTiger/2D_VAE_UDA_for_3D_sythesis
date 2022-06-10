import numpy as np
import torch
from typing import Union
def crop_or_pad_volume_to_size_along_x(vol, nx):
    x = vol.shape[0]
    x_s = (x - nx) // 2
    x_c = (nx - x) // 2

    if x > nx:  # original volume has more slices that the required number of slices
        vol_cropped = vol[x_s:x_s + nx, :, :]
    else:  # original volume has equal of fewer slices that the required number of slices
        vol_cropped = torch.zeros((nx, vol.shape[1], vol.shape[2])).to(vol.device)
        vol_cropped[x_c:x_c + x, :, :] = vol

    return vol_cropped


# ===============================================================
# ===============================================================
def crop_or_pad_volume_to_size_along_y(vol, ny):
    y = vol.shape[1]
    y_s = (y - ny) // 2
    y_c = (ny - y) // 2

    if y > ny:  # original volume has more slices that the required number of slices
        vol_cropped = vol[:, y_s:y_s + ny, :]
    else:  # original volume has equal of fewer slices that the required number of slices
        vol_cropped = torch.zeros((vol.shape[0], ny, vol.shape[2])).to(vol.device)
        vol_cropped[:, y_c:y_c + y, :] = vol

    return vol_cropped


# ===============================================================
# ===============================================================
def crop_or_pad_volume_to_size_along_z(vol, nz):
    z = vol.shape[2]
    z_s = (z - nz) // 2
    z_c = (nz - z) // 2

    if z > nz:  # original volume has more slices that the required number of slices
        vol_cropped = vol[:, :, z_s:z_s + nz]
    else:  # original volume has equal of fewer slices that the required number of slices
        vol_cropped = torch.zeros((vol.shape[0], vol.shape[1], nz)).to(vol.device)
        vol_cropped[:, :, z_c:z_c + z] = vol

    return vol_cropped
def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result
# ===================================================
def normalise_image(image, norm_type='div_by_max'):
    if norm_type == 'zero_mean':
        img_o = image.copy()
        m = torch.mean(img_o)
        s = torch.std(img_o)
        normalized_img = torch.divide((img_o - m), s)

    elif norm_type == 'div_by_max':
        perc1 = percentile(image, 1)
        perc99 = percentile(image, 99)
        normalized_img = torch.divide((image - perc1), (perc99 - perc1))
        normalized_img[normalized_img < 0] = 0.0
        normalized_img[normalized_img > 1] = 1.0

    return normalized_img
