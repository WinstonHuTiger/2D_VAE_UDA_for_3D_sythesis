import nibabel as nib
import numpy as np
import os


# ===================================================
# ===================================================
def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


# ===================================================
# ===================================================
def load_nii(img_path):
    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


# ===================================================
# ===================================================
def save_nii(img_path, data, affine, header=None):
    '''
    Shortcut to save a nifty file
    '''
    if header == None:
        nimg = nib.Nifti1Image(data, affine=affine)
    else:
        nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


# ===================================================
# ===================================================
def normalise_image(image, norm_type='div_by_max'):
    if norm_type == 'zero_mean':
        img_o = np.float32(image.copy())
        m = np.mean(img_o)
        s = np.std(img_o)
        normalized_img = np.divide((img_o - m), s)

    elif norm_type == 'div_by_max':
        perc1 = np.percentile(image, 1)
        perc99 = np.percentile(image, 99)
        normalized_img = np.divide((image - perc1), (perc99 - perc1))
        normalized_img[normalized_img < 0] = 0.0
        normalized_img[normalized_img > 1] = 1.0

    return normalized_img


# ===============================================================
# ===============================================================
def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


# ===============================================================
# ===============================================================
def crop_or_pad_volume_to_size_along_x(vol, nx):
    x = vol.shape[0]
    x_s = (x - nx) // 2
    x_c = (nx - x) // 2

    if x > nx:  # original volume has more slices that the required number of slices
        vol_cropped = vol[x_s:x_s + nx, :, :]
    else:  # original volume has equal of fewer slices that the required number of slices
        vol_cropped = np.zeros((nx, vol.shape[1], vol.shape[2]))
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
        vol_cropped = np.zeros((vol.shape[0], ny, vol.shape[2]))
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
        vol_cropped = np.zeros((vol.shape[0], vol.shape[1], nz))
        vol_cropped[:, :, z_c:z_c + z] = vol

    return vol_cropped


# ===============================================================
# Group the segmentation classes into the required categories
# ===============================================================
def group_segmentation_classes(seg_mask):
    seg_mask_modified = group_segmentation_classes_15(seg_mask)
    return seg_mask_modified


# ===============================================================
# Group the segmentation classes into the required categories
# ===============================================================
def group_segmentation_classes_15(a):
    """
    Args:
    label_data : Freesurfer generated Labels Data of a 3D MRI scan.
    Returns:
    relabelled_data
    """

    background_ids = [0]  # [background]
    csf_ids = [24]  # [csf]
    brainstem_ids = [16]  # [brain stem]
    cerebellum_wm_ids = [7, 46]
    cerebellum_gm_ids = [8, 47]
    cerebral_wm_ids = [2, 41, 251, 252, 253, 254, 255]
    cerebral_gm_ids = np.arange(1000, 3000)
    cerebral_cortex_ids = [3, 42]
    thalamus_ids = [10, 49]
    hippocampus_ids = [17, 53]
    amygdala_ids = [18, 54]
    ventricle_ids = [4, 43, 14, 15, 72]  # lat, 3rd, 4th, 5th
    choroid_plexus_ids = [31, 63]
    caudate_ids = [11, 50]
    putamen_ids = [12, 51]
    pallidum_ids = [13, 52]
    accumbens_ids = [26, 58]
    ventral_DC_ids = [28, 60]
    misc_ids = [5, 44, 30, 62, 77, 80, 85]  # inf lat ventricle, right, left vessel, hypointensities, optic-chiasm

    a = np.array(a, dtype='uint16')
    b = np.zeros((a.shape[0], a.shape[1], a.shape[2]), dtype='uint16')

    unique_ids = np.unique(a)
    # print("Unique labels in the original segmentation mask:", unique_ids)

    for i in unique_ids:
        if (i in cerebral_gm_ids):
            b[a == i] = 3
        elif (i in cerebral_cortex_ids):
            b[a == i] = 3
        elif (i in accumbens_ids):
            b[a == i] = 3
        elif (i in background_ids):
            b[a == i] = 0
        elif (i in cerebellum_gm_ids):
            b[a == i] = 1
        elif (i in cerebellum_wm_ids):
            b[a == i] = 2
        elif (i in cerebral_wm_ids):
            b[a == i] = 4
        elif (i in misc_ids):
            b[a == i] = 4
        elif (i in thalamus_ids):
            b[a == i] = 5
        elif (i in hippocampus_ids):
            b[a == i] = 6
        elif (i in amygdala_ids):
            b[a == i] = 7
        elif (i in ventricle_ids):
            b[a == i] = 8
        elif (i in choroid_plexus_ids):
            b[a == i] = 8
        elif (i in caudate_ids):
            b[a == i] = 9
        elif (i in putamen_ids):
            b[a == i] = 10
        elif (i in pallidum_ids):
            b[a == i] = 11
        elif (i in ventral_DC_ids):
            b[a == i] = 12
        elif (i in csf_ids):
            b[a == i] = 13
        elif (i in brainstem_ids):
            b[a == i] = 14
        else:
            print('unknown id:', i)
            print('num_voxels:', np.shape(np.where(a == i))[1])

    print("Unique labels in the modified segmentation mask: ", np.unique(b))

    return b


# ===============================================================
# ===============================================================
def make_onehot(arr, nlabels):
    # taken from https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy/36960495
    ncols = nlabels
    out = np.zeros((arr.size, ncols), dtype=np.uint8)
    out[np.arange(arr.size), arr.ravel()] = 1
    out.shape = arr.shape + (ncols,)
    return out
