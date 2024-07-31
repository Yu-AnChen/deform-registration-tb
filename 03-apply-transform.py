import pathlib

import cv2
import itk
import matplotlib.pyplot as plt
import numpy as np
import tifffile


# comparison of cross correlation and normalized dot product
# https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html
def _ncc(img1, img2):
    """
    Normalized Inner Product (NIP)
    """
    return np.sum(img1.astype("float") * img2.astype("float")) / (
        np.linalg.norm(img1) * np.linalg.norm(img2)
    )


def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1P
    """
    mean_data = np.mean(data)
    std_data = np.std(data, ddof=1)
    # return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data - mean_data) / (std_data)


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0 / (data0.size - 1)) * np.sum(norm_data(data0) * norm_data(data1))


def warp_moving(moving, elastix_tform_paths):
    elastix_parameter = itk.ParameterObject.New()
    for ff in elastix_tform_paths:
        elastix_parameter.AddParameterFile(str(ff))
    warpped_moving = itk.transformix_filter(
        moving, transform_parameter_object=elastix_parameter
    )
    return warpped_moving


def to_deformation_field(elastix_tform_paths):
    elastix_parameter = itk.ParameterObject.New()
    for ff in elastix_tform_paths:
        elastix_parameter.AddParameterFile(str(ff))
    shape = elastix_parameter.GetParameterMap(0).get("Size")[::-1]
    shape = np.array(shape, dtype="int")
    return itk.transformix_deformation_field(
        itk.GetImageFromArray(np.zeros(shape, dtype="uint8")), elastix_parameter
    )


def compare_deformed(
    ref_path,
    moving_path,
    elastix_tform_paths,
):
    ref = tifffile.imread(ref_path)
    moving = tifffile.imread(moving_path)

    warpped_moving = warp_moving(moving, elastix_tform_paths)

    print(f"{ncc(ref, moving):.4f} --> {ncc(ref, warpped_moving):.4f}")
    return (ncc(ref, moving), ncc(ref, warpped_moving))


elastix_dir = pathlib.Path(
    r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/reg-param/tform"
)

section_number = list(range(1, 40))
ref_number = section_number[:-1]
moving_number = section_number[1:]

_deform_fields = []
for nnref, nnmoving in zip(ref_number, moving_number):
    print(f"B5_3DHE_{nnmoving:03}")

    elastix_tform_paths = [
        elastix_dir
        / f"B5_3DHE_{nnmoving:03}-to-B5_3DHE_{nnref:03}-tform-elastix-param-0.txt",
        elastix_dir
        / f"B5_3DHE_{nnmoving:03}-to-B5_3DHE_{nnref:03}-tform-elastix-param-1.txt",
    ]

    deformation_field = np.moveaxis(to_deformation_field(elastix_tform_paths), 2, 0)
    _deform_fields.append(deformation_field)


H, W = _deform_fields[0].shape[1:]
_deform_fields.insert(0, np.zeros_like(_deform_fields[0]))

deform_fields = (
    np.add.accumulate(_deform_fields) + np.mgrid[:H, :W].astype("float32")[::-1]
)

imgs = []
warpped_imgs = []
for nn, dd in zip(section_number, deform_fields):
    ref_path = rf"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_{nn:03}-affine.ome.tif"
    img0 = tifffile.imread(ref_path)
    imgs.append(img0)
    warpped_imgs.append(
        cv2.remap(img0.astype("float32"), dd[0], dd[1], cv2.INTER_LINEAR)
    )


import napari  # noqa: E402

v = napari.Viewer()
v.add_image(-1 * np.asarray(warpped_imgs))
# ---------------------------------------------------------------------------- #
#                               for presentation                               #
# ---------------------------------------------------------------------------- #


# -------------- calculate masked ncc and plot its distribution -------------- #
def compute_mask(img, out_shape=None):
    import palom
    import skimage.transform

    mask = palom.img_util.entropy_mask(palom.img_util.cv2_downscale_local_mean(img, 8))
    mask = palom.img_util.repeat_2d(mask, (8, 8))
    if out_shape is not None:
        mask = skimage.transform.warp(
            mask, np.eye(3), output_shape=out_shape, preserve_range=True
        )
        mask = np.round(mask).astype("bool")
    return mask


results = []
for nnref, nnmoving in zip(ref_number, moving_number):
    print(f"B5_3DHE_{nnmoving:03}")

    ref_path = rf"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_{nnref:03}-affine.ome.tif"
    img0 = tifffile.imread(ref_path)
    # without masking the result will be skewed by the background (no tissue)
    # regions
    mask = compute_mask(img0, out_shape=img0.shape)

    moving_path = rf"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_{nnmoving:03}-affine.ome.tif"
    elastix_tform_paths = [
        elastix_dir
        / f"B5_3DHE_{nnmoving:03}-to-B5_3DHE_{nnref:03}-tform-elastix-param-0.txt",
        elastix_dir
        / f"B5_3DHE_{nnmoving:03}-to-B5_3DHE_{nnref:03}-tform-elastix-param-1.txt",
    ]
    img3 = tifffile.imread(moving_path)
    img4 = warp_moving(img3, elastix_tform_paths)

    results.append([ncc(img0[mask], ii[mask]) for ii in [img3, img4]])

results = np.array(results)

tform_methods = ["Affine", "Non-linear"]
plt.figure()
plt.plot(results.T[:, :-2], c="#555555", alpha=0.1)
plt.plot(results.T[:, -2:-1], c="#555555", alpha=0.1, label="002 - 038")
plt.plot(results.T[:, -1:], c="#555555", alpha=0.8, label="039")
_ = plt.boxplot(
    results,
    positions=range(len(tform_methods)),
    widths=0.2,
    tick_labels=tform_methods,
)
plt.ylabel("Normalized cross correlation coefficient")
plt.legend()

import pandas as pd

df = pd.DataFrame(
    results,
    index=[f"B5_3DHE_{nnmoving:03}" for nnmoving in moving_number],
    columns=tform_methods,
)

# # ---------------------------- make example image ---------------------------- #
# imgs = []
# for nn in [40]:
#     print(f"C{nn:02}")
#     ref_path = rf"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240714-deform-registration-crc/img-data/C{nn:02}-ref.tif"
#     moving_ori_path = rf"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240714-deform-registration-crc/img-data/C{nn:02}-moving-ori.tif"
#     img0 = tifffile.imread(ref_path)
#     img1 = tifffile.imread(moving_ori_path)
#     shift = 4 * np.array(translate_moving(img0, img1, 4))
#     img2 = skimage.transform.warp(
#         np.flipud(img1),
#         skimage.transform.AffineTransform(translation=shift[::-1]).inverse,
#         output_shape=img0.shape,
#         preserve_range=True,
#     ).astype("uint16")
#     img1 = skimage.transform.warp(
#         np.flipud(img1),
#         np.eye(3),
#         output_shape=img0.shape,
#         preserve_range=True,
#     ).astype("uint16")

#     moving_path = rf"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240714-deform-registration-crc/img-data/C{nn:02}-moving.tif"
#     elastix_tform_paths = [
#         elastix_dir / f"C{nn:02}-tform-elastix-param-0.txt",
#         elastix_dir / f"C{nn:02}-tform-elastix-param-1.txt",
#     ]
#     img3 = tifffile.imread(moving_path)
#     img4 = warp_moving(img3, elastix_tform_paths)
#     imgs.append([img0, img1, img2, img3, img4])

# # --------------------------- feature matching demo -------------------------- #
# import palom  # noqa: E402

# palom.register.feature_based_registration(
#     palom.img_util.cv2_downscale_local_mean(img0, 4),
#     palom.img_util.cv2_downscale_local_mean(img1, 4),
#     plot_match_result=True,
#     auto_mask=True,
#     n_keypoints=1500,
# )

# # -------------------------- plot deformation field -------------------------- #
# elastix_parameter = itk.ParameterObject.New()
# for ff in elastix_tform_paths:
#     elastix_parameter.AddParameterFile(str(ff))
# shape = elastix_parameter.GetParameterMap(0).get("Size")[::-1]
# shape = np.array(shape, dtype="int")
# deformation_field = itk.transformix_deformation_field(
#     itk.GetImageFromArray(np.zeros(shape, dtype="uint8")), elastix_parameter
# )

# inverted_fixed_point = itk.FixedPointInverseDisplacementFieldImageFilter(
#     deformation_field,
#     NumberOfIterations=10,
#     Size=deformation_field.shape[:2][::-1],
# )
# dx, dy = np.moveaxis(inverted_fixed_point, 2, 0)

# _, ax = plt.subplots()
# ax.imshow(np.log1p(img3), cmap="gray", vmin=3, vmax=15)
# nvec = 40  # Number of vectors to be displayed along each image dimension
# nl, nc = shape
# step = max(nl // nvec, nc // nvec)

# y, x = np.mgrid[:nl:step, :nc:step]
# dx_ = dx[::step, ::step]
# dy_ = dy[::step, ::step]

# ax.quiver(
#     x,
#     y,
#     dx_,
#     dy_,
#     color="deepskyblue",
#     units="xy",
#     angles="xy",
#     scale_units="xy",
#     scale=1 / 7,
# )
# ax.set_axis_off()
