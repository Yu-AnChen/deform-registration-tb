import palom
import skimage.transform
import numpy as np
import itk


def get_default_crc_params(
    grid_size: float = 80.0,
    sample_region_size: float = 300.0,
    sample_number_of_pixels: int = 4_000,
    number_of_iterations: int = 1_000,
):
    parameter_object = itk.ParameterObject.New()
    # deformation
    p = parameter_object.GetDefaultParameterMap("bspline")

    del p["FinalGridSpacingInPhysicalUnits"]

    p["ASGDParameterEstimationMethod"] = ["DisplacementDistribution"]
    p["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    p["HowToCombineTransforms"] = ["Compose"]
    p["Interpolator"] = ["BSplineInterpolator"]
    p["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    p["Transform"] = ["RecursiveBSplineTransform"]

    # metrics: higher weight on the bending energy panelty to reduce distortion
    p["Metric"] = ["AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"]
    p["Metric0Weight"] = ["1.0"]
    p["Metric1Weight"] = ["100.0"]

    # these should be pixel-size & image size related
    p["NumberOfResolutions"] = ["4"]
    p["GridSpacingSchedule"] = [f"{2**i}" for i in range(0, 4)[::-1]]
    p["FinalGridSpacingInVoxels"] = [f"{grid_size}", f"{grid_size}"]
    p["NumberOfSamplesForExactGradient"] = [f"{10_000}"]
    p["NumberOfSpatialSamples"] = [f"{sample_number_of_pixels}"]
    # p["NumberOfSpatialSamples"] = [f"{5000 // 2**i}" for i in range(4)[::-1]]
    p["UseRandomSampleRegion"] = ["true"]
    p["SampleRegionSize"] = [f"{sample_region_size}"]
    p["NumberOfHistogramBins"] = ["32"]

    # number if iterations in gradient descent
    p["MaximumNumberOfIterations"] = [f"{number_of_iterations}"]

    # must set to write result image, could be a bug?!
    p["WriteResultImage"] = ["true"]
    p["ResultImageFormat"] = ["tif"]

    return p


def _run_one_setting(ref, moving, setting):
    if setting is None:
        setting = {}
    elastix_parameter = itk.ParameterObject.New()
    elastix_parameter.AddParameterMap(
        elastix_parameter.GetDefaultParameterMap("rigid", 4)
    )
    elastix_parameter.AddParameterMap(get_default_crc_params(**setting))
    warpped_moving, transform_parameter = itk.elastix_registration_method(
        ref,
        moving,
        parameter_object=elastix_parameter,
        log_to_console=False,
    )
    return warpped_moving, transform_parameter, elastix_parameter


r1 = palom.reader.OmePyramidReader(
    r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_005-ori.ome.tif"
)
r2 = palom.reader.OmePyramidReader(
    r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_006-ori.ome.tif"
)

aligner = palom.align.Aligner(
    ref_img=r1.pyramid[0][0],
    moving_img=r2.pyramid[0][0],
    # do feature detection and matching at resolution of ~40 µm/pixel
    ref_thumbnail=palom.img_util.cv2_downscale_local_mean(r1.pyramid[2][0], 2),
    moving_thumbnail=palom.img_util.cv2_downscale_local_mean(r2.pyramid[2][0], 2),
    ref_thumbnail_down_factor=2 ** (3 - 0),
    moving_thumbnail_down_factor=2 ** (3 - 0),
)
aligner.coarse_register_affine(
    n_keypoints=10_000,
    test_flip=False,
    test_intensity_invert=False,
    auto_mask=True,
)

ref = aligner.ref_thumbnail
moving_ori = aligner.moving_thumbnail
moving = skimage.transform.warp(
    aligner.moving_thumbnail,
    np.linalg.inv(aligner.coarse_affine_matrix),
    output_shape=ref.shape,
    preserve_range=True,
)
moving = np.round(moving).astype(ref.dtype)

wmoving, tform, _ = _run_one_setting(
    ref, moving, dict(sample_region_size=100, grid_size=40)
)


def to_deformation_field(elastix_parameter):
    shape = elastix_parameter.GetParameterMap(0).get("Size")[::-1]
    shape = np.array(shape, dtype="int")
    return itk.transformix_deformation_field(
        itk.GetImageFromArray(np.zeros(shape, dtype="uint8")), elastix_parameter
    )


dform = np.moveaxis(to_deformation_field(tform), 2, 0)


# ---------------------------------------------------------------------------- #
#                   affine + deform apply to moving thumbnail                  #
# ---------------------------------------------------------------------------- #
mx = aligner.coarse_affine_matrix[:2, :2]
ddx, ddy = (np.linalg.inv(mx) @ dform.reshape(2, -1)).reshape(dform.shape)

shape = np.add(ref.shape, 100)

tmgrid = skimage.transform.warp(
    np.dstack(np.mgrid[: shape[0], : shape[1]].astype("float32")),
    np.linalg.inv(aligner.coarse_affine_matrix),
    preserve_range=True,
    output_shape=shape,
)
tmy, tmx = np.moveaxis(tmgrid, 2, 0)

import napari

v = napari.Viewer()

for ii in [ref, moving_ori, moving, wmoving]:
    v.add_image(ii, colormap="cividis")

v.add_image(
    skimage.transform.warp(
        moving_ori,
        np.array(
            [
                tmy + np.pad(ddy, [(0, 100), (0, 100)]),
                tmx + np.pad(ddx, [(0, 100), (0, 100)]),
            ]
        ),
        output_shape=shape,
    )
)
# ---------------------------------------------------------------------------- #
#                     affine + deform apply to moving image                    #
# ---------------------------------------------------------------------------- #
mx = aligner.coarse_affine_matrix[:2, :2]
downscale = aligner.ref_thumbnail_down_factor
mx_d = skimage.transform.AffineTransform(scale=downscale).params

ddx, ddy = (
    # FIXME confirm whether it's generalizable!
    (np.linalg.inv(mx) @ dform.reshape(2, -1)).T @ mx_d[:2, :2]
).T.reshape(dform.shape)

shape = aligner.ref_img.shape

tmgrid = skimage.transform.warp(
    np.dstack(np.mgrid[: shape[0], : shape[1]].astype("float32")),
    np.linalg.inv(aligner.affine_matrix),
    preserve_range=True,
    output_shape=shape,
)

ddy_ddx = skimage.transform.warp(
    np.dstack([ddy, ddx]),
    np.linalg.inv(mx_d),
    preserve_range=True,
    output_shape=shape,
)

dtmgrid = tmgrid + ddy_ddx


import napari

v = napari.Viewer()

for ii in [ref, moving_ori, moving, wmoving]:
    v.add_image(ii, colormap="cividis", translate=(0.5, 0.5))

v.add_image(aligner.ref_img, scale=(1 / 8,) * 2, translate=(0.5 * 1 / 8,) * 2)

v.add_image(
    skimage.transform.warp(
        np.asarray(aligner.moving_img),
        np.moveaxis(dtmgrid, 2, 0),
        preserve_range=True,
        output_shape=shape,
    ),
    scale=(1 / 8,) * 2,
    translate=(0.5 * 1 / 8,) * 2,
)

# ---------------------------------------------------------------------------- #
#                           affine + deform + padding                          #
# ---------------------------------------------------------------------------- #
padding_xy_ul = (50, 70)
out_shape = np.add(aligner.ref_img.shape, 500)

Affine = skimage.transform.AffineTransform
tform_ref = (
    Affine(scale=1 / aligner.ref_thumbnail_down_factor)
    + Affine(translation=padding_xy_ul)
    + Affine(scale=aligner.ref_thumbnail_down_factor)
)
tform_moving = (
    Affine(scale=1 / aligner.moving_thumbnail_down_factor)
    + Affine(matrix=aligner.coarse_affine_matrix)
    + Affine(translation=padding_xy_ul)
    + Affine(scale=aligner.ref_thumbnail_down_factor)
)

mx = aligner.coarse_affine_matrix[:2, :2]
mx_d = (
    Affine(translation=padding_xy_ul) + Affine(scale=aligner.ref_thumbnail_down_factor)
).params

ddx, ddy = (
    # FIXME confirm whether it's generalizable!
    (np.linalg.inv(mx) @ dform.reshape(2, -1)).T @ mx_d[:2, :2]
).T.reshape(dform.shape)

shape = out_shape

tmgrid = skimage.transform.warp(
    np.dstack(np.mgrid[: shape[0], : shape[1]].astype("float32")),
    np.linalg.inv(tform_moving.params),
    preserve_range=True,
    output_shape=shape,
)

ddy_ddx = skimage.transform.warp(
    np.dstack([ddy, ddx]),
    np.linalg.inv(mx_d),
    preserve_range=True,
    output_shape=shape,
)

dtmgrid = tmgrid + ddy_ddx


v.add_image(
    skimage.transform.warp(
        np.asarray(aligner.ref_img),
        tform_ref.inverse,
        preserve_range=True,
        output_shape=shape,
    ),
    scale=(1 / 8,) * 2,
    translate=(0.5 * 1 / 8,) * 2,
)

v.add_image(
    skimage.transform.warp(
        np.asarray(aligner.moving_img),
        np.moveaxis(dtmgrid, 2, 0),
        preserve_range=True,
        output_shape=shape,
    ),
    scale=(1 / 8,) * 2,
    translate=(0.5 * 1 / 8,) * 2,
)


# -------- affine transform full size grid + rescale deformation field ------- #
shape = aligner.ref_img.shape
# scikit-image use "float64" internally
my, mx = np.mgrid[: shape[0], : shape[1]].astype("float32")

tmy = skimage.transform.warp(
    my,
    skimage.transform.AffineTransform(matrix=aligner.affine_matrix).inverse,
    preserve_range=True,
    output_shape=shape,
)

tmx = skimage.transform.warp(
    mx,
    skimage.transform.AffineTransform(matrix=aligner.affine_matrix).inverse,
    preserve_range=True,
    output_shape=shape,
)


dx, dy = dform

ddx = skimage.transform.warp(
    dx * aligner.ref_thumbnail_down_factor,
    skimage.transform.AffineTransform(
        scale=(1 / aligner.ref_thumbnail_down_factor,) * 2
    ),
    preserve_range=True,
    output_shape=shape,
)
ddy = skimage.transform.warp(
    dy * aligner.ref_thumbnail_down_factor,
    skimage.transform.AffineTransform(
        scale=(1 / aligner.ref_thumbnail_down_factor,) * 2
    ),
    preserve_range=True,
    output_shape=shape,
)

warpped_moving_image = skimage.transform.warp(
    np.asarray(aligner.moving_img),
    np.array([tmy + ddy, tmx + ddx]),
    preserve_range=True,
    cval=234,
)

# what if affine transform deformation field?
# this seems wrong....
import napari

v = napari.Viewer()
v.add_image(-ref, name="ref")
v.add_image(-aligner.moving_thumbnail, name="moving")
v.add_image(-moving, name="moving-affine")
v.add_image(-wmoving, name="moving-elastix")

dx, dy = dform
my, mx = np.mgrid[: ref.shape[0], : ref.shape[1]]
wmoving_sk = skimage.transform.warp(
    moving,
    np.array([my + dy, mx + dx]),
    preserve_range=True,
    output_shape=ref.shape,
)

v.add_image(-np.round(wmoving_sk).astype("uint8"), name="moving-elastix-sk")


wdx = skimage.transform.warp(
    dx,
    aligner.coarse_affine_matrix,
    preserve_range=True,
)

wdy = skimage.transform.warp(
    dy,
    aligner.coarse_affine_matrix,
    preserve_range=True,
)

v.add_image(
    skimage.transform.warp(
        -aligner.moving_thumbnail,
        np.array([my + wdy, mx + wdx]),
        preserve_range=True,
        output_shape=ref.shape,
    )
)


# ---------------------------------------------------------------------------- #
#                                    asfsda                                    #
# ---------------------------------------------------------------------------- #


shape = ref.shape
my, mx = np.mgrid[
    : aligner.moving_thumbnail.shape[0], : aligner.moving_thumbnail.shape[1]
].astype("float32")

tmy = skimage.transform.warp(
    my,
    skimage.transform.AffineTransform(matrix=aligner.coarse_affine_matrix).inverse,
    preserve_range=True,
    output_shape=shape,
)

tmx = skimage.transform.warp(
    mx,
    skimage.transform.AffineTransform(matrix=aligner.coarse_affine_matrix).inverse,
    preserve_range=True,
    output_shape=shape,
)

dx, dy = dform

wmi = skimage.transform.warp(
    aligner.moving_thumbnail.astype("float32"),
    np.array([tmy + dy, tmx + dx]),
    preserve_range=True,
    cval=0,
)

# try again
skimage.transform = skimage.transform

shape = ref.shape
my, mx = np.mgrid[: shape[0], : shape[1]].astype("float32")
dx, dy = dform

tty = skimage.transform.warp(
    my + dy,
    skimage.transform.AffineTransform(matrix=aligner.coarse_affine_matrix).inverse,
    preserve_range=True,
    output_shape=shape,
)

ttx = skimage.transform.warp(
    mx + dx,
    skimage.transform.AffineTransform(matrix=aligner.coarse_affine_matrix).inverse,
    preserve_range=True,
    output_shape=shape,
)

wmi = skimage.transform.warp(
    aligner.moving_thumbnail,
    np.array([tty, ttx]),
    preserve_range=True,
    cval=234,
)


# ---------------------------------------------------------------------------- #
#                                 demo and dev                                 #
# ---------------------------------------------------------------------------- #
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import skimage.transform  # noqa: E402

img = np.eye(3) * 12
dform = np.full((2, 3, 3), -1, dtype="float32")
mapping = dform + np.mgrid[:3, :3].astype("float32")[::-1]

wimg = cv2.remap(img, mapping[0], mapping[1], cv2.INTER_LINEAR)

limg = skimage.transform.rescale(img, (3, 3), preserve_range=True, order=0)
scale = 3
sdform = np.array(
    [
        skimage.transform.warp(
            dff * scale,
            skimage.transform.AffineTransform(scale=(1 / scale, 1 / scale)),
            output_shape=limg.shape,
            preserve_range=True,
        )
        for dff in dform
    ]
)

smapping = sdform + np.mgrid[: limg.shape[0], : limg.shape[1]].astype("float32")[::-1]
wlimg = cv2.remap(limg, smapping[0], smapping[1], cv2.INTER_LINEAR)
