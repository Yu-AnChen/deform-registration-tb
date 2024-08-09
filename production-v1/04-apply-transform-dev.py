import pathlib

import numpy as np
import palom
import skimage.transform
import tifffile

file_paths = """
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_001-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_002-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_003-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_004-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_005-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_006-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_007-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_008-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_009-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_010-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_011-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_012-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_013-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_014-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_015-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_016-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_017-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_018-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_019-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_020-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_021-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_022-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_023-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_024-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_025-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_026-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_027-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_028-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_029-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_030-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_031-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_032-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_033-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_034-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_035-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_036-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_037-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_038-ori.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-ori/B5_3DHE_039-ori.ome.tif
""".strip().split("\n")[:2]

mx_dir = pathlib.Path(
    r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/production-v1/img-data/affine"
)
df_dir = pathlib.Path(
    r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/production-v1/img-data/elastix/registered-thumbnail"
)


dfs_to_first = []
for ff in file_paths:
    ff = pathlib.Path(ff)
    moving_name = ff.name.replace("-ori.ome.tif", "").replace("-ori", "")
    df_filename = df_dir / f"{moving_name}-elastix-deformation-field-xy.ome.tif"
    if not df_filename.exists():
        print(df_filename, "does not exist")
        dfs_to_first.append(None)
        continue
    dfs_to_first.append(
        tifffile.imread(df_dir / f"{moving_name}-elastix-deformation-field-xy.ome.tif")
    )

mxs_to_first = []
for ff in file_paths:
    ff = pathlib.Path(ff)
    mxs_to_first.append(
        np.loadtxt(
            mx_dir / ff.name.replace("-ori.ome.tif", "-affine-matrix.csv"),
            delimiter=",",
        )
    )


# add upper-left padding
padding = 0.1
ref_reader = palom.reader.OmePyramidReader(file_paths[0])
ref_shape = ref_reader.pyramid[0].shape[1:]

padded_shape = np.ceil(np.multiply(ref_shape, 2 * padding + 1)).astype("int")
# translation will be applied post alignment to the first at the lower resolution
offset = padding * np.divide(ref_shape, 2 ** (3 - 0))[::-1]


# hairy settings for scaling...
d_pyramid = 2
moving_pyramid_level = 0
d_moving = d_pyramid**3  # same as aligner.moving_thumbnail_down_factor
d_ref = d_pyramid**3  # same as aligner.ref_thumbnail_down_factor
d_output = d_pyramid**moving_pyramid_level * d_moving / d_ref


out_dir = pathlib.Path(
    r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/production-v1/img-data/elastix/registered"
)

Affine = skimage.transform.AffineTransform
for mx, dform, ff in zip(mxs_to_first, dfs_to_first, file_paths):
    print(pathlib.Path(ff).name)
    mx_d = Affine(scale=4).params

    tform = (
        Affine(scale=1 / d_moving)
        + Affine(matrix=mx)
        + Affine(translation=offset)
        + Affine(scale=d_ref)
    )

    ddx, ddy = (
        # FIXME confirm whether it's generalizable!
        (np.linalg.inv(mx[:2, :2]) @ dform.reshape(2, -1)).T @ mx_d[:2, :2]
    ).T.reshape(dform.shape)

    tmgrid = skimage.transform.warp(
        np.dstack(np.mgrid[: padded_shape[0], : padded_shape[1]].astype("float32")),
        np.linalg.inv(tform.params),
        preserve_range=True,
        output_shape=padded_shape,
        cval=np.nan,
    )

    ddy_ddx = skimage.transform.warp(
        np.dstack([ddy, ddx]),
        np.linalg.inv(mx_d),
        preserve_range=True,
        output_shape=padded_shape,
        cval=0,
    )

    tmgrid += ddy_ddx
    np.nan_to_num(tmgrid, copy=False, nan=-1)

    moving = tifffile.imread(ff)
    warped_moving = skimage.transform.warp(
        moving,
        np.moveaxis(tmgrid, 2, 0),
        preserve_range=True,
        output_shape=padded_shape,
        cval=np.percentile(moving, 99.5),
    )

    out_filename = pathlib.Path(ff).name.replace("-ori.ome.tif", "-elastix.ome.tif")
    tifffile.imwrite(
        out_dir / out_filename,
        np.floor(warped_moving).astype("uint8"),
        compression="zlib",
    )




# ---------------------------------------------------------------------------- #
#                                   asdfasdf                                   #
# ---------------------------------------------------------------------------- #



ttt = (
    Affine(scale=1/2)
    + Affine(translation=-offset)
    + Affine(matrix=np.linalg.inv(mx))
    + Affine(scale=8)
)

ddx, ddy = dform

ddy_ddx = skimage.transform.warp(
    np.dstack([ddy, ddx]),
    ttt,
    preserve_range=True,
    output_shape=padded_shape,
    cval=0,
)
mmm = np.moveaxis(ddy_ddx, 2, 0) + np.mgrid[:padded_shape[0], :padded_shape[1]]

warped_moving2 = skimage.transform.warp(
    moving,
    mmm,
    preserve_range=True,
    output_shape=padded_shape,
    cval=np.percentile(moving, 99.5),
)



# ---------------------------------------------------------------------------- #
#                                 doesn't work                                 #
# ---------------------------------------------------------------------------- #


H, W = dform[0].shape
gy, gx = (
    skimage.transform.warp_coords(
        (
            Affine(scale=1 / 2)
            + Affine(matrix=mx)
            + Affine(translation=offset)
            + Affine(scale=2)
        ).inverse,
        dform[0].shape,
    )
    - np.mgrid[:H, :W]
)


dddd = np.array([gx, gy]) + dform

ddx, ddy = (
    # FIXME confirm whether it's generalizable!
    (np.linalg.inv(mx[:2, :2]) @ dddd.reshape(2, -1)).T @ mx_d[:2, :2]
).T.reshape(dddd.shape)


ddy_ddx = skimage.transform.warp(
    np.dstack([ddy, ddx]),
    np.linalg.inv(mx_d),
    preserve_range=True,
    output_shape=padded_shape,
    cval=0,
)

mapping = np.mgrid[: padded_shape[0], : padded_shape[1]].astype(
    "float32"
) + np.moveaxis(ddy_ddx, 2, 0)

warped_moving2 = skimage.transform.warp(
    moving,
    mapping,
    preserve_range=True,
    output_shape=padded_shape,
    cval=np.percentile(moving, 99.5),
)
