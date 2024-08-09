import pathlib

import cv2
import dask.array as da
import numpy as np
import palom
import skimage.transform
import tifffile


ref_file_path = (
    r"Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_001.ome.tif"
)

file_paths = r"""
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_001.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_002.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_003.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_004.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_005.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_006.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_007.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_008.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_009.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_010.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_011.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_012.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_013.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_014.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_015.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_016.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_017.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_018.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_019.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_020.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_021.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_022.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_023.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_024.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_025.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_026.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_027.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_028.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_029.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_030.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_031.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_032.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_033.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_034.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_035.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_036.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_037.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_038.ome.tif
Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP\B5_3DHE_039.ome.tif
""".strip().split("\n")[-1:]

mx_dir = pathlib.Path(r"Z:\yc296\computation\YC-20240801-soheil-3d-reg\affine-mx")
df_dir = pathlib.Path(
    r"Z:\yc296\computation\YC-20240801-soheil-3d-reg\deformation-field"
)


dfs_to_first = []
for ff in file_paths:
    ff = pathlib.Path(ff)
    moving_name = ff.name.replace(".ome.tif", "")
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
            mx_dir / ff.name.replace(".ome.tif", "-affine-matrix.csv"),
            delimiter=",",
        )
    )


# add upper-left padding
padding = 0.1
ref_reader = palom.reader.OmePyramidReader(ref_file_path)
ref_shape = ref_reader.pyramid[0].shape[1:]

padded_shape = np.ceil(np.multiply(ref_shape, 2 * padding + 1)).astype("int")
# translation will be applied post alignment to the first at the lower resolution
offset = padding * np.divide(ref_shape, 2 ** (5 - 0))[::-1]


# hairy settings for scaling...
d_pyramid = 2
moving_pyramid_level = 0
d_moving = d_pyramid**5  # same as aligner.moving_thumbnail_down_factor
d_ref = d_pyramid**5  # same as aligner.ref_thumbnail_down_factor
d_output = d_pyramid**moving_pyramid_level * d_moving / d_ref


out_dir = pathlib.Path(
    r"Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP-registered-v4"
)
out_dir.mkdir(exist_ok=True, parents=True)

Affine = skimage.transform.AffineTransform
for mx, dform, ff in zip(mxs_to_first, dfs_to_first, file_paths):
    print(pathlib.Path(ff).name)
    downscale_dform = 2**4
    mx_d = Affine(scale=downscale_dform).params

    tform = (
        Affine(scale=1 / d_moving)
        + Affine(matrix=mx)
        + Affine(translation=offset)
        + Affine(scale=d_ref)
    )

    ddx, ddy = (
        (
            # FIXME confirm whether it's the right math!
            (np.linalg.inv(mx[:2, :2]) @ dform.reshape(2, -1)).T @ mx_d[:2, :2]
        )
        .T.reshape(dform.shape)
        .astype("float32")
    )

    mgrid = da.meshgrid(
        da.arange(padded_shape[0], dtype="float32", chunks=1024),
        da.arange(padded_shape[1], dtype="float32", chunks=1024),
        indexing="ij",
    )

    _tform = tform + Affine(scale=1 / downscale_dform)
    _mgrid = skimage.transform.warp_coords(_tform.inverse, ddy.shape)

    _mgrid += np.array([ddy, ddx])

    tmgrid = palom.align.block_affine_transformed_moving_img(
        ref_img=mgrid[0],
        moving_img=_mgrid.astype("float32"),
        mxs=Affine(scale=downscale_dform).params,
        is_mask=False,
        # rescaling from a padded image, there shouldn't be empty pixels
        fill_empty=0,
    )

    moving = palom.reader.OmePyramidReader(ff).pyramid[0]

    def _wrap_cv2(dform, img, cval):
        dform = np.array(dform)
        return cv2.remap(img, dform[1], dform[0], cv2.INTER_LINEAR, borderValue=cval)

    mosaics = []
    for channel in moving:
        cval = np.percentile(np.asarray(channel), 75).item()
        warped_moving = tmgrid.map_blocks(
            _wrap_cv2,
            img=channel,
            cval=cval,
            dtype=moving.dtype,
            drop_axis=0,
        )
        mosaics.append(warped_moving)

    out_filename = pathlib.Path(ff).name.replace(".ome.tif", "-elastix.ome.tif")
    palom.pyramid.write_pyramid(
        mosaics,
        output_path=out_dir / out_filename,
        pixel_size=2,
        channel_names=list("RGB"),
        downscale_factor=4,
        compression="zlib",
        save_RAM=True,
        tile_size=1024,
    )
