import pathlib

import cv2
import dask.array as da
import numpy as np
import palom
import skimage.transform
import tifffile


def _wrap_cv2(dform, img, cval):
    dform = np.array(dform)
    return cv2.remap(img, dform[1], dform[0], cv2.INTER_LINEAR, borderValue=cval)


ref_file_path = r"\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24521.ome.tif"

file_paths = r"""
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24521.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24524.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24527.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24530.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24533.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24536.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24539.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24542.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24545.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24548.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24551.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24554.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24557.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24560.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24563.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24566.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24569.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24572.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24575.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24578.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24581.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24584.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24587.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24590.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24593.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24596.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24599.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24602.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24605.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24608.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24611.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24614.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24617.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24620.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24623.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24626.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24629.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24632.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24635.ome.tif
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24638.ome.tif
""".strip().split("\n")[:]

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
