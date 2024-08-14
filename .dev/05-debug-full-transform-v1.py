import pathlib

import cv2
import dask.array as da
import numpy as np
import palom
import skimage.transform
import tifffile


def _warp_coords_cv2(mx, row_slice, col_slice, dtype="float32"):
    assert mx.shape == (3, 3)
    xx, yy = (
        np.arange(*col_slice, dtype="float64"),
        np.arange(*row_slice, dtype="float64"),
    )
    grid = np.reshape(
        np.meshgrid(xx, yy, indexing="xy"),
        (2, 1, -1),
    ).T
    grid = cv2.transform(grid, mx[:2, :]).astype(dtype)

    return np.squeeze(grid).T.reshape(2, len(yy), len(xx))[::-1]


def warp_coords_cv2(mx, shape, dtype="float32"):
    return _warp_coords_cv2(mx, (0, shape[0]), (0, shape[1]), dtype=dtype)


def _wrap_cv2_large_proper(dform, img, mx, cval, module="cv2", block_info=None):
    assert module in ["cv2", "skimage"]
    assert mx.shape == (3, 3)

    dtype = "float32"

    _, rslice, cslice = block_info[0]["array-location"]

    mgrid = _warp_coords_cv2(mx, rslice, cslice, dtype)

    # remap functions in opencv convert coordinates into 16-bit integer; for
    # large image/coordinates, slice the appropiate image block and
    # re-position the coordinate origin is required
    dform = np.array(dform) + np.array(mgrid)
    # add extra pixel for linear interpolation
    rmin, cmin = np.floor(dform.min(axis=(1, 2))).astype("int") - 1
    rmax, cmax = np.ceil(dform.max(axis=(1, 2))).astype("int") + 1

    if np.any(np.asarray([rmax, cmax]) <= 0):
        return np.full(dform.shape[1:], fill_value=cval, dtype=img.dtype)

    rmin, cmin = np.clip([rmin, cmin], 0, None)
    rmax, cmax = np.clip([rmax, cmax], None, img.shape)

    dform -= np.reshape([rmin, cmin], (2, 1, 1))
    dform = dform.astype("float32")

    img = np.array(img[rmin:rmax, cmin:cmax])

    if 0 in img.shape:
        return np.full(dform.shape[1:], fill_value=cval, dtype=img.dtype)
    if module == "cv2":
        return cv2.remap(img, dform[1], dform[0], cv2.INTER_LINEAR, borderValue=cval)
    return np.round(skimage.transform.warp(img, dform, preserve_range=True)).astype(
        img.dtype
    )


ref_file_path = r"\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24521.ome.tif"

file_paths = r"""
\\research.files.med.harvard.edu\hits\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\ome-tiff images\LSP24521.ome.tif
""".strip().split("\n")[:1]

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


Affine = skimage.transform.AffineTransform

# hairy settings for scaling...
pyramid_level = 0
d_moving = (
    4 ** (2 - pyramid_level) * 2**3
)  # downsize factor used in the initial affine mx calculation
d_ref = (
    4 ** (2 - pyramid_level) * 2**3
)  # downsize factor used in the initial affine mx calculation
downscale_dform = 4 ** (2 - pyramid_level) * 2**2
mx_d = Affine(scale=downscale_dform).params


# add upper-left padding
padding = 0.1
ref_reader = palom.reader.OmePyramidReader(ref_file_path)
ref_shape = ref_reader.pyramid[pyramid_level].shape[1:]


padded_shape = np.ceil(np.multiply(ref_shape, 2 * padding + 1)).astype("int")
# translation will be applied post alignment to the first at the lower resolution
offset = padding * np.divide(ref_shape, 4**2 * 2**3)[::-1]


out_dir = pathlib.Path(
    r"Z:\yc296\computation\YC-20240801-soheil-3d-reg\registered-full-res-skimage"
)
out_dir.mkdir(exist_ok=True, parents=True)

for mx, dform, ff in zip(mxs_to_first, dfs_to_first, file_paths):
    print(pathlib.Path(ff).name)

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

    mapping = da.zeros((2, *padded_shape), dtype="float32", chunks=1024)

    _tform = tform + Affine(scale=1 / downscale_dform)
    # add extra pixel for linear interpolation
    _mgrid = np.zeros((2, *np.add(ddy.shape, 1)), dtype="float32")

    # NOTE temp remove non-linear transformation
    _mgrid[:, : ddy.shape[0], : ddy.shape[1]] += np.array([ddy, ddx])

    tmgrid = palom.align.block_affine_transformed_moving_img(
        ref_img=mapping[0],
        moving_img=_mgrid.astype("float32"),
        mxs=Affine(scale=downscale_dform).params,
        is_mask=False,
    )

    # the chunk size (256, 256, 3) isn't ideal to be loaded with dask; hard-code
    # the reading and axis swap
    moving = tifffile.imread(ff, level=pyramid_level)
    moving = np.moveaxis(moving, 2, 0)
    # moving = palom.reader.OmePyramidReader(ff).pyramid[pyramid_level]

    mosaics = []
    for channel in moving[:]:
        cval = 0
        warped_moving = tmgrid.map_blocks(
            _wrap_cv2_large_proper,
            img=channel,
            mx=np.linalg.inv(tform.params),
            cval=cval,
            dtype=moving.dtype,
            drop_axis=0,
        )
        mosaics.append(warped_moving)

    out_filename = pathlib.Path(ff).name.replace(".ome.tif", "-elastix.ome.tif")
    palom.pyramid.write_pyramid(
        mosaics,
        output_path=out_dir / out_filename,
        pixel_size=ref_reader.pixel_size,
        channel_names=list("RGB"),
        downscale_factor=4,
        compression="zlib",
        save_RAM=True,
        tile_size=1024,
    )
