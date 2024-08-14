import pathlib
import sys

import cv2
import dask.array as da
import fire
import numpy as np
import palom
import skimage.transform
import tifffile


def _warp_coords_cv2(mx, row_slice, col_slice, out_dtype="float64"):
    assert mx.shape == (3, 3)
    xx, yy = (
        np.arange(*col_slice, dtype="float64"),
        np.arange(*row_slice, dtype="float64"),
    )
    grid = np.reshape(
        np.meshgrid(xx, yy, indexing="xy"),
        (2, 1, -1),
    ).T
    grid = cv2.transform(grid, mx[:2, :]).astype(out_dtype)

    return np.squeeze(grid).T.reshape(2, len(yy), len(xx))[::-1]


def warp_coords_cv2(mx, shape, dtype="float64"):
    return _warp_coords_cv2(mx, (0, shape[0]), (0, shape[1]), out_dtype=dtype)


def _wrap_cv2_large_proper(dform, img, mx, cval, module="cv2", block_info=None):
    assert module in ["cv2", "skimage"]
    assert mx.shape == (3, 3)
    assert dform.ndim == 3

    _, H, W = dform.shape

    dtype = "float64"

    _, rslice, cslice = block_info[0]["array-location"]

    if np.all(mx == 0):
        dform = np.array(dform)
    else:
        mgrid = _warp_coords_cv2(mx, rslice, cslice, out_dtype=dtype)
        # remap functions in opencv convert coordinates into 16-bit integer; for
        # large image/coordinates, slice the appropiate image block and
        # re-position the coordinate origin is required
        dform = np.array(dform) + mgrid

    # add extra pixel for linear interpolation
    rmin, cmin = np.floor(dform.min(axis=(1, 2))).astype("int") - 1
    rmax, cmax = np.ceil(dform.max(axis=(1, 2))).astype("int") + 1

    if np.any(np.asarray([rmax, cmax]) <= 0):
        return np.full((H, W), fill_value=cval, dtype=img.dtype)

    rmin, cmin = np.clip([rmin, cmin], 0, None)
    rmax, cmax = np.clip([rmax, cmax], None, img.shape)

    dform -= np.reshape([rmin, cmin], (2, 1, 1))
    # cast mapping down to 32-bit float for speed and compatibility
    dform = dform.astype("float32")

    img = np.array(img[rmin:rmax, cmin:cmax])

    if 0 in img.shape:
        return np.full((H, W), fill_value=cval, dtype=img.dtype)
    if module == "cv2":
        return cv2.remap(img, dform[1], dform[0], cv2.INTER_LINEAR, borderValue=cval)
    return skimage.transform.warp(img, dform, preserve_range=True, cval=cval).astype(
        img.dtype
    )


def run_transform(
    file_path,
    out_path,
    ref_file_path,
    affine_mx_path,
    deformation_field_path,
    pyramid_level=0,
):
    Affine = skimage.transform.AffineTransform

    # hairy settings for scaling...
    pyramid_level = pyramid_level
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
    offset = padding * np.divide(ref_shape, 4 ** (2 - pyramid_level) * 2**3)[::-1]

    out_dir = pathlib.Path(out_path).parent
    out_dir.mkdir(exist_ok=True, parents=True)

    # for mx, dform, ff in zip(mxs_to_first, dfs_to_first, file_paths):
    print(pathlib.Path(file_path).name)

    mx = np.loadtxt(affine_mx_path, delimiter=",")
    dform = tifffile.imread(deformation_field_path)

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
        .astype("float64")
    )

    mapping = da.zeros((2, *padded_shape), dtype="float64", chunks=1024)

    _tform = tform + Affine(scale=1 / downscale_dform)
    # add extra pixel for linear interpolation
    _mgrid = skimage.transform.warp_coords(_tform.inverse, np.add(ddy.shape, 1))

    _mgrid[:, : ddy.shape[0], : ddy.shape[1]] += np.array([ddy, ddx])

    gy_gx = da.array(
        [
            mapping.map_blocks(
                _wrap_cv2_large_proper,
                gg,
                mx=np.linalg.inv(Affine(scale=downscale_dform).params),
                cval=0,
                module="skimage",
                dtype="float64",
                drop_axis=0,
            )
            for gg in _mgrid
        ]
    )

    # the chunk size (256, 256, 3) isn't ideal to be loaded with dask; hard-code
    # the reading and axis swap
    moving = tifffile.imread(file_path, level=pyramid_level)
    moving = np.moveaxis(moving, 2, 0)

    mosaics = []
    for channel in moving:
        cval = np.percentile(channel[::10, ::10], 75).item()
        warped_moving = gy_gx.map_blocks(
            _wrap_cv2_large_proper,
            channel,
            mx=np.zeros((3, 3)),
            cval=cval,
            module="skimage",
            dtype="uint8",
            drop_axis=0,
        )
        mosaics.append(warped_moving)

    palom.pyramid.write_pyramid(
        mosaics,
        output_path=out_path,
        pixel_size=ref_reader.pixel_size,
        channel_names=list("RGB"),
        downscale_factor=4,
        compression="zlib",
        save_RAM=True,
        tile_size=1024,
    )
    return


"""
run_transform(
    "/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24524.ome.tif",
    "/n/scratch/users/y/yc296/17-TB-HE-registration/registered/LSP24524-elastix.ome.tif",
    "/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24521.ome.tif",
    "/n/scratch/users/y/yc296/17-TB-HE-registration/affine-mx/LSP24524-affine-matrix.csv",
    "/n/scratch/users/y/yc296/17-TB-HE-registration/deformation-field/LSP24524-elastix-deformation-field-xy.ome.tif",
    pyramid_level=1,
)
"""


def main():
    sys.exit(fire.Fire(run_transform))


if __name__ == "__main__":
    main()
