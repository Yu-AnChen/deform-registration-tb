import numpy as np
import dask.array as da
import skimage.transform
import dask.diagnostics
import zarr


def da_warp_coords(mx, shape, chunks=1024, dtype="float32"):
    assert len(shape) == 2
    assert mx.shape == (3, 3)

    out = da.empty((2, *shape), chunks=chunks, dtype=dtype)

    def warp_coord_local(img, mx, block_info=None):
        _, rslice, cslice = block_info[0]["array-location"]

        coords_xyz = np.concatenate(
            [
                np.ones_like(img[0], dtype=dtype),
                *np.meshgrid(
                    np.arange(*rslice, dtype=dtype),
                    np.arange(*cslice, dtype=dtype),
                    indexing="ij",
                ),
            ][::-1]
        ).reshape(3, -1)

        return np.dot(coords_xyz.T, mx.T).T[:2][::-1].reshape(img.shape).astype(dtype)

    return out.map_blocks(warp_coord_local, mx=mx, dtype=dtype)


def da_to_zarr(da_img, zarr_store=None, num_workers=None, out_shape=None, chunks=None):
    if zarr_store is None:
        if out_shape is None:
            out_shape = da_img.shape
        if chunks is None:
            chunks = da_img.chunksize
        zarr_store = zarr.create(
            out_shape, chunks=chunks, dtype=da_img.dtype, overwrite=True
        )
    with dask.diagnostics.ProgressBar():
        da_img.to_zarr(zarr_store, compute=False).compute(num_workers=num_workers)
    return zarr_store


tform = skimage.transform.AffineTransform(
    scale=(0.9, 1.1), rotation=np.deg2rad(5), translation=(3, 17)
)

ooo = da_to_zarr(
    da_warp_coords(np.linalg.inv(tform.params), (20_000, 20_000), 1024),
)


# ---------------------------------------------------------------------------- #
#                                     dev3                                     #
# ---------------------------------------------------------------------------- #
import numpy as np
import skimage.transform


tform = skimage.transform.AffineTransform(
    scale=(0.9, 1.1), rotation=np.deg2rad(5), translation=(3, 17)
)

ref1 = skimage.transform.warp_coords(tform.inverse, (50, 50))
ref2 = skimage.transform.warp(np.mgrid[:50, :50][0], tform.inverse, preserve_range=True)

tform2 = tform + skimage.transform.AffineTransform(scale=1 / 5)

test1 = skimage.transform.warp_coords(tform2.inverse, (11, 11))
test2 = skimage.transform.warp(
    test1[0],
    skimage.transform.AffineTransform(scale=5).inverse,
    preserve_range=True,
    output_shape=(50, 50),
)

# ---------------------------------------------------------------------------- #
#                                      dev                                     #
# ---------------------------------------------------------------------------- #

import palom
import numpy as np
import dask.array as da
import skimage.transform
import dask.diagnostics
import zarr
import cv2

tform = skimage.transform.AffineTransform(
    matrix=np.array(
        [
            [1.03334245e00, 8.97292266e-04, 1.62444507e03],
            [2.86208660e-02, 1.05341145e00, 4.06258662e03],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
)

ref = skimage.transform.warp_coords(tform.inverse, (1000, 1000))

Affine = skimage.transform.AffineTransform

downscale_dform = 16
padded_shape = np.array([19656, 21490])
mgrid = da.meshgrid(
    da.arange(padded_shape[0], dtype="float32", chunks=1024),
    da.arange(padded_shape[1], dtype="float32", chunks=1024),
    indexing="ij",
)

_tform = tform + Affine(scale=1 / downscale_dform)
_shape = 1 + np.ceil(np.divide(padded_shape, downscale_dform)).astype(int)
_mgrid = skimage.transform.warp_coords(_tform.inverse, _shape)

tmgrid = palom.align.block_affine_transformed_moving_img(
    ref_img=mgrid[0],
    moving_img=_mgrid,
    mxs=Affine(scale=downscale_dform).params,
    is_mask=False,
)

ref1 = skimage.transform.warp(
    _mgrid[0],
    Affine(scale=downscale_dform).inverse,
    preserve_range=True,
    output_shape=(1000, 1000),
)

ref2 = cv2.warpAffine(
    _mgrid[0],
    Affine(scale=downscale_dform).params[:2, :],
    (1000, 1000),
    flags=cv2.INTER_LINEAR,
)

test1 = tmgrid.compute()[:, :1000, :1000]


downscale_dform = 4
_tform = tform + Affine(scale=1 / downscale_dform)
_shape = 1 + np.ceil(np.divide(padded_shape, downscale_dform)).astype(int)
_mgrid = skimage.transform.warp_coords(_tform.inverse, _shape)

tmgrid = palom.align.block_affine_transformed_moving_img(
    ref_img=mgrid[0],
    moving_img=_mgrid,
    mxs=Affine(scale=downscale_dform).params,
    is_mask=False,
)

test2 = tmgrid.compute()[0, :1000, :1000]



# ---------------------------------------------------------------------------- #
#                                     dev 2                                    #
# ---------------------------------------------------------------------------- #

import skimage.data
import skimage.transform
import numpy as np

img = skimage.data.astronaut()[..., 1]

Affine = skimage.transform.AffineTransform

tform_affine = Affine(
    translation=(100, 50), rotation=np.deg2rad(-15), scale=(1.8, 2.3), shear=0.2
)
img_affine = skimage.transform.warp(img, tform_affine.inverse)

mgrid = np.mgrid[: img.shape[0], : img.shape[1]].astype("float32")
my, mx = mgrid

deformation_field = 1e-3 * (mgrid - 128) * np.linalg.norm(mgrid - 128, axis=0)
dy, dx = deformation_field

img_deform = skimage.transform.warp(img_affine, np.array([dy + my, dx + mx]))

ddx, ddy = (
    np.linalg.inv(tform_affine.params[:2, :2]) @ np.array([dx.flat, dy.flat])
).reshape(2, 512, 512)

skimage.transform.warp(
    img,
    np.array(
        [
            skimage.transform.warp(my, tform_affine.inverse, preserve_range=True) + ddy,
            skimage.transform.warp(mx, tform_affine.inverse, preserve_range=True) + ddx,
        ]
    ),
)


# ---------------------------------------------------------------------------- #
#                                     dev 1                                    #
# ---------------------------------------------------------------------------- #

ddx, ddy = (
    np.linalg.inv(tform_affine.params[:2, :2]) @ np.array([dx.flat, dy.flat])
).reshape(2, 512, 512)
skimage.transform.warp(
    img,
    np.array(
        [
            skimage.transform.warp(
                skimage.transform.warp(ddy, tform_affine, preserve_range=True) + my,
                tform_affine.inverse,
                preserve_range=True,
            ),
            skimage.transform.warp(
                skimage.transform.warp(ddx, tform_affine, preserve_range=True) + mx,
                tform_affine.inverse,
                preserve_range=True,
            ),
        ]
    ),
)


# ---------------------------------------------------------------------------- #
#                                     dev 2                                    #
# ---------------------------------------------------------------------------- #

ady = skimage.transform.warp(dy, tform_affine, preserve_range=True)
adx = skimage.transform.warp(dx, tform_affine, preserve_range=True)

skimage.transform.warp(img, np.array([ady + my, adx + mx]))


skimage.transform.warp(
    -img,
    np.array(
        [
            dy + skimage.transform.warp(my, tform_affine.inverse, preserve_range=True),
            dx + skimage.transform.warp(mx, tform_affine.inverse, preserve_range=True),
        ]
    ),
)


fff = skimage.transform.warp(
    img,
    np.array(
        [
            skimage.transform.warp(my, tform_affine.inverse, preserve_range=True)
            - my
            + dy
            + my,
            skimage.transform.warp(mx, tform_affine.inverse, preserve_range=True)
            - mx
            + dx
            + mx,
        ]
    ),
)

# v.add_image(skimage.transform.warp(fff, np.array([dy + my, dx + mx])))
