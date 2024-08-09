import functools
import pathlib

import numpy as np
import palom
import skimage.transform
import tifffile
import tqdm

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
""".strip().split("\n")[:]

mxs = [np.eye(3)]
for f1, f2 in zip(file_paths[:-1], file_paths[1:]):
    r1 = palom.reader.OmePyramidReader(f1)
    r2 = palom.reader.OmePyramidReader(f2)
    print(r2.path.name)

    aligner = palom.align.Aligner(
        ref_img=r1.pyramid[2][1],
        moving_img=r2.pyramid[2][1],
        # perform feature detection and matching at 32x downsample (64 µm/pixel?!)
        ref_thumbnail=palom.img_util.cv2_downscale_local_mean(r1.pyramid[4][1], 2),
        moving_thumbnail=palom.img_util.cv2_downscale_local_mean(r2.pyramid[4][1], 2),
        ref_thumbnail_down_factor=2 ** (5 - 2),
        moving_thumbnail_down_factor=2 ** (5 - 2),
    )
    aligner.coarse_register_affine(
        n_keypoints=10_000,
        test_flip=False,
        test_intensity_invert=False,
        auto_mask=True,
    )
    mxs.append(aligner.affine_matrix)

# add upper-left padding
ul_padding = skimage.transform.AffineTransform(translation=(250, 0)).params
mxs_to_first = [
    functools.reduce(np.dot, mxs[:i] + [ul_padding]) for i in range(1, len(mxs) + 1)
]

shape = palom.reader.OmePyramidReader(file_paths[0]).pyramid[2][1].shape

out_dir = pathlib.Path(r"Z:\yc296\computation\YC-20240801-soheil-3d-reg\8MPP-affine")
out_dir.mkdir(exist_ok=True, parents=True)

for ff in tqdm.tqdm(file_paths):
    reader = palom.reader.OmePyramidReader(ff)
    img = reader.pyramid[2][1].compute()
    tifffile.imwrite(
        out_dir / reader.path.name.replace(".ome.tif", "-ori.ome.tif"),
        img,
        compression="zlib",
    )

for ff, mm in zip(file_paths[:], tqdm.tqdm(mxs_to_first)):
    reader = palom.reader.OmePyramidReader(ff)
    img = reader.pyramid[2][1].compute()
    moving = skimage.transform.warp(
        img,
        skimage.transform.AffineTransform(matrix=mm).inverse,
        preserve_range=True,
        # account for upper-left padding and lower right padding
        output_shape=np.array(shape) + 500,
        cval=np.percentile(img, 99),
    )
    moving = np.floor(moving).astype(img.dtype)
    tifffile.imwrite(
        out_dir / reader.path.name.replace(".ome.tif", "-affine.ome.tif"),
        moving,
        compression="zlib",
    )

# ---------------------------------------------------------------------------- #
#                               check with napari                              #
# ---------------------------------------------------------------------------- #

import napari  # noqa: E402

v = napari.Viewer()

for ff, mm in zip(file_paths, mxs_to_first):
    reader = palom.reader.OmePyramidReader(ff)
    v.add_image(
        [-pp[1] for pp in reader.pyramid[2:]],
        affine=palom.img_util.to_napari_affine(mm),
        blending="additive",
        visible=False,
        colormap="bop orange",
    )


# ---------------------------------------------------------------------------- #
#                         warp with displacement field                         #
# ---------------------------------------------------------------------------- #

# scikit-image use "float64" internally
my, mx = np.mgrid[: shape[0], : shape[1]].astype("float32")

tmy = skimage.transform.warp(
    my,
    skimage.transform.AffineTransform(matrix=aligner.affine_matrix).inverse,
    preserve_range=True,
)

tmx = skimage.transform.warp(
    mx,
    skimage.transform.AffineTransform(matrix=aligner.affine_matrix).inverse,
    preserve_range=True,
)

warpped_moving_image = skimage.transform.warp(
    img,
    np.array([tmy, tmx]),
    preserve_range=True,
)
