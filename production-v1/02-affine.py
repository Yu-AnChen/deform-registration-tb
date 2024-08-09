import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import palom
import skimage.transform
import tifffile


def set_matplotlib_font(font_size=12):
    font_families = matplotlib.rcParams["font.sans-serif"]
    if font_families[0] != "Arial":
        font_families.insert(0, "Arial")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams.update({"font.size": font_size})


def save_all_figs(dpi=300, format="pdf", out_dir=None, prefix=None):
    figs = [plt.figure(i) for i in plt.get_fignums()]
    if prefix is not None:
        for f in figs:
            if f._suptitle:
                f.suptitle(f"{prefix} {f._suptitle.get_text()}")
            else:
                f.suptitle(prefix)
    names = [f._suptitle.get_text() if f._suptitle else "" for f in figs]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for f, n, nm in zip(figs, plt.get_fignums(), names):
        f.savefig(out_dir / f"{n}-{nm}.{format}", dpi=dpi, bbox_inches="tight")
        plt.close(f)


def set_subplot_size(w, h, ax=None):
    """w, h: width, height in inches"""
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


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
""".strip().split("\n")[:]


# write affine aligned image to file
out_dir = pathlib.Path(
    r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/production-v1/img-data/affine"
)
out_dir.mkdir(exist_ok=True, parents=True)


mxs = [np.eye(3)]
for f1, f2 in zip(file_paths[:-1], file_paths[1:]):
    r1 = palom.reader.OmePyramidReader(f1)
    r2 = palom.reader.OmePyramidReader(f2)
    print(r2.path.name)

    aligner = palom.align.Aligner(
        ref_img=r1.pyramid[0][0],
        moving_img=r2.pyramid[0][0],
        # perform feature detection and matching at 32x downsample (64 µm/pixel?!)
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
    mxs.append(aligner.coarse_affine_matrix)

    # optional: write QC plots to file
    set_matplotlib_font(font_size=8)
    fig, ax = plt.gcf(), plt.gca()
    fig.suptitle(f"{r2.path.name} (coarse alignment)", fontsize=8)
    ax.set_title(f"{r1.path.name} - {r2.path.name}", fontsize=6)
    # manually tweak the image contrast for this dataset
    ax.images[0].set_clim(vmin=6)
    im_h, im_w = ax.images[0].get_array().shape
    set_subplot_size(im_w / 144, im_h / 144, ax=ax)
    ax.set_anchor("N")
    # use 0.5 inch on the top for figure title
    fig.subplots_adjust(top=1 - 0.5 / fig.get_size_inches()[1])
    save_all_figs(out_dir=out_dir / "qc", format="jpg", dpi=144)

mxs_to_first = [
    np.linalg.multi_dot([np.eye(3)] + mxs[: ii + 1]) for ii, _ in enumerate(mxs)
]


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


# intensity percentile used to fill blank
cval_percentile = 99.5


for ff, mx in zip(file_paths, mxs_to_first):
    reader = palom.reader.OmePyramidReader(ff)
    img = palom.img_util.cv2_downscale_local_mean(
        reader.pyramid[moving_pyramid_level][0], 1
    )
    Affine = skimage.transform.AffineTransform

    tform = (
        Affine(scale=1 / d_moving)
        + Affine(matrix=mx)
        + Affine(translation=offset)
        + Affine(scale=d_ref)
    )
    wimg = skimage.transform.warp(
        img,
        tform.inverse,
        output_shape=np.ceil(padded_shape / d_output).astype("int"),
        cval=np.percentile(img, cval_percentile),
        preserve_range=True,
    )

    tifffile.imwrite(
        out_dir / reader.path.name.replace("-ori.ome.tif", "-affine.ome.tif"),
        np.floor(wimg).astype(img.dtype),
        compression="zlib",
    )
    np.savetxt(
        out_dir / reader.path.name.replace("-ori.ome.tif", "-affine-matrix.csv"),
        mx,
        delimiter=",",
    )
