import pathlib

import cv2
import dask.array as da
import dask.diagnostics
import numpy as np
import palom
import skimage.transform
import tifffile
import zarr
from numcodecs import Zstd


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


def _wrap_cv2_large_proper(
    dform, img, mx, cval, sigma=0, module="cv2", block_info=None
):
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

    crop_img = np.array(img[rmin:rmax, cmin:cmax])

    if 0 in crop_img.shape:
        return np.full((H, W), fill_value=cval, dtype=img.dtype)

    if sigma != 0:
        pad = sigma * 4
        pad_rmin, pad_cmin = np.clip(np.subtract([rmin, cmin], pad), 0, None)
        pad_rmax, pad_cmax = np.clip(np.add([rmax, cmax], pad), None, img.shape)
        _crop_img = np.array(img[pad_rmin:pad_rmax, pad_cmin:pad_cmax])
        border_mode = cv2.BORDER_REPLICATE
        _crop_img = cv2.GaussianBlur(_crop_img, (0, 0), sigma, borderType=border_mode)
        crop_img = _crop_img[
            rmin - pad_rmin : rmin - pad_rmin + crop_img.shape[0],
            cmin - pad_cmin : cmin - pad_cmin + crop_img.shape[1],
        ]

    if 0 in img.shape:
        return np.full((H, W), fill_value=cval, dtype=img.dtype)
    if module == "cv2":
        return cv2.remap(
            crop_img, dform[1], dform[0], cv2.INTER_LINEAR, borderValue=cval
        )
    return skimage.transform.warp(
        crop_img, dform, preserve_range=True, cval=cval
    ).astype(crop_img.dtype)


def run_transform(
    file_path: str,
    out_path: str,
    ref_file_path: str,
    affine_mx_path: str,
    deformation_field_path: str,
    temp_zarr_store_dir: str = None,
    pre_filter_sigma: int = 1,
    pyramid_level: int = 0,
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
    r2 = palom.reader.OmePyramidReader(file_path)
    _moving = r2.pyramid[pyramid_level]
    chunks = np.ceil(np.divide(2048, _moving.chunksize[1:3])) * np.array(
        _moving.chunksize[1:3]
    )
    store = None
    if temp_zarr_store_dir is not None:
        store = zarr.TempStore(dir=temp_zarr_store_dir)
    moving = zarr.group(store=store, overwrite=True)
    for idx, channel in enumerate(_moving):
        moving[idx] = zarr.empty(
            channel.shape,
            chunks=chunks.astype("int"),
            dtype=_moving.dtype,
            compressor=Zstd(),
        )
        with dask.diagnostics.ProgressBar():
            channel.to_zarr(moving[idx])

    mosaics = []
    for channel in moving.values():
        cval = 0.0
        warped_moving = gy_gx.map_blocks(
            _wrap_cv2_large_proper,
            channel,
            mx=np.zeros((3, 3)),
            cval=cval,
            sigma=pre_filter_sigma,
            module="skimage",
            dtype="float32",
            drop_axis=0,
        )
        mosaics.append(warped_moving)

    palom.pyramid.write_pyramid(
        mosaics,
        output_path=out_path,
        pixel_size=ref_reader.pixel_size * 4**pyramid_level,
        channel_names=["proba-0", "proba-1"],
        downscale_factor=4,
        compression="zlib",
        save_RAM=True,
        tile_size=1024,
    )


def run_batch(csv_path, print_args=True, dryrun=False, **kwargs):
    import csv
    import inspect
    import pprint
    import types

    from fire.parser import DefaultParseValue

    func = run_transform

    if print_args:
        _args = [str(vv) for vv in inspect.signature(func).parameters.values()]
        print(f"\nFunction args\n{pprint.pformat(_args, indent=4)}\n")
    _arg_types = inspect.get_annotations(func)
    arg_types = {}
    for k, v in _arg_types.items():
        if isinstance(v, types.UnionType):
            v = v.__args__[0]
        arg_types[k] = v

    with open(csv_path) as f:
        csv_kwargs = [
            {
                kk: arg_types[kk](DefaultParseValue(vv))
                for kk, vv in rr.items()
                if (kk in arg_types) & (vv is not None)
            }
            for rr in csv.DictReader(f)
        ]

    if dryrun:
        for kk in csv_kwargs:
            pprint.pprint({**kwargs, **kk}, sort_dicts=False)
            print()
        return

    for kk in csv_kwargs:
        func(**{**kwargs, **kk})


def main():
    import fire

    fire.Fire({"run": run_transform, "run-batch": run_batch})


if __name__ == "__main__":
    main()


ref_file_path = r"/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24521.tif"

file_paths = r"""
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24521.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24524.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24527.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24530.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24536.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24539.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24542.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24545.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24548.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24551.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24554.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24557.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24560.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24563.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24566.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24569.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24572.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24575.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24578.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24581.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24584.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24587.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24590.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24593.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24596.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24599.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24602.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24605.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24608.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24611.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24614.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24617.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24620.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24623.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24626.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24629.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24632.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24635.tif
/n/scratch/users/y/yc296/17-TB-HE-registration/LSP24638.tif
""".strip().split("\n")[4:]


df_dir = pathlib.Path(
    r"/n/scratch/users/y/yc296/17-TB-HE-registration/deformation-field"
)
df_paths = [
    df_dir
    / pathlib.Path(ff).name.replace(".tif", "-elastix-deformation-field-xy.ome.tif")
    for ff in file_paths
]

mx_dir = pathlib.Path(r"/n/scratch/users/y/yc296/17-TB-HE-registration/affine-mx")
mx_paths = [
    mx_dir / pathlib.Path(ff).name.replace(".tif", "-affine-matrix.csv")
    for ff in file_paths
]

out_dir = pathlib.Path(r"/n/scratch/users/y/yc296/17-TB-HE-registration/registered")
out_dir.mkdir(exist_ok=True, parents=True)
output_paths = [
    out_dir / pathlib.Path(ff).name.replace(".tif", "-elastix.ome.tif")
    for ff in file_paths
]


for ff, oo, mm, dd in zip(file_paths[:1], output_paths, mx_paths, df_paths):
    run_transform(
        file_path=ff,
        out_path=oo,
        ref_file_path=ref_file_path,
        affine_mx_path=mm,
        deformation_field_path=dd,
        pyramid_level=0,
    )


"""
cd /n/scratch/users/y/yc296/17-TB-HE-registration

sbatch --mem=20G -c 8 -t 0-00:30 -p short --wrap="/home/yc296/mambaforge/envs/pyimagej/bin/python /home/yc296/project/20240813-17-TB-HE-registration/05-apply-transform-full-res-o2.py run LSP24521.tif registered/LSP24521-elastix.ome.tif LSP24521.tif affine-mx/LSP24521-affine-matrix.csv deformation-field/LSP24521-elastix-deformation-field-xy.ome.tif"
sbatch --mem=20G -c 8 -t 0-00:30 -p short --wrap="/home/yc296/mambaforge/envs/pyimagej/bin/python /home/yc296/project/20240813-17-TB-HE-registration/05-apply-transform-full-res-o2.py run LSP24524.tif registered/LSP24524-elastix.ome.tif LSP24521.tif affine-mx/LSP24524-affine-matrix.csv deformation-field/LSP24524-elastix-deformation-field-xy.ome.tif"
sbatch --mem=20G -c 8 -t 0-00:30 -p short --wrap="/home/yc296/mambaforge/envs/pyimagej/bin/python /home/yc296/project/20240813-17-TB-HE-registration/05-apply-transform-full-res-o2.py run LSP24527.tif registered/LSP24527-elastix.ome.tif LSP24521.tif affine-mx/LSP24527-affine-matrix.csv deformation-field/LSP24527-elastix-deformation-field-xy.ome.tif"
sbatch --mem=20G -c 8 -t 0-00:30 -p short --wrap="/home/yc296/mambaforge/envs/pyimagej/bin/python /home/yc296/project/20240813-17-TB-HE-registration/05-apply-transform-full-res-o2.py run LSPXXXXX.tif registered/LSPXXXXX-elastix.ome.tif LSP24521.tif affine-mx/LSPXXXXX-affine-matrix.csv deformation-field/LSPXXXXX-elastix-deformation-field-xy.ome.tif"









/home/yc296/mambaforge/envs/pyimagej/bin/python /home/yc296/project/20240813-17-TB-HE-registration/05-apply-transform-full-res-o2.py run LSP24521.tif registered/LSP24521-elastix.ome.tif LSP24521.tif affine-mx/LSP24521-affine-matrix.csv deformation-field/LSP24521-elastix-deformation-field-xy.ome.tif


file_path = r"\\research.files.med.harvard.edu\HITS\lsp-data\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\CD0302.08 (7923).svs"
ref_file_path = r"\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\STIC_Batch6_2023\p110_STIC\LSP19420\registration\LSP19420.ome.tif"
affine_mx_path = r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\thumbnail\LSP19420-affine-matrix.csv"
deformation_field_path = r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\deformation-field\LSP19420-elastix-deformation-field-xy.ome.tif"
out_path = r"\\research.files.med.harvard.edu\HITS\lsp-data\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\CD0302.08 (7923)-registered-to-LSP19420-test-1.ome.tif"
temp_zarr_store_dir = r"C:\Temp\temp-zarr-store"

run_transform(
    file_path=file_path,
    out_path=out_path,
    ref_file_path=ref_file_path,
    affine_mx_path=affine_mx_path,
    deformation_field_path=deformation_field_path,
    temp_zarr_store_dir=temp_zarr_store_dir,
    pre_filter_sigma=1,
)
"""
