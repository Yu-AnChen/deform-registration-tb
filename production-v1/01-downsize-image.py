import pathlib

import palom
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


out_dir = pathlib.Path(r"Z:\yc296\computation\YC-20240801-soheil-3d-reg\8MPP-affine")
out_dir.mkdir(exist_ok=True, parents=True)

for ff in tqdm.tqdm(file_paths):
    reader = palom.reader.OmePyramidReader(ff)
    img = reader.pyramid[0][1].compute()
    img = palom.img_util.cv2_downscale_local_mean(img, 4)
    tifffile.imwrite(
        out_dir / reader.path.name.replace(".ome.tif", "-ori.ome.tif"),
        img,
        compression="zlib",
    )


for ff in file_paths:
    reader = palom.reader.OmePyramidReader(ff)
