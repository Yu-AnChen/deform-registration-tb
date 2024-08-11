import pathlib

import palom
import tifffile
import tqdm

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


out_dir = pathlib.Path(r"Z:\yc296\computation\YC-20240801-soheil-3d-reg\16x-downsize")
out_dir.mkdir(exist_ok=True, parents=True)

for ff in tqdm.tqdm(file_paths):
    reader = palom.reader.OmePyramidReader(ff)
    img = reader.pyramid[2][1].compute()
    tifffile.imwrite(
        out_dir / reader.path.name.replace(".ome.tif", "-ori.ome.tif"),
        img,
        compression="zlib",
    )


for ff in file_paths:
    reader = palom.reader.OmePyramidReader(ff)
