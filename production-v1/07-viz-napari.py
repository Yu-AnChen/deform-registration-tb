import palom
import napari
import pathlib
import dask.array as da


folder_path = r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/production-v1/img-data/elastix/rgb"
# folder_path = r"Z:\yc296\computation\YC-20240801-soheil-3d-reg\2MPP-registered\rgb"

files = sorted(pathlib.Path(folder_path).glob("*.ome.tif"))

readers = [palom.reader.OmePyramidReader(ff) for ff in files]

v = napari.Viewer()

stack_pyramid = []
for ii in range(len(readers[0].pyramid)):
    stack_pyramid.append(da.array([rr.pyramid[ii] for rr in readers]))

v.add_image(stack_pyramid, channel_axis=1, colormap=['red', 'green', 'blue'])

stack_half = stack_pyramid[1][:, 1, ...].compute()


v2 = napari.Viewer()
v2.add_image(-stack_half, scale=(50, 1, 1), name="stack-half")