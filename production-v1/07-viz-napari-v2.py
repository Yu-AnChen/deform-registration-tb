# ---------------------------------------------------------------------------- #
#                              palom reader class                              #
# ---------------------------------------------------------------------------- #
import itertools
import pathlib

import dask.array as da
import numpy as np
import ome_types
import pint
import tifffile
import zarr
import logging


logger = logging.getLogger("3d_napari")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)


class DaPyramidChannelReader:
    def __init__(self, pyramid: list[da.Array], channel_axis: int) -> None:
        self.pyramid = pyramid
        self.channel_axis = channel_axis
        if self.validate_pyramid(self.pyramid, self.channel_axis):
            self.pyramid = self.normalize_axis_order()

    @staticmethod
    def validate_pyramid(pyramid: list[da.Array], channel_axis: int) -> bool:
        for i, level in enumerate(pyramid):
            assert level.ndim == 3
            if np.argmin(level.shape) != channel_axis:
                logger.warning(
                    f"level {i} has shape of {level.shape} while given"
                    f" `channel_axis` is {channel_axis}"
                )
        return True

    def normalize_axis_order(self):
        if self.channel_axis == 0:
            return self.pyramid
        return [da.moveaxis(level, self.channel_axis, 0) for level in self.pyramid]

    def read_level_channels(self, level: int, channels: int | list[int]) -> da.Array:
        target_level = self.pyramid[level]
        return target_level[channels]

    @property
    def level_downsamples(self) -> dict[int, int]:
        heights = [ss.shape[1] for ss in self.pyramid]
        heights.insert(0, heights[0])
        downsamples = [round(h1 / h2) for h1, h2 in itertools.pairwise(heights)]
        return dict(enumerate(itertools.accumulate(downsamples, func=np.multiply)))

    @property
    def pixel_dtype(self) -> np.dtype:
        return self.pyramid[0].dtype

    def get_thumbnail_level_of_size(self, size: float) -> int:
        shapes = [np.abs(np.mean(level.shape[1:3]) - size) for level in self.pyramid]
        return np.argmin(shapes)


class OmePyramidReader(DaPyramidChannelReader):
    def __init__(
        self, path: str | pathlib.Path, pixel_size: float | None = None
    ) -> None:
        self.path = pathlib.Path(path)
        pyramid = self.pyramid_from_ometiff(self.path)
        channel_axis = 0
        self._pixel_size = pixel_size
        super().__init__(pyramid, channel_axis)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["pyramid"]
        state["path"] = state["path"].resolve()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__(path=state["path"], pixel_size=state["_pixel_size"])

    @staticmethod
    def pyramid_from_ometiff(path: str | pathlib.Path) -> list[da.Array]:
        with tifffile.TiffFile(path) as tif:
            num_series = len(tif.series)
            if num_series == 1:
                pyramid = tif.series[0].levels
            elif num_series > 1:
                pyramid = tif.series
            zarr_pyramid = [zarr.open(level.aszarr(), "r") for level in pyramid]
            da_pyramid = []
            for z in zarr_pyramid:
                if issubclass(type(z), zarr.hierarchy.Group):
                    da_level = da.from_zarr(z[0])
                else:
                    da_level = da.from_zarr(z)
                da_level = da_level.squeeze()
                if da_level.ndim == 2:
                    da_level = da_level.reshape(1, *da_level.shape)
                elif da_level.ndim == 3:
                    if da_level.shape[2] in (3, 4):
                        da_level = da.moveaxis(da_level, 2, 0)
                else:
                    raise ValueError(
                        f"Image with {da_level.ndim} dimension {da_level.shape} is not supported"
                    )
                da_pyramid.append(da_level)
        return da_pyramid

    @property
    def pixel_size(self) -> float:
        if self._pixel_size is not None:
            return self._pixel_size
        try:
            # ome-types v0.4 does not have `parser` kwarg in `from_tiff`
            import inspect

            kwargs = dict(path=self.path, validate=False)
            keys = inspect.signature(ome_types.from_tiff).parameters
            if "parser" in keys:
                kwargs.update(dict(parser="lxml"))
            ome = ome_types.from_tiff(**kwargs)
            px_size = ome.images[0].pixels.physical_size_x
            # convert length unit to µm
            unit = ome.images[0].pixels.physical_size_x_unit.value
            ureg = pint.UnitRegistry()
            px_size_micron = px_size * ureg(unit).to(ureg.micron).magnitude
            logger.info(f"Detected pixel size: {px_size_micron:.4f} µm")
            self._pixel_size = px_size_micron
            return self._pixel_size
        except Exception:
            logger.warning(
                f"Unable to parse pixel size from {self.path.name};"
                f" assuming 1 µm. Use `_pixel_size` to set it manually"
            )
            self._pixel_size = 1
            return self._pixel_size


# ---------------------------------------------------------------------------- #
#                    use napari (0.5.6) for 3d visualizaion                    #
# ---------------------------------------------------------------------------- #
import napari
import pathlib
import numpy as np
import dask.array as da


files = sorted(
    pathlib.Path(
        r"//research.files.med.harvard.edu/HITS/lsp-data/cycif-production/17-tuberculosis-aldridge/p17e21_3D_HE/YC-registered-HE-elastix/4x-downsize-zlib"
    ).glob("*.ome.tif")
)
files = filter(lambda x: "LSP24533" not in x.name, files)
readers = [OmePyramidReader(pp) for pp in files]
n_levels = len(readers[0].pyramid)

v = napari.Viewer()

v.add_image(
    [da.array([rr.pyramid[ii] for rr in readers]) for ii in range(n_levels)],
    channel_axis=1,
    colormap=list("rgb"),
    rendering="minip",
    scale=4 * np.array((200, 1, 1)),
    translate=np.array((0.5, 4 * 0.5, 4 * 0.5)),
    blending="additive",
)

proba_files = sorted(
    pathlib.Path(
        r"\\research.files.med.harvard.edu\HITS\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\production-v1\img-data-v2\elastix\LA_feature_maps\registered"
    ).glob("*.ome.tif")
)[:]
proba_readers = [OmePyramidReader(pp) for pp in proba_files]
n_levels_proba = len(proba_readers[0].pyramid)

v.add_image(
    [
        da.array([rr.pyramid[ii] for rr in proba_readers])
        for ii in range(n_levels_proba)
    ],
    channel_axis=1,
    colormap=["yellow", "green"],
    rendering="iso",
    scale=np.array((800, 1, 1)),
    translate=np.array((0.5, 0.5, 0.5)),
)
