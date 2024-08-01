import pathlib

import itk
import napari
import numpy as np
import palom
import skimage.metrics
import tifffile


def get_default_crc_params(
    grid_size: float = 80.0,
    sample_region_size: float = 300.0,
    sample_number_of_pixels: int = 4_000,
    number_of_iterations: int = 1_000,
):
    parameter_object = itk.ParameterObject.New()
    # deformation
    p = parameter_object.GetDefaultParameterMap("bspline")

    del p["FinalGridSpacingInPhysicalUnits"]

    p["ASGDParameterEstimationMethod"] = ["DisplacementDistribution"]
    p["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    p["HowToCombineTransforms"] = ["Compose"]
    p["Interpolator"] = ["BSplineInterpolator"]
    p["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    p["Transform"] = ["RecursiveBSplineTransform"]

    # metrics: higher weight on the bending energy panelty to reduce distortion
    p["Metric"] = ["AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"]
    p["Metric0Weight"] = ["1.0"]
    p["Metric1Weight"] = ["100.0"]

    # these should be pixel-size & image size related
    p["NumberOfResolutions"] = ["4"]
    p["GridSpacingSchedule"] = [f"{2**i}" for i in range(0, 4)[::-1]]
    p["FinalGridSpacingInVoxels"] = [f"{grid_size}", f"{grid_size}"]
    p["NumberOfSamplesForExactGradient"] = [f"{10_000}"]
    p["NumberOfSpatialSamples"] = [f"{sample_number_of_pixels}"]
    # p["NumberOfSpatialSamples"] = [f"{5000 // 2**i}" for i in range(4)[::-1]]
    p["UseRandomSampleRegion"] = ["true"]
    p["SampleRegionSize"] = [f"{sample_region_size}"]
    p["NumberOfHistogramBins"] = ["32"]

    # number if iterations in gradient descent
    p["MaximumNumberOfIterations"] = [f"{number_of_iterations}"]

    # must set to write result image, could be a bug?!
    p["WriteResultImage"] = ["true"]
    p["ResultImageFormat"] = ["tif"]

    return p


def run_one_setting(ref_path, moving_path, setting):
    ref = tifffile.imread(ref_path)
    moving = tifffile.imread(moving_path)
    return _run_one_setting(ref, moving, setting)


def _run_one_setting(ref, moving, setting):
    if setting is None:
        setting = {}
    elastix_parameter = itk.ParameterObject.New()
    elastix_parameter.AddParameterMap(
        elastix_parameter.GetDefaultParameterMap("rigid", 4)
    )
    elastix_parameter.AddParameterMap(get_default_crc_params(**setting))
    warpped_moving, transform_parameter = itk.elastix_registration_method(
        ref,
        moving,
        parameter_object=elastix_parameter,
        log_to_console=False,
    )
    return warpped_moving, transform_parameter, elastix_parameter


def try_conditions(ref_path, moving_path, conditions):
    warpped_moving, transform_parameter, elastix_parameter = None, None, None
    for cc in conditions:
        try:
            print(cc)
            warpped_moving, transform_parameter, elastix_parameter = run_one_setting(
                ref_path,
                moving_path,
                cc,
            )

        except RuntimeError:
            continue
        else:
            # check MI metrics of the warpped with the ref
            nmi = skimage.metrics.normalized_mutual_information
            downscale = palom.img_util.cv2_downscale_local_mean
            mi_ori = nmi(
                downscale(tifffile.imread(ref_path), 10),
                downscale(tifffile.imread(moving_path), 10),
            )
            mi_reg = nmi(
                downscale(tifffile.imread(ref_path), 10), downscale(warpped_moving, 10)
            )
            print(cc, f"{mi_ori:.5f} vs {mi_reg:.5f}")
            if mi_reg < mi_ori:
                continue
            else:
                break
    return warpped_moving, transform_parameter, elastix_parameter


def write_parameter(param_obj, out_dir, prefix=None):
    assert isinstance(param_obj, itk.ParameterObject)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or ""
    out_paths = []
    for idx in range(param_obj.GetNumberOfParameterMaps()):
        out_path = out_dir / f"{prefix}elastix-param-{idx}.txt"
        itk.ParameterObject.New().WriteParameterFile(
            param_obj.GetParameterMap(idx), str(out_path)
        )
        out_paths.append(out_path)
    return out_paths


def map_moving_points(points_xy, param_obj):
    return _map_points(points_xy, param_obj, is_from_moving=True)


def map_fixed_points(points_xy, param_obj):
    return _map_points(points_xy, param_obj, is_from_moving=False)


def _map_points(points_xy, param_obj, is_from_moving=True):
    import scipy.ndimage as ndi

    points = np.asarray(points_xy)
    assert points.shape[1] == 2
    assert points.ndim == 2

    shape = param_obj.GetParameterMap(0).get("Size")[::-1]
    shape = np.array(shape, dtype="int")

    deformation_field = itk.transformix_deformation_field(
        itk.GetImageFromArray(np.zeros(shape, dtype="uint8")), param_obj
    )
    dx, dy = np.moveaxis(deformation_field, 2, 0)

    if is_from_moving:
        inverted_fixed_point = itk.FixedPointInverseDisplacementFieldImageFilter(
            deformation_field,
            NumberOfIterations=10,
            Size=deformation_field.shape[:2][::-1],
        )
        dx, dy = np.moveaxis(inverted_fixed_point, 2, 0)

    my, mx = np.mgrid[: shape[0], : shape[1]].astype("float32")

    mapped_points_xy = np.vstack(
        [
            ndi.map_coordinates(mx + dx, np.fliplr(points).T),
            ndi.map_coordinates(my + dy, np.fliplr(points).T),
        ]
    ).T
    return mapped_points_xy


_file_paths = """
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_001-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_002-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_003-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_004-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_005-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_006-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_007-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_008-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_009-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_010-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_011-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_012-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_013-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_014-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_015-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_016-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_017-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_018-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_019-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_020-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_021-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_022-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_023-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_024-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_025-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_026-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_027-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_028-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_029-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_030-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_031-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_032-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_033-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_034-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_035-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_036-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_037-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_038-affine.ome.tif
/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/img-data/8MPP-affine/B5_3DHE_039-affine.ome.tif
""".strip().split("\n")

section_pairs = [
    (1, 0),
    (2, 1),
    (3, 2),
    (4, 3),
    (5, 4),
    (6, 4),
    (7, 6),
    (8, 7),
    (9, 8),
    (10, 9),
    (11, 9),
    (12, 11),
    (13, 11),
    (14, 11),
    (15, 11),
    (16, 11),
    (17, 16),
    (18, 17),
    (19, 18),
    (20, 19),
    (21, 20),
    (22, 21),
    (23, 22),
    (24, 23),
    (25, 24),
    (26, 25),
    (27, 26),
    (28, 27),
    (29, 28),
    (30, 29),
    (31, 30),
    (32, 31),
    (33, 32),
    (34, 33),
    (35, 34),
    (36, 35),
    (37, 36),
    (38, 37),
]

# difficult and manually configured pairs
# section_pairs = [
#     (6, 4),
#     (10, 9),
#     (11, 9),
#     (13, 11),
#     (16, 11),
#     (24, 23),
#     (25, 24),
#     (26, 25),
#     (27, 26),
#     (29, 28),
#     (31, 30),
#     (33, 32),
#     (36, 35),
#     (37, 36),
# ][:]

import itertools  # noqa: E402

sizes = [700, 800, 900]
n_samples = [4000, 3000]
grid_sizes = [20, 40, 80]
conditions = [
    {"sample_region_size": ss, "sample_number_of_pixels": ns, "grid_size": gs}
    for ss, ns, gs in itertools.product(sizes, n_samples, grid_sizes)
]

elastix_config_dir = "/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/reg-param-pair-manual/config"
elastix_tform_dir = "/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240729-deform-registration-soheil/reg-param-pair-manual/tform"

file_paths = [(_file_paths[i0], _file_paths[i1]) for i0, i1 in section_pairs]

v = napari.Viewer()
for f2, f1 in file_paths:
    ref_path = pathlib.Path(f1)
    moving_path = pathlib.Path(f2)
    ref_name = ref_path.name.split("-")[0]
    moving_name = moving_path.name.split("-")[0]
    print(moving_name)

    img, params_tform, params_reg = try_conditions(
        ref_path=ref_path, moving_path=moving_path, conditions=conditions
    )

    # write parameters to disk
    write_parameter(params_reg, elastix_config_dir, f"{moving_name}-to-{ref_name}-")
    write_parameter(
        params_tform, elastix_tform_dir, f"{moving_name}-to-{ref_name}-tform-"
    )

    napari_kwargs = dict(blending="additive", visible=False, name=moving_name)
    v.add_image(
        -tifffile.imread(ref_path), colormap="bop blue", visible=False, name=ref_name
    )
    v.add_image(-tifffile.imread(moving_path), colormap="bop purple", **napari_kwargs)
    v.add_image(-img, colormap="bop orange", **napari_kwargs)


# ---------------------------------------------------------------------------- #
#                                  dev section                                 #
# ---------------------------------------------------------------------------- #

# starting from lower res image and test what parameter works, then increase
# resolution
ref = tifffile.imread(file_paths[0])
moving = tifffile.imread(file_paths[1])

w1, *_ = _run_one_setting(
    palom.img_util.cv2_downscale_local_mean(ref, 4),
    palom.img_util.cv2_downscale_local_mean(moving, 4),
    None,
)
w2, *_ = _run_one_setting(
    palom.img_util.cv2_downscale_local_mean(ref, 2),
    palom.img_util.cv2_downscale_local_mean(moving, 2),
    dict(sample_region_size=400),
)
w3, *_ = _run_one_setting(ref, moving, dict(sample_region_size=700))

_run_one_setting(ref, moving, dict(sample_region_size=600))

import networkx as nx  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import cv2  # noqa: E402


v = napari.Viewer()
G = nx.DiGraph(section_pairs[:])
for mm, rr in section_pairs[:]:
    ref_path = pathlib.Path(_file_paths[rr])
    moving_path = pathlib.Path(_file_paths[mm])

    ref = tifffile.imread(ref_path)
    moving = tifffile.imread(moving_path)

    print(
        mm,
        moving_path.name.replace("-affine.ome.tif", ""),
        "-",
        rr,
        ref_path.name.replace("-affine.ome.tif", ""),
    )

    wimg, tform, _ = _run_one_setting(
        palom.img_util.cv2_downscale_local_mean(ref, 4),
        palom.img_util.cv2_downscale_local_mean(moving, 4),
        dict(sample_region_size=200),
    )

    G.edges[mm, rr]["tform"] = tform

    napari_kwargs = dict(
        blending="additive",
        visible=False,
        name=moving_path.name.replace("-affine.ome.tif", ""),
    )
    v.add_image(
        -palom.img_util.cv2_downscale_local_mean(ref, 4),
        colormap="bop blue",
        visible=False,
        name=ref_path.name.replace("-affine.ome.tif", ""),
    )
    v.add_image(-wimg, colormap="bop orange", **napari_kwargs)


# ---------------------------- visualize the graph --------------------------- #
options = {
    "font_size": 12,
    "node_size": 300,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 1,
    "width": 1,
}
pos = {
    ii: (ii, 10 * (ii % 2 - 0.5) + 2 * (np.random.rand() - 0.5))
    for ii in sorted(G.nodes)
}

plt.figure()
nx.draw_networkx(G, pos=pos, **options)

# use actual slide number instead of index in the list
G_viz = nx.DiGraph(
    [
        (
            pathlib.Path(_file_paths[mm])
            .name.replace("-affine.ome.tif", "")
            .replace("B5_3DHE_", ""),
            pathlib.Path(_file_paths[rr])
            .name.replace("-affine.ome.tif", "")
            .replace("B5_3DHE_", ""),
        )
        for mm, rr in section_pairs[:]
    ]
)
options = {
    "font_size": 12,
    "node_size": 700,
    "node_color": "none",
    "edgecolors": "none",
    "linewidths": 1,
    "width": 1,
}
pos = {
    ii: (idx, 10 * (idx % 2 - 0.5) + 6 * (np.random.rand() - 0.5))[::-1]
    for idx, ii in enumerate(sorted(G_viz.nodes))
}

plt.figure()
nx.draw_networkx(G_viz, pos=pos, **options)


# -------------------------- warp all image to first ------------------------- #
def to_deformation_field(elastix_parameter):
    shape = elastix_parameter.GetParameterMap(0).get("Size")[::-1]
    shape = np.array(shape, dtype="int")
    return itk.transformix_deformation_field(
        itk.GetImageFromArray(np.zeros(shape, dtype="uint8")), elastix_parameter
    )


def aggregate_deformation_field(elastix_parameters):
    shape = elastix_parameters[0].GetParameterMap(0).get("Size")[::-1]
    shape = np.array(shape, dtype="int")
    out = np.zeros((*shape, 2), dtype="float32")
    for elastix_parameter in elastix_parameters:
        out += to_deformation_field(elastix_parameter)
    return out


v = napari.Viewer()
for ii in sorted(G.nodes):
    moving_path = pathlib.Path(_file_paths[ii])
    moving_name = moving_path.name.replace("-affine.ome.tif", "")
    print(moving_name)
    moving = palom.img_util.cv2_downscale_local_mean(
        tifffile.imread(_file_paths[ii]), 4
    )
    sub_edges = G.subgraph(nx.descendants(G, ii).union({ii})).edges
    if sub_edges:
        deformation_field = np.moveaxis(
            aggregate_deformation_field([G.edges[ee]["tform"] for ee in sub_edges]),
            2,
            0,
        )
        _, H, W = deformation_field.shape
        dd = deformation_field + np.mgrid[:H, :W].astype("float32")[::-1]

        moving = cv2.remap(
            moving,
            dd[0],
            dd[1],
            cv2.INTER_LINEAR,
        )
    v.add_image(
        -moving,
        visible=True,
        blending="additive",
        name=moving_name,
        colormap="bop orange",
    )
