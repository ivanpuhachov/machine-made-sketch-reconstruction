import os
import warnings

import igl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import defaultdict
from camera_transformations import get_camera_matrices, get_pixel_coordinates, transform_normals_global_to_camera
from myTriangulation import myTraingulation
from refine_segmentation import refine_segmentation, report_segmentation_image, plot_segmentation_image
from pyomo_edge_optimization import edgeCutTriangulation, edgeCutOptimization
from pyomo_vertex_positions_opt import vertexOpt

from refine_me import refine, build_fillmap_correspondance, log3d, report_values, do_local_fits
import pickle
import json
from timeit import default_timer as timer

from vis_preparations import write_logs_for_blender, write_npz_patchparams, make_a_report


def init_triangulation_routine(
        pngname,
        pixels_to_camera_coordinates: np.array,
        sampling_distance=10,
        flags='YYqpa50',
):
    """
    Triangulate the region (pixels) and compute camera coordinates of vertices

    :param pngname:
    :param pixels_to_camera_coordinates:
    :param sampling_distance:
    :param flags:
    :return:
    """
    svgtriang, svg_points, svg_edges, svg_paths_edges = myTraingulation.from_svg(
        path_to_svg=f"results/{pngname}/{pngname}_clean.svg",
        svg_sampling_distance=sampling_distance,
        triang_flags=flags,
        # svg_sampling_distance=50,
        # triang_flags='YYqpa1500',
    )
    svgtriang.plot(faces=True, show=False, saveto=Path(f"results/{pngname}/reports/"), name="triang_init")

    triang_x = svgtriang.interpolate_f_on_vertices(f_grid=pixels_to_camera_coordinates[..., 0])
    triang_y = svgtriang.interpolate_f_on_vertices(f_grid=pixels_to_camera_coordinates[..., 1])
    triang_z = svgtriang.interpolate_f_on_vertices(f_grid=pixels_to_camera_coordinates[..., 2])

    with open(f"results/{pngname}/pkl/triangulation_logs.pkl", 'wb') as f:
        pickle.dump(
            [
                svgtriang,
                svg_points,
                svg_edges,
                svg_paths_edges,
                triang_x,
                triang_y,
                triang_z,
            ],
            f,
            protocol=-1,
        )
    return svgtriang, svg_points, svg_edges, svg_paths_edges, triang_x, triang_y, triang_z,


def init_trappedball_routine(
        image: np.array,
        raw_segmentation: np.array,
        pngname: str,
):
    """
    Run trappedball regions, refine segmentation on regions, region to patch type dict
    :param pngname: name to plot images
    :param image:
    :param raw_segmentation:
    :return:
    """
    refined_segm, dict_region_to_patch_type, fillmap, thin_fillmap = refine_segmentation(
        image=image,
        raw_segmentation=raw_segmentation,
        plot=True,
        saveto=f"results/{pngname}/",
        name=pngname,
    )

    trapped_ball_segmentation = np.copy(refined_segm)

    with open(Path(f"results/{pngname}/") / "segmentation_labels.json", 'w') as f:
        json.dump(
            obj=dict_region_to_patch_type,
            fp=f,
            ensure_ascii=False,
            indent=4,
        )

    label_keys = [
            "Background", "Plane", "Cylinder", "Cone", "Sphere",
            "Torus", "Revolution", "Extrusion", "BSpline", "Other",
        ]
    if os.path.exists(Path(f"results/{pngname}/") / "MANUAL_labels.json"):
        with open(Path(f"results/{pngname}/") / "MANUAL_labels.json", 'r') as f:
            manual_labeled_regions = json.load(f)
            # json wraps ints to str, so use int
            for k, v in manual_labeled_regions.items():
                dict_region_to_patch_type[int(k)] = v
                refined_segm[fillmap == int(k)] = label_keys.index(v)
            print(manual_labeled_regions)
            warnings.warn("USING MANUAL SEGMENTATION")
            report_segmentation_image(
                segm_image=refined_segm,
                title="manual segmentation",
                reportsdir=Path(f"results/{pngname}/reports/"),
            )

    list_unsupported_regions = ["Torus", "Revolution", "Extrusion", "BSpline", "Cone"]
    print(dict_region_to_patch_type.items())
    for region_id, label in dict_region_to_patch_type.items():
        if label in list_unsupported_regions:
            print(f"Region {region_id} has unsupported type {label}! Replace it with type Other")
            dict_region_to_patch_type[region_id] = "Other"
            refined_segm[fillmap == region_id] = label_keys.index("Other")

    with open(f"results/{pngname}/pkl/trappedball_logs.pkl", 'wb') as f:
        pickle.dump(
            [
                raw_segmentation,
                refined_segm,
                dict_region_to_patch_type,
                fillmap,
                thin_fillmap,
            ],
            f,
            protocol=-1,
        )

    plt.figure(figsize=(10, 3))
    plt.subplot(131)
    plot_segmentation_image(raw_segmentation)
    plt.title("raw")
    plt.subplot(132)
    plot_segmentation_image(trapped_ball_segmentation)
    plt.title("trapped ball refined")
    plt.subplot(133)
    plot_segmentation_image(refined_segm)
    plt.title("in use")
    plt.savefig(Path(f"results/{pngname}/reports/segmentation_report.png"), bbox_inches="tight", dpi=150)
    plt.close()

    plt.imsave(fname=f"results/{pngname}/reports/raw_trappedball.png", arr=fillmap, cmap="tab20b", vmin=0, vmax=20)

    return refined_segm, dict_region_to_patch_type, fillmap, thin_fillmap


def triang_trappedball_routine(
        triangulation: myTraingulation,
        svg_points,
        svg_edges,
        predicted_depth,
        fillmap: np.array,
        saveto=Path("reports/"),
        name="test_routine",
):
    """
    obtain regions to triangulation vertices correspondences

    :param triangulation:
    :param svg_points:
    :param svg_edges:
    :param predicted_depth:
    :param fillmap:
    :param saveto:
    :param name:
    :return:
    """
    # find pure junction idx
    # triangulation.plot_on_top_of_image(image=fillmap)
    _, _, junction_vertices, _ = \
        triangulation.build_region_to_vertex_dicts(fillmap=fillmap, plot=True, svg_edges=svg_edges,
                                                   name=name, saveto=saveto / "reports/")
    face_to_fillmap = triangulation.build_face_to_fillmap_id(fillmap=fillmap)
    # shuffle vertices such that vertex junction are at front
    vertex_junction_mask = np.zeros_like(triangulation.vertex_markers, dtype=int)
    vertex_junction_mask[[junction_vertices]] = 1
    triangulation.reshuffle_triangulation_vertices(use_this_marker=vertex_junction_mask)

    # build correspondances once more (this time true junctions are at front)
    dict_region_to_internal_triangulation_vertices_idx, dict_region_to_junction_triangulation_vertices_idx, junction_vertices, dict_vertex_to_region = \
        triangulation.build_region_to_vertex_dicts(fillmap=fillmap, plot=True, svg_edges=svg_edges,
                                                   name=name, saveto=saveto / "reports/")

    dict_junction_to_regions = defaultdict(list)
    for r, vs in dict_region_to_junction_triangulation_vertices_idx.items():
        for v in vs:
            dict_junction_to_regions[v].append(r)

    with open(saveto / "pkl" / "triang_regions_logs.pkl", 'wb') as f:
        pickle.dump(
            [
                junction_vertices,
                dict_region_to_junction_triangulation_vertices_idx,
                dict_region_to_internal_triangulation_vertices_idx,
                dict_junction_to_regions,  # contains only junction to region correspondence
                dict_vertex_to_region,  # contains all vertices to region correspondence
            ],
            f,
            protocol=-1,
        )
    with open(saveto / "pkl" / "face_to_fillmap.pkl", 'wb') as f:
        pickle.dump(
            [
                face_to_fillmap,
            ],
            f,
            protocol=-1,
        )
    return junction_vertices, dict_region_to_junction_triangulation_vertices_idx, \
        dict_region_to_internal_triangulation_vertices_idx, dict_junction_to_regions


def local_fits_routine(
        fillmap: np.array,
        pixels_to_camera_coordinates: np.array,
        camera_normals: np.array,
        dict_region_to_patch_type: dict,
        saveto: Path,
):
    temp_depth = np.copy(pixels_to_camera_coordinates[..., 2])
    foregr = temp_depth[fillmap != 1]
    temp_depth[fillmap == 1] = np.max(foregr)
    plt.imsave(fname=saveto / "reports/depthdata.png", arr=temp_depth, cmap="binary_r", )

    fillmap, refined_pixelsZ, dict_trapregion_to_type, dict_patch_to_params = do_local_fits(
        fillmap=fillmap,
        pixels_to_camera_coordinates=pixels_to_camera_coordinates,
        camera_normals=camera_normals,
        dict_trapregion_to_type=dict_region_to_patch_type,
        saveto=saveto,
    )

    write_npz_patchparams(dict_patch_to_params, dict_trapregion_to_type, file_to_write=saveto / "npz/localfits.npz")

    with open(saveto / "pkl/localfit_logs.pkl", 'wb') as f:
        pickle.dump(
            [
                refined_pixelsZ,
                dict_trapregion_to_type,
                dict_patch_to_params,
            ],
            f,
            protocol=-1,
        )
    return refined_pixelsZ, dict_trapregion_to_type, dict_patch_to_params


def main(
        pngname: str,
        sampling_distance=10,
        triang_flags='YYqpa50',
        n_iterations=10000,
        logpixels=False,
        logtriang=True,
):
    start_time = timer()
    workfolder = Path(f"results/{pngname}/")
    report_folder = workfolder / "reports"
    pkl_folder = workfolder / "pkl"
    os.makedirs(report_folder, exist_ok=True)
    os.makedirs(pkl_folder, exist_ok=True)
    os.makedirs(workfolder / "predicted_patches", exist_ok=True)
    os.makedirs(workfolder / "final_patches", exist_ok=True)

    pred_depth_data = np.load(str(workfolder / "npz" / f"{pngname}_depth.npz"))
    pred_depth = pred_depth_data["depth"]
    input_image = pred_depth_data["sketch"]

    pred_segmentation_data = np.load(str(workfolder / "npz" / f"{pngname}_segm.npz"))
    pred_segmentation = pred_segmentation_data["classes"]

    pred_normals_data = np.load(str(workfolder / "npz" / f"{pngname}_norms.npz"))
    pred_normals = pred_normals_data["normals"]

    pixels_to_camera_coordinates = get_pixel_coordinates(depth_values=pred_depth)
    camera_normals = transform_normals_global_to_camera(global_normals=pred_normals)

    # log3d(
    #     background_mask=pred_segmentation == 0,
    #     depth_camera=pred_depth,
    #     segmentation=pred_segmentation,
    #     normals_camera=transform_normals_global_to_camera(global_normals=pred_normals),
    #     name=f"{pngname}/0_{pngname}_pixel_predicted"
    # )

    svgtriang, svg_points, svg_edges, svg_paths_edges, triang_x, triang_y, triang_z, =\
        init_triangulation_routine(
            pngname=pngname,
            pixels_to_camera_coordinates=pixels_to_camera_coordinates,
            flags=triang_flags,
            sampling_distance=sampling_distance,
    )

    refined_segm, dict_region_to_patch_type, fillmap, thin_fillmap = init_trappedball_routine(
        image=input_image,
        raw_segmentation=pred_segmentation,
        pngname=pngname,
    )

    junction_vertices, dict_region_to_junction_triangulation_vertices_idx, \
        dict_region_to_internal_triangulation_vertices_idx, dict_junction_to_regions =\
        triang_trappedball_routine(
            triangulation=svgtriang,
            svg_points=svg_points,
            svg_edges=svg_edges,
            fillmap=fillmap,
            predicted_depth=pred_depth,
            saveto=workfolder,
            name=pngname,
        )

    local_fits_routine(
        fillmap=fillmap,
        pixels_to_camera_coordinates=pixels_to_camera_coordinates,
        camera_normals=camera_normals,
        dict_region_to_patch_type=dict_region_to_patch_type,
        saveto=workfolder,
    )

    edgeCutOptimization(
        pngname=pngname,
        max_iterations=n_iterations,
    )

    edgeCutTriangulation(
        pngname=pngname,
        triang_flags=triang_flags,
    )

    vertexOpt(
        pngname=pngname,
    )

    write_logs_for_blender(pngname=pngname)
    make_a_report(pngname=pngname, target="output/")

    end_time = timer()
    print(f"Script for {pngname} - time {end_time - start_time} seconds")


if __name__ == "__main__":
    myname = "line_armchair1_0_clean"
    parser = argparse.ArgumentParser(description="Run all our scripts")
    parser.add_argument(
        "--pngname",
        type=str,
        default=myname,
        help="name of image",
    )
    parser.add_argument("--maxiter", type=int, default=10000, help="number of iterations")
    args = parser.parse_args()

    main(
        pngname=args.pngname,
        n_iterations=args.maxiter,
    )
