import igl
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from pathlib import Path
import argparse
from collections import defaultdict

from MyMesh import MyMesh, MyMesh3D
from test_cylinder_fit import CylinderFit, cylinder_alignment_geomfitty_xyz, cylinder_evalZ, plot_cylinder, \
    cylinder_estimate_r, cylinder_find_closest, cylinder_projectZ_or_closest
from sphere_fit import SphereFit, sphere_evalZ, sphere_alignment_energy2, sphere_projectZ_or_closest
from plane_fit import fit_plane_euclidean, plane_evalZ, plane_eval_dist, plane_eval_dist2, plane_eval_distZ2
from camera_transformations import get_camera_matrices, get_pixel_coordinates, transform_normals_global_to_camera
from myTriangulation import myTraingulation
from refine_segmentation import refine_segmentation
from utils_fillmap import dict_patch_template, build_region_to_junctions_dicts
from poisson_inflate import inflate_2d_mesh
from cutTriangulation import cutTriangulation
from smooth_patches import do_smoothing_iterations

from refine_me import refine, build_fillmap_correspondance, log3d, report_values
from improve_global_optimization import patches_global_optimization
import pickle


def log3d_svg(
        svgtri: myTraingulation,
        depth_camera,
        segmentation,
        name="testsvg",
        saveto=Path("results/"),
):
    """
    logs data in 3d. given triangulation, interpolates depth on vertices using depth camera grid and colors by segmentation
    :param svgtri:
    :param depth_camera:
    :param segmentation:
    :param name:
    :return:
    """
    vertex_colors = svgtri.get_vertex_class_from_segmentation(segm=segmentation)
    vertex_colors[:svgtri.n_svg_points] = 9  # highlight svg path vertices
    vertex_depth = svgtri.interpolate_f_on_vertices(f_grid=depth_camera)

    pixels_to_camera_coordinates = get_pixel_coordinates(depth_values=depth_camera)
    X = pixels_to_camera_coordinates[..., 0]
    Y = pixels_to_camera_coordinates[..., 1]
    Z = pixels_to_camera_coordinates[..., 2]
    triang_x = svgtri.interpolate_f_on_vertices(f_grid=X)
    triang_y = svgtri.interpolate_f_on_vertices(f_grid=Y)
    triang_z = - vertex_depth
    # triang_z = svgtri.interpolate_f_on_vertices(f_grid=-depth_camera)

    return log3d_mesh(
        v_x=triang_x,
        v_y=triang_y,
        v_z=triang_z,
        faces=np.copy(svgtri.faces),
        name=name,
        saveto=saveto,
        v_colors=vertex_colors,
    )


def log3d_mesh(
        v_x,
        v_y,
        v_z,
        faces,
        name="test",
        saveto=Path("results/"),
        v_colors=None,
):
    triang_3d_vertices = np.vstack((v_x, v_y, v_z)).transpose()
    print(triang_3d_vertices.shape)
    triang_mesh3d = MyMesh3D(vertices=triang_3d_vertices, faces=faces, vertex_markers=np.array([]),
                             holes=[])
    triang_mesh3d.export_obj(file_path=saveto / f"{name}.obj")
    # save ply
    if v_colors is None:
        v_colors = np.ones_like(v_x, dtype=int)
    triang_mesh3d.export_colored_ply(
        file_path=str(saveto / f"{name}.ply"),
        vertex_color_index=v_colors,
    )


def get_flat_adjacency_matrices(
        junction_vertices: np.array,
        path_to_flat_mesh: str,
):
    # flat_v, flat_f = igl.read_triangle_mesh(str(saveto / f"{name}_triang_cut_flat.obj"))
    flat_v, flat_f = igl.read_triangle_mesh(path_to_flat_mesh)
    flat_laplacian = igl.cotmatrix(flat_v, flat_f)
    flat_adjacent_matrix = igl.adjacency_matrix(flat_f)
    flat_adjacency_list = igl.adjacency_list(flat_f)
    slicing_matrix = scipy.sparse.csr_matrix((len(junction_vertices), flat_laplacian.shape[1]))
    for i in range(len(junction_vertices)):
        slicing_matrix[i, i] = 1
    flat_laplacian_junctions = slicing_matrix.dot(flat_laplacian).dot(slicing_matrix.transpose())
    flat_adjacency_junctions = slicing_matrix.dot(flat_adjacent_matrix).dot(slicing_matrix.transpose())
    for i in range(len(junction_vertices)):
        flat_laplacian_junctions[i, i] -= flat_laplacian_junctions.getrow(i).sum()
        flat_adjacency_junctions[i, i] -= flat_adjacency_junctions.getrow(i).sum()
    return flat_laplacian_junctions, flat_adjacency_list, flat_adjacency_junctions


def improve_me(
        image,
        predicted_depth,
        segmentation,
        svg_edges,
        svg_points,
        svg_paths_edges,
        fillmap,
        refined_pixelsZ,
        triangulation: myTraingulation,
        dict_trapregion_to_type: dict,
        dict_patch_to_params: dict,
        name: str,
        saveto=Path("results/"),
        plot=False,
        save3d=False,
        savenpz=False,
        inflate=False,
        maxiterations=500,
):
    n_patches = np.max(fillmap) + 1

    cut_triangulation = True

    lists_of_duplicates = []  # will contain vertices involved in mesh cut (a list for each chain), duplicates since they share XY coordinates
    list_of_chains = []  # will contain list of cutChains returned by cutTriangulation

    if not cut_triangulation:
        # find pure junction idx
        _, _, junction_vertices, _ = \
            triangulation.build_region_to_vertex_dicts(fillmap=fillmap, plot=True, name=name, saveto=saveto / "reports/")
        # shuffle vertices such that vertex junction are at front
        vertex_junction_mask = np.zeros_like(triangulation.vertex_markers, dtype=int)
        vertex_junction_mask[[junction_vertices]] = 1
        triangulation.reshuffle_triangulation_vertices(use_this_marker=vertex_junction_mask)

        # build correspondances once more (this time true junctions are at front)
        dict_region_to_internal_triangulation_vertices_idx, dict_region_to_junction_triangulation_vertices_idx, junction_vertices, dict_vertex_to_region = \
            triangulation.build_region_to_vertex_dicts(fillmap=fillmap, plot=True, name=name, saveto=saveto / "reports/")
    else:
        triangulation, dict_region_to_internal_triangulation_vertices_idx, \
        dict_region_to_junction_triangulation_vertices_idx, junction_vertices,\
        lists_of_duplicates, list_of_chains = cutTriangulation(
            triang=triangulation,
            svg_points=svg_points,
            svg_edges=svg_edges,
            depthimage=predicted_depth,
            fillmap=fillmap,
            saveto=saveto / "reports/",
        )

    dict_junction_to_regions = defaultdict(list)
    for r, vs in dict_region_to_junction_triangulation_vertices_idx.items():
        for v in vs:
            dict_junction_to_regions[v].append(r)

    dict_patch_to_image_junction_mask, dict_patch_to_image_junction_idx = build_region_to_junctions_dicts(
        fillmap=fillmap,
        name=name,
        plot=plot,
    )  # give us masks to get internal and junction pixels coordinates

    pixels_to_camera_coordinates = get_pixel_coordinates(depth_values=predicted_depth)

    pixelsX = pixels_to_camera_coordinates[..., 0]
    pixelsY = pixels_to_camera_coordinates[..., 1]
    pixelsZ = pixels_to_camera_coordinates[..., 2]

    dict_patch_to_x = build_fillmap_correspondance(fillmap,
                                                   gridvalues=pixelsX)  # contain X coordinates for points in a patch
    dict_patch_to_y = build_fillmap_correspondance(fillmap,
                                                   gridvalues=pixelsY)  # contain Y coordinates for points in a patch
    dict_patch_to_z = build_fillmap_correspondance(fillmap,
                                                   gridvalues=pixelsZ)  # contain Z coordinates for points in a patch

    triang_x = triangulation.interpolate_f_on_vertices(f_grid=pixelsX)
    triang_y = triangulation.interpolate_f_on_vertices(f_grid=pixelsY)
    triang_z = triangulation.interpolate_f_on_vertices(f_grid=pixelsZ)

    triang_vertex_class = triangulation.get_vertex_class_from_segmentation(segm=segmentation)
    triang_vertex_class[:triangulation.n_svg_points] = 9

    log3d_mesh(
        v_x=triang_x,
        v_y=triang_y,
        v_z=np.zeros_like(triang_x),
        faces=np.copy(triangulation.faces),
        name=f"{name}_triang_cut_flat",
        saveto=saveto,
        v_colors=triang_vertex_class,
    )

    with open(saveto / "triangulation_logs.pkl", 'wb') as f:
        pickle.dump(
            [
                triangulation,
                svg_points,
                svg_edges,
                svg_paths_edges,
                lists_of_duplicates,
                triang_x,
                triang_y,
                triang_z,
                list_of_chains,
                fillmap,
                predicted_depth,
            ],
            f,
            protocol=-1,
        )

    flat_laplacian_junctions, flat_adjacency_list, flat_adjacency_junctions = get_flat_adjacency_matrices(
        junction_vertices=junction_vertices,
        path_to_flat_mesh=str(saveto / f"{name}_triang_cut_flat.obj"),
    )

    if len(lists_of_duplicates) > 0:
        # if we did some cuts, then move vertices on cut boundary closer to their remaining neighbours
        for chain in lists_of_duplicates:
            for iv in chain:
                temp_z = 0
                temp_n = 0
                adj_vertices = flat_adjacency_list[iv]
                for adjacent_v in adj_vertices:
                    if adjacent_v >= len(junction_vertices):
                        temp_z += triang_z[adjacent_v]
                        temp_n += 1
                if temp_n > 0:
                    triang_z[iv] = temp_z / temp_n

    triang_refined_z = np.copy(triang_z)

    log3d_mesh(
        v_x=triang_x,
        v_y=triang_y,
        v_z=triang_z,
        faces=np.copy(triangulation.faces),
        name=f"1_{name}_triang_init",
        saveto=saveto,
        v_colors=triang_vertex_class,
    )

    print("\n ------------> Projecting triangulation vertices on REFINED depth")

    for regionidx in range(2, n_patches):
        region_type = dict_trapregion_to_type[regionidx]
        region_params = dict_patch_to_params[regionidx]
        this_patch_triang_internal_idx = dict_region_to_internal_triangulation_vertices_idx[regionidx]
        this_patch_triang_junction_idx = dict_region_to_junction_triangulation_vertices_idx[regionidx]
        triang_patchx = triang_x[this_patch_triang_internal_idx]
        triang_patchy = triang_y[this_patch_triang_internal_idx]
        triang_patchz = triang_z[this_patch_triang_internal_idx]
        triang_jx = triang_x[this_patch_triang_junction_idx]
        triang_jy = triang_y[this_patch_triang_junction_idx]
        triang_jz = triang_z[this_patch_triang_junction_idx]
        if region_type == "Plane":
            triang_refined_z[this_patch_triang_internal_idx] = plane_evalZ(triang_patchx, triang_patchy, *region_params)
            triang_refined_z[this_patch_triang_junction_idx] = 0.5 * triang_refined_z[this_patch_triang_junction_idx] +\
                                                               0.5 * plane_evalZ(triang_jx, triang_jy, *region_params)
            for vi in range(len(this_patch_triang_junction_idx)):
                v = this_patch_triang_junction_idx[vi]
                if len(dict_junction_to_regions[v]) == 1:
                    triang_refined_z[v] = plane_evalZ(triang_jx[vi], triang_jy[vi], *region_params)
        if region_type == "Cylinder":
            c = np.array(region_params[:3])
            w = np.array(region_params[3:6])
            r2 = cylinder_estimate_r(
                c=c,
                w=w,
                x=triang_patchx,
                y=triang_patchy,
                z=triang_patchz,
            ) ** 2
            triang_refined_z[this_patch_triang_internal_idx] = cylinder_evalZ(
                triang_patchx, triang_patchy, triang_patchz,
                c=c, w=w, r2=r2, debug=False,
            )
            triang_refined_z[this_patch_triang_junction_idx] = cylinder_evalZ(
                triang_jx, triang_jy, triang_jz,
                c=c, w=w, r2=r2, debug=False,
            )
        # if region_type == "Cone":
        #     coneV = np.array(region_params[:3])
        #     coneU = np.array(region_params[3:6])
        #     coneTheta = np.array(region_params[6])
        #     triang_refined_z[this_patch_triang_internal_idx] = cone_evalZ(
        #         triang_patchx, triang_patchy, triang_patchz,
        #         v=coneV, u=coneU, theta=coneTheta, debug=False,
        #     )
        #     triang_refined_z[this_patch_triang_junction_idx] = cone_evalZ(
        #         triang_jx, triang_jy, triang_jz,
        #         v=coneV, u=coneU, theta=coneTheta, debug=False,
        #     )
        if region_type == "Sphere":
            sphereC = np.array(region_params[:3])
            sphereR2 = region_params[3]
            triang_refined_z[this_patch_triang_internal_idx] = sphere_evalZ(
                triang_patchx, triang_patchy, triang_patchz,
                c=sphereC, r2=sphereR2,
            )
            triang_refined_z[this_patch_triang_junction_idx] = sphere_evalZ(
                triang_jx, triang_jy, triang_jz,
                c=sphereC, r2=sphereR2,
            )

    log3d_mesh(
        v_x=triang_x,
        v_y=triang_y,
        v_z=triang_refined_z,
        faces=np.copy(triangulation.faces),
        name=f"2_{name}_triang_refined",
        saveto=saveto,
        v_colors=triang_vertex_class,
    )

    if inflate:
        inflate_2d_mesh(
            vertices2d=np.vstack((triang_x, triang_y)).transpose(),
            vertex2d_markers=triangulation.vertex_markers,
            faces2d=np.copy(triangulation.faces),
            holes=triangulation.holes,
            vertex_z_values=triang_refined_z,
            vertex_classes=triang_vertex_class,
            name=f"{name}_triang_refined_inflated",
            saveto=saveto,
        )

    with open(saveto / "global_opt_logs.pkl", 'wb') as f:
        pickle.dump(
            [
                n_patches,
                dict_patch_to_params,
                triang_x,
                triang_y,
                triang_z,
                triang_refined_z,
                junction_vertices,
                dict_trapregion_to_type,
                dict_patch_to_x,
                dict_patch_to_y,
                dict_patch_to_z,
                dict_region_to_junction_triangulation_vertices_idx,
                dict_region_to_internal_triangulation_vertices_idx,
                flat_adjacency_junctions,
                flat_laplacian_junctions,
                lists_of_duplicates,
            ],
            f,
            protocol=-1,
        )

    improved_junctions_x, improved_junctions_y, improved_junctions_z, improved_patch_params = \
        patches_global_optimization(
            n_patches=n_patches,
            dict_patch_to_params=dict_patch_to_params,
            triang_x=triang_x,
            triang_y=triang_y,
            triang_z=triang_z,
            triang_refined_z=triang_refined_z,
            pure_junction_vertices=junction_vertices,
            junction_laplacian=flat_adjacency_junctions,
            # junction_laplacian=flat_laplacian_junctions,
            dict_trapregion_to_type=dict_trapregion_to_type,
            dict_patch_to_x=dict_patch_to_x,
            dict_patch_to_y=dict_patch_to_y,
            dict_patch_to_z=dict_patch_to_z,
            dict_region_to_junction_triangulation_vertices_idx=dict_region_to_junction_triangulation_vertices_idx,
            dict_region_to_internal_triangulation_vertices_idx=dict_region_to_internal_triangulation_vertices_idx,
            pairs_of_duplicates=lists_of_duplicates,
            maxiterations=maxiterations,
        )

    with open(saveto / "improved_logs.pkl", 'wb') as f:
        pickle.dump(
            [
                improved_junctions_x,
                improved_junctions_y,
                improved_junctions_z,
                improved_patch_params,
            ],
            f,
            protocol=-1,
        )

    print(f"\n\n==================\n ----> Points projection <----")
    # result.x = x0

    improved_triang_x = np.copy(triang_x)
    improved_triang_y = np.copy(triang_y)
    improved_triang_z = np.copy(triang_refined_z)
    improved_pixelsZ = np.copy(refined_pixelsZ)
    improved_triang_x[junction_vertices] = improved_junctions_x
    improved_triang_y[junction_vertices] = improved_junctions_y
    improved_triang_z[junction_vertices] = improved_junctions_z
    improved_depth = -np.copy(refined_pixelsZ)

    patch_params_to_save = [[] for _ in range(n_patches)]
    # fill depth values inside patches

    for i_patch in range(2, n_patches):
        len_patch_params = len(dict_patch_to_params[i_patch])
        patch_params = improved_patch_params[i_patch]
        patch_params_to_save[i_patch] = patch_params.tolist()
        print(f"patch {i_patch} - {dict_trapregion_to_type[i_patch]}, params: ", patch_params)
        # print(f"Patch {i_patch} type {dict_trapregion_to_type[i_patch]}, params: ", patch_params)
        this_patch_triang_internal_idx = dict_region_to_internal_triangulation_vertices_idx[i_patch]
        this_patch_triang_junction_idx = dict_region_to_junction_triangulation_vertices_idx[i_patch]
        triang_patchx = triang_x[this_patch_triang_internal_idx]
        triang_patchy = triang_y[this_patch_triang_internal_idx]
        triang_patchz = triang_z[this_patch_triang_internal_idx]
        patchx, patchy = dict_patch_to_x[i_patch], dict_patch_to_y[i_patch]
        patchz = dict_patch_to_z[i_patch]
        nearby_junctions_mask = dict_patch_to_image_junction_mask[i_patch]
        patch_junction_x = pixelsX[nearby_junctions_mask]
        patch_junction_y = pixelsY[nearby_junctions_mask]
        patch_junction_z = pixelsZ[nearby_junctions_mask]
        triang_junctions_x = improved_junctions_x[this_patch_triang_junction_idx]
        triang_junctions_y = improved_junctions_y[this_patch_triang_junction_idx]
        triang_junctions_z = improved_junctions_z[this_patch_triang_junction_idx]
        improvedz_internal = np.copy(refined_pixelsZ[fillmap == i_patch])
        improvedz_junct = np.copy(refined_pixelsZ[nearby_junctions_mask])
        improved_internal_triangle = np.copy(triang_patchz)
        if dict_trapregion_to_type[i_patch] == "Plane":
            improvedz_internal = plane_evalZ(patchx, patchy, *patch_params)
            improvedz_junct = plane_evalZ(patch_junction_x, patch_junction_y, *patch_params)
            improved_internal_triangle = plane_evalZ(triang_patchx, triang_patchy, *patch_params)
        if dict_trapregion_to_type[i_patch] == "Cylinder":
            c = np.array([patch_params[0], patch_params[1], patch_params[2]])
            w = np.array([patch_params[3], patch_params[4], patch_params[5]])
            # we need to re-estimate r2 as we do not optimize for it in our problem
            r2 = cylinder_estimate_r(
                c=c,
                w=w,
                x=triang_junctions_x,
                y=triang_junctions_y,
                z=triang_junctions_z,
            ) ** 2
            print("estimated r2 = ", r2)
            patch_params_to_save[i_patch][-1] = r2
            improvedz_internal = cylinder_evalZ(patchx, patchy, patchz, c=c, w=w, r2=r2, debug=True)
            improvedz_junct = cylinder_evalZ(patch_junction_x, patch_junction_y, patch_junction_z, c=c, w=w, r2=r2,
                                             debug=True)
            # improved_internal_triangle = cylinder_evalZ(triang_patchx, triang_patchy, triang_patchz, c=c, w=w, r2=r2,
            #                                             debug=True)
            # new_triang_internal_x, new_triang_internal_y, new_triang_internal_z = cylinder_find_closest(
            new_triang_internal_x, new_triang_internal_y, new_triang_internal_z = cylinder_projectZ_or_closest(
                triang_patchx, triang_patchy, triang_patchz,
                c=c, w=w, r2=r2,
                debug=True,
            )
            improved_triang_x[this_patch_triang_internal_idx] = new_triang_internal_x
            improved_triang_y[this_patch_triang_internal_idx] = new_triang_internal_y
            improved_triang_z[this_patch_triang_internal_idx] = new_triang_internal_z
            improved_internal_triangle = new_triang_internal_z

        # if dict_trapregion_to_type[i_patch] == "Cone":
        #     v = np.array([patch_params[0], patch_params[1], patch_params[2]])
        #     u = np.array([patch_params[3], patch_params[4], patch_params[5]])
        #     theta = patch_params[6]
        #     improvedz_internal = cone_evalZ(patchx, patchy, patchz, v=v, u=u, theta=theta, debug=False)
        #     improvedz_junct = cone_evalZ(patch_junction_x, patch_junction_y, patch_junction_z, v=v, u=u, theta=theta,
        #                                  debug=False)
        #     improved_internal_triangle = cone_evalZ(triang_patchx, triang_patchy, triang_patchz, v=v, u=u, theta=theta,
        #                                             debug=False)
        if dict_trapregion_to_type[i_patch] == "Sphere":
            c = np.array([patch_params[0], patch_params[1], patch_params[2]])
            r2 = patch_params[3]
            improvedz_internal = sphere_evalZ(patchx, patchy, patchz, c=c, r2=r2)
            improvedz_junct = sphere_evalZ(patch_junction_x, patch_junction_y, patch_junction_z, c=c, r2=r2)
            new_triang_internal_x, new_triang_internal_y, new_triang_internal_z = sphere_projectZ_or_closest(
                triang_patchx,
                triang_patchy,
                triang_patchz,
                c=c,
                r2=r2,
            )
            improved_triang_x[this_patch_triang_internal_idx] = new_triang_internal_x
            improved_triang_y[this_patch_triang_internal_idx] = new_triang_internal_y
            improved_triang_z[this_patch_triang_internal_idx] = new_triang_internal_z
            improved_internal_triangle = new_triang_internal_z
        improved_pixelsZ[fillmap == i_patch] = improvedz_internal
        improved_pixelsZ[nearby_junctions_mask] = improvedz_junct
        improved_depth[fillmap == i_patch] = -improvedz_internal
        improved_depth[nearby_junctions_mask] = -improvedz_junct
        improved_triang_z[[this_patch_triang_internal_idx]] = improved_internal_triangle

    # fill depth values on junction points
    # improved_depth[junction_points_mask] = - result.x[n_patches_params:]

    log3d_mesh(
        v_x=improved_triang_x,
        v_y=improved_triang_y,
        v_z=improved_triang_z,
        faces=np.copy(triangulation.faces),
        name=f"3_{name}_triang_improved",
        saveto=saveto,
        v_colors=triang_vertex_class,
    )

    final_params_array = np.zeros(shape=(n_patches, 7))
    for i_patch in range(2, n_patches):
        pparams = patch_params_to_save[i_patch]
        final_params_array[i_patch, :len(pparams)] = np.array(pparams)

    np.savez_compressed(
        file=saveto / f"data_{name}_improved_params.npz",
        patches=np.array([v for key, v in dict_trapregion_to_type.items()]),
        params=final_params_array,
    )
    np.savez_compressed(
        file=saveto / f"data_{name}_triang_internal.npz",
        **{
            f"internal_{k}": v
            for k, v in dict_region_to_internal_triangulation_vertices_idx.items()
        }
    )
    np.savez_compressed(
        file=saveto / f"data_{name}_triang_junction.npz",
        **{
            f"junction_{k}": v
            for k, v in dict_region_to_junction_triangulation_vertices_idx.items()
        }
    )

    do_smoothing_iterations(
        pngname=name,
        n_iterations=10,
    )

    if inflate:
        inflate_2d_mesh(
            vertices2d=np.vstack((improved_triang_x, improved_triang_y)).transpose(),
            vertex2d_markers=triangulation.vertex_markers,
            faces2d=triangulation.faces,
            holes=triangulation.holes,
            vertex_z_values=improved_triang_z,
            vertex_classes=triang_vertex_class,
            name=f"{name}_triang_improved_inflated",
        )

    if plot:
        plt.figure()
        plt.imshow(improved_depth, interpolation="nearest")
        plt.title("improved_depth depth")
        plt.colorbar()
        plt.savefig(f"data/{name}_improved_depth.png", bbox_inches="tight", dpi=150)
        plt.close()

    if save3d:
        # to highlight junction points
        # refined_segm[fillmap == 0] = 9
        log3d(
            background_mask=segmentation == 0,
            depth_camera=improved_depth,
            segmentation=segmentation,
            name=f"3_{name}_pixel_improved",
            saveto=saveto,
        )

    if savenpz:
        np.savez_compressed(
            file=f"results/{name}_predicted.npz",
            sketch=image,
            depth=predicted_depth,
            classes=segmentation,
        )
        np.savez_compressed(
            file=f"results/{name}_improved.npz",
            sketch=image,
            depth=improved_depth,
            classes=segmentation,
        )
        np.savez_compressed(
            file=f"results/{name}_refined.npz",
            sketch=image,
            depth=-refined_pixelsZ,
            classes=segmentation,
        )


def main(
        pngname="Bearing_Like_Parts_008_1",
        n_iterations=10,
):
    workfolder = Path(f"results/{pngname}/")
    pred_depth_data = np.load(str(workfolder / "npz" / f"{pngname}_depth.npz"))
    pred_depth = pred_depth_data["depth"]
    # pred_segmentation_data = np.load(str(workfolder / "npz" / f"{pngname}_segm.npz"))
    pred_segmentation_data = np.load(str(workfolder / "npz" / f"{pngname}_refined_segm.npz"))
    # pred_segmentation_data = np.load(str(workfolder / f"{pngname}_manual_segm.npz"))
    pred_segmentation = pred_segmentation_data["classes"]
    pred_normals_data = np.load(str(workfolder / "npz" / f"{pngname}_norms.npz"))
    pred_normals = pred_normals_data["normals"]

    gtimage = pred_depth_data["sketch"]
    # gtimage = skimage.transform.rescale(gtimage, scale=0.5, order=3)
    print("gtimage.shape: ", gtimage.shape)
    # pred_depth = skimage.transform.rescale(pred_depth, scale=0.5, order=0)
    print("pred_depth.shape: ", pred_depth.shape)
    # pred_normals = skimage.transform.rescale(pred_normals, scale=0.5, order=0, channel_axis=2)
    print("pred_normals.shape: ", pred_normals.shape)
    # pred_segmentation = pred_segmentation[::2, ::2]
    print("pred_segmentation.shape: ", pred_segmentation.shape)
    # depth = depth[::2, ::2]

    log3d(
        background_mask=pred_segmentation == 0,
        depth_camera=pred_depth,
        segmentation=pred_segmentation,
        normals_camera=transform_normals_global_to_camera(global_normals=pred_normals),
        name=f"{pngname}/0_{pngname}_pixel_predicted"
    )

    # svg triangulation
    svgtriang, svg_points, svg_edges, svg_paths_edges = myTraingulation.from_svg(
        path_to_svg=f"results/{pngname}/{pngname}_clean.svg",
        svg_sampling_distance=10,
        triang_flags='YYqpa50',
        # svg_sampling_distance=50,
        # triang_flags='YYqpa1500',
    )

    # svgtriang.reshuffle_triangulation_vertices()
    svgtriang.plot(faces=True, show=False, saveto=Path(f"results/{pngname}/"), )

    # svgtriang.vertices += 8  # pad with 8
    # svgtriang.vertices /= 2  # shrink in size, as we did with other images
    log3d_svg(
        svgtri=svgtriang,
        depth_camera=pred_depth,
        segmentation=pred_segmentation,
        name=f"0_{pngname}_triang_predicted",
        saveto=Path(f"results/{pngname}/"),
    )

    refined_segm, _, _, _ = refine_segmentation(
        image=gtimage,
        raw_segmentation=pred_segmentation,
        plot=True,
        name=pngname,
        saveto=Path(f"results/{pngname}/"),
    )

    fillmap, refined_pixelsZ, dict_trapregion_to_type, dict_patch_to_params = refine(
        image=gtimage,
        segmentation=pred_segmentation,
        depth=pred_depth,
        camera_normals=transform_normals_global_to_camera(global_normals=pred_normals),
        plot=True,
        save3d=True,
        name=pngname,
        saveto=Path(f"results/{pngname}/"),
    )

    improve_me(
        image=gtimage,
        predicted_depth=pred_depth,
        segmentation=refined_segm,
        fillmap=fillmap,
        refined_pixelsZ=refined_pixelsZ,
        triangulation=svgtriang,
        svg_points=svg_points,
        svg_edges=svg_edges,
        svg_paths_edges=svg_paths_edges,
        dict_trapregion_to_type=dict_trapregion_to_type,
        dict_patch_to_params=dict_patch_to_params,
        name=pngname,
        saveto=Path(f"results/{pngname}/"),
        plot=True,
        save3d=True,
        savenpz=False,
        inflate=False,
        maxiterations=n_iterations,
    )


if __name__ == "__main__":
    # pngname = "6_freestyle_288_01"
    # pngname = "Bearing_Like_Parts_002_1"
    # pngname = "Nuts_013_1"
    # pngname = "Nuts_014_1"
    # pngname = "Cylindrical_Parts_011_1"
    # pngname = "Cylindrical_Parts_008_1"
    # pngname = "Round_Change_At_End_011_1"
    parser = argparse.ArgumentParser(description="Eval model")
    parser.add_argument(
        "--pngname",
        type=str,
        # default="Non-90_degree_elbows_003_1",
        # default="Posts_008_1",
        # default="Bearing_Blocks_001_1",
        # default="Round_Change_At_End_018_1",
        # default="assorted_Posts_008_1",
        # default="Nuts_013_1",
        # default="npr_1016_62.54_-148.24_1.4",
        # default="Pulley_Like_Parts_008_1",
        # default="edited_Flange_Like_Parts_003_1",
        # default="Long_Machine_Elements_015_1",5
        # default="Prismatic_Stock_003_1",
        default="Cylindrical_Parts_011_1",
        # default="Round_Change_At_End_011_1",
        # default="6_freestyle_288_01",
        # default="assorted_Posts_007_1",
        help="name of image",
    )
    parser.add_argument("--maxiter", type=int, default=50000, help="number of iterations")
    args = parser.parse_args()

    main(
        pngname=args.pngname,
        n_iterations=args.maxiter,
    )
