import warnings

import numpy as np
from pathlib import Path
import igl
import pickle
from shutil import copy2
import argparse
from admm_smooth_projection import project_points_on_patch
from distutils.dir_util import copy_tree


def cut_mesh_by_points(
        oldV,
        oldF,
        list_of_vertices: list,
):
    """

    :param oldV: shape (N, 3)
    :param oldF:
    :param list_of_vertices:
    :return:
    """
    set_of_vertices = set(list_of_vertices)
    map_old_to_new_vertices = {list_of_vertices[i]: i for i in range(len(list_of_vertices))}
    newV = np.copy(oldV[list_of_vertices, :])
    newF = list()
    for f in oldF:
        if (f[0] in set_of_vertices) and (f[1] in set_of_vertices) and (f[2] in set_of_vertices):
            newF.append(
                [
                    map_old_to_new_vertices[f[0]],
                    map_old_to_new_vertices[f[1]],
                    map_old_to_new_vertices[f[2]],
                ]
            )
    newF = np.array(newF, dtype=int)
    return newV, newF


def cylinder_span(
        cylinder_points: np.array,
        cylc: np.array,
        cylw: np.array,
):
    """

    :param cylw: cylinder axis, np.array (3)
    :param cylc: cylinder center, np.array(3)
    :param cylinder_points: (N,3) array of points
    :return:
    """
    heights = []
    for i_p in range(len(cylinder_points)):
        p = cylinder_points[i_p]
        cyl_h = (p - cylc).dot(cylw) / np.linalg.norm(cylw)
        heights.append(cyl_h)
    heights = np.array(heights)
    medianh = np.median(heights)
    betterc = cylc + cylw * medianh / np.linalg.norm(cylw)
    better_h_range = np.max(heights - medianh)
    return betterc, better_h_range


def subdivide_patch(
        init_v,
        init_f,
        patch_to_type,
        patch_to_internals,
        patch_to_junctions,
        patch_to_params,
        selected_patch: int,
):
    vertex_to_patches = {x: set() for x in range(len(init_v))}
    for k, v in patch_to_internals.items():
        for i_v in v:
            vertex_to_patches[i_v].add(k)

    for k, v in patch_to_junctions.items():
        for i_v in v:
            vertex_to_patches[i_v].add(k)

    faces_id_to_remove = []
    faces_to_append = []
    vertices_to_append = []
    for i_f in range(len(init_f)):
        iv1, iv2, iv3 = init_f[i_f]
        r1, r2, r3 = vertex_to_patches[iv1], vertex_to_patches[iv2], vertex_to_patches[iv3]
        common_region_set = r1.intersection(r2).intersection(r3)
        if len(common_region_set) == 1:
            common_region = common_region_set.pop()
            if common_region == selected_patch:
                faces_id_to_remove.append(i_f)
                newvertex = (init_v[iv1] + init_v[iv2] + init_v[iv3]) / 3
                newvertex[2] = project_points_on_patch(
                    init_verts=newvertex.reshape(1,3),
                    idx_points_to_project=[0],
                    this_patch_type=patch_to_type[common_region],
                    this_patch_params=patch_to_params[common_region],
                )[0, 2]
                newvertex_id = len(init_v) + len(vertices_to_append)
                vertices_to_append.append(newvertex.tolist())
                faces_to_append.extend([
                    [iv1, iv2, newvertex_id],
                    [iv1, newvertex_id, iv3],
                    [newvertex_id, iv2, iv3],
                ])

    temp_v = np.vstack((init_v, vertices_to_append))
    temp_f = faces_to_append
    selected_v = patch_to_junctions[selected_patch].tolist()
    selected_v.extend(patch_to_internals[selected_patch].tolist())
    selected_v.extend(range(len(init_v), len(init_v) + len(vertices_to_append)))

    orig_vertices_set = set(patch_to_junctions[selected_patch].tolist())
    orig_vertices_set = orig_vertices_set.union(set(patch_to_internals[selected_patch].tolist()))
    EV, FE, EF = igl.edge_topology(v=temp_v, f=np.array(temp_f))
    faces_id_to_remove = []
    flipped_f = []
    for i_e in range(len(EV)):
        e = EV[i_e]
        iv1, iv2 = e[0], e[1]
        if1, if2 = EF[i_e][0], EF[i_e][1]
        if (iv1 in orig_vertices_set) and (iv2 in orig_vertices_set) and (if1 > -1) and (if2 > -1):
            fff1, fff2 = temp_f[if1], temp_f[if2]
            c = (set(fff1) - {iv1, iv2}).pop()
            d = (set(fff2) - {iv1, iv2}).pop()
            faces_id_to_remove.extend([if1, if2])
            flipped_f.append(
                [iv1, d, c]
            )
            flipped_f.append(
                [c, d, iv2]
            )

    for x in sorted(faces_id_to_remove, reverse=True):
        del temp_f[x]
    temp_f.extend(flipped_f)

    patchverts, patchfaces = cut_mesh_by_points(
        oldV=temp_v,
        oldF=temp_f,
        list_of_vertices=selected_v,
    )

    return patchverts, patchfaces


def write_subdivided_patches(
        verts,
        faces,
        patch_to_internal_vert: dict,
        patch_to_junction_vert: dict,
        patch_to_params: dict,
        patch_to_type: dict,
        saveto: Path,
):
    for p in patch_to_junction_vert.keys():
        if p < 2:
            continue
        if len(patch_to_internal_vert[p]) < 10 :
            continue
        if patch_to_type[p] in ["Cylinder", "Planes"]:
            pv, pf = subdivide_patch(
                init_v=verts,
                init_f=faces,
                patch_to_internals=patch_to_internal_vert,
                patch_to_junctions=patch_to_junction_vert,
                patch_to_type=patch_to_type,
                patch_to_params=patch_to_params,
                selected_patch=p,
            )
            igl.write_obj(str(saveto / f"subd_patch_{p}.obj"), pv, pf)


def write_npz_patchparams(
        dict_patch_to_params: dict,
        dict_patch_to_type: dict,
        file_to_write: Path,
):
    optParams = np.zeros((max(dict_patch_to_params.keys()) + 1, 8))
    for k, v in dict_patch_to_params.items():
        optParams[k, :len(v)] = v
        optParams[k, 7] = 1

    np.savez_compressed(
        file=file_to_write,
        patches=np.array([v for key, v in dict_patch_to_type.items()]),
        params=optParams,
    )


def write_logs_for_blender(
        pngname: str,
):
    with open(f"results/{pngname}/pkl/edge_opt_logs.pkl", "rb") as f:
        triangulation, \
            svg_points, \
            svg_paths_edges, \
            input_dict_patch_to_params, \
            input_triang_x, \
            input_triang_y, \
            input_triang_z, \
            edge_opt_z, \
            list_of_duplicates, \
            list_of_chains, \
            input_pure_junction_vertices, \
            svg_free_boundary_vertices, \
            input_dict_trapregion_to_type, \
            input_dict_region_to_junction_triangulation_vertices_idx, \
            input_dict_region_to_internal_triangulation_vertices_idx, = pickle.load(f)

    # update svg_path_edges with cutchains
    for i_c in range(len(list_of_chains)):
        cutchain_verts = list_of_chains[i_c]
        dup1, dup2 = list_of_duplicates[2*i_c], list_of_duplicates[2*i_c + 1]
        newpath = dup1
        if len(cutchain_verts) == len(dup1) + 1:
            newpath.append(cutchain_verts[-1])
        if len(cutchain_verts) == len(dup1) + 1:
            newpath.insert(0, cutchain_verts[0])
            newpath.append(cutchain_verts[-1])
        this_path_edges = [[newpath[i], newpath[i+1]] for i in range(len(newpath)-1)]
        svg_paths_edges.append(this_path_edges)

    with open(f"results/{pngname}/pkl/face_to_fillmap.pkl", "rb") as f:
        face_to_fillmap = pickle.load(f)
    face_to_fillmap = face_to_fillmap[0]

    final_verts, final_faces = igl.read_triangle_mesh(filename=f"results/{pngname}/final_{pngname}_triang_imp.obj")

    with open(Path(f"results/{pngname}/pkl/") / "vis_logs.pkl", 'wb') as f:
        pickle.dump(
            [
                svg_points,
                svg_paths_edges,
                input_dict_trapregion_to_type,
                face_to_fillmap,
                input_dict_region_to_junction_triangulation_vertices_idx,
                input_dict_region_to_internal_triangulation_vertices_idx,
            ],
            f,
            protocol=-1,
        )

    optParams = np.zeros((max(input_dict_patch_to_params.keys())+1, 8))
    for k, v in input_dict_patch_to_params.items():
        optParams[k, :len(v)] = v
        if input_dict_trapregion_to_type[k] == "Cylinder":
            patch_ids = input_dict_region_to_internal_triangulation_vertices_idx[k].tolist()
            patch_ids.extend(input_dict_region_to_junction_triangulation_vertices_idx[k].tolist())
            print(f"Cylinder {k} vertices:")
            print(patch_ids)
            bc, bd = v[:3], 1
            try:
                bc, bd = cylinder_span(
                    cylinder_points=final_verts[patch_ids, :],
                    cylc=v[:3],
                    cylw=v[3:6],
                )
            except Exception as e:
                warnings.warn(f"cylinder_span failed on {k}")
            optParams[k, :3] = bc
            optParams[k, 7] = bd

    np.savez_compressed(
        file=Path(f"results/{pngname}/npz/") / f"blender_{pngname}_patch_params.npz",
        patches=np.array([v for key, v in input_dict_trapregion_to_type.items()]),
        params=optParams,
    )

    write_patches_triang(
        verts=final_verts,
        faces=final_faces,
        patch_to_junction_vert=input_dict_region_to_junction_triangulation_vertices_idx,
        patch_to_internal_vert=input_dict_region_to_internal_triangulation_vertices_idx,
        saveto=Path(f"results/{pngname}/final_patches/"),
    )

    write_subdivided_patches(
        verts=final_verts,
        faces=final_faces,
        patch_to_junction_vert=input_dict_region_to_junction_triangulation_vertices_idx,
        patch_to_internal_vert=input_dict_region_to_internal_triangulation_vertices_idx,
        saveto=Path(f"results/{pngname}/final_patches/"),
        patch_to_params=input_dict_patch_to_params,
        patch_to_type=input_dict_trapregion_to_type,
    )


def make_a_report(
        pngname: str,
        target="results_collection",
):
    workfolder = Path(f"results/{pngname}/")
    targetfolder = Path(f"{target}")
    # copy2(workfolder / f"{pngname}.png", targetfolder / f"{pngname}.png")
    copy2(workfolder / f"{pngname}_resized512.png", targetfolder / f"{pngname}_resized512.png")
    # copy2(workfolder / f"npz/blender_{pngname}_patch_params.npz", targetfolder / f"{pngname}_patch_params.npz")
    # copy2(workfolder / f"npz/localfits.npz", targetfolder / f"{pngname}_local_patch_params.npz")
    # copy2(workfolder / f"final_{pngname}_triang_imp.ply", targetfolder / f"{pngname}_final.ply")
    copy2(workfolder / f"final_{pngname}_triang_imp.obj", targetfolder / f"{pngname}_result.obj")
    # try:
    #     copy2(workfolder / f"admm.obj", targetfolder / f"{pngname}_admm.obj")
    # except:
    #     warnings.warn("no admm found")
    # copy2(workfolder / f"pkl/vis_logs.pkl", targetfolder / f"{pngname}_vis_logs.pkl")
    # copy2(workfolder / f"reports/edgeOpt_cuts.svg", targetfolder / f"{pngname}_edgeOpt_cuts.svg")
    # copy2(workfolder / f"reports/edgeOpt_edges.svg", targetfolder / f"{pngname}_displaced_edges.svg")
    # copy2(workfolder / f"reports/segmentation_report.png", targetfolder / f"{pngname}_segm.png")
    # copy_tree(str(workfolder / f"final_patches/"), str(targetfolder / f"{pngname}_patches/"))
    # copy_tree(str(workfolder / f"predicted_patches/"), str(targetfolder / f"{pngname}_predicted_patches/"))
    try:
        copy2(workfolder / f"stitches_{pngname}.obj", targetfolder / f"{pngname}_patches/stitches.obj")
    except:
        warnings.warn("no stitches found")


def write_patches_triang(
        verts,
        faces,
        patch_to_internal_vert: dict,
        patch_to_junction_vert: dict,
        saveto: Path,
):
    for p in patch_to_junction_vert.keys():
        if p < 2:
            continue
        allpoints = patch_to_junction_vert[p].tolist()
        allpoints.extend(patch_to_internal_vert[p].tolist())
        if len(allpoints) == 0:
            warnings.warn(f"Patch {p} has 0 points, skip")
            continue
        patchverts, patchfaces = cut_mesh_by_points(oldV=verts, oldF=faces, list_of_vertices=allpoints)
        if (len(patchverts) > 0) and (len(patchfaces) > 0):
            igl.write_obj(str(saveto / f"patch_{p}.obj"), patchverts, patchfaces)
        else:
            warnings.warn(f"patch {p} has {len(patchverts)} vertices and {len(patchfaces)}, I cannot write it")


if __name__ == "__main__":
    myname = "Cylindrical_Parts_011_1"

    parser = argparse.ArgumentParser(description="Run vis preparations")
    parser.add_argument(
        "--pngname",
        type=str,
        default=myname,
        help="name of image",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="results_collection",
        help="where to put the results",
    )
    args = parser.parse_args()

    write_logs_for_blender(args.pngname)
    make_a_report(pngname=args.pngname, target=args.target)
