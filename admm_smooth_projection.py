import pickle
import warnings
import numpy as np
import igl
import scipy
from plane_fit import plane_evalZ
from test_cylinder_fit import cylinder_evalZ, cylinder_find_closest, cylinder_projectZ_or_closest, cylinder_find_closest_alongZ
from sphere_fit import sphere_evalZ


def solve_smoothness(
        LtL,
        init_z,
        keepz_idx,
        weight_close_to_init=0.01,
):
    """

    :param LtL: laplacian to use
    :param init_z: initial z values
    :param keepz_idx: ids where z should be preserved
    :param weight_close_to_init: weight on "how close to init positions"
    :return:
    """
    S = scipy.sparse.csr_matrix(
        (
            np.ones_like(keepz_idx).astype(float),
            (keepz_idx, keepz_idx)
        ), shape=(LtL.shape[0], LtL.shape[1]))
    w_keep = 100
    A = LtL + weight_close_to_init * scipy.sparse.eye(m=LtL.shape[0]) + w_keep * S
    B = weight_close_to_init * init_z + w_keep * S * init_z
    z = scipy.sparse.linalg.spsolve(A, B)
    return z


def project_points_on_patch(
        init_verts: np.array,
        idx_points_to_project: np.array,
        this_patch_type: str,
        this_patch_params: np.array,
        keepxy=False,
):
    patch_verts = np.copy(init_verts[idx_points_to_project, :])
    if this_patch_type == "Plane":
        patch_verts[:, 2] = plane_evalZ(patch_verts[:, 0], patch_verts[:, 1], *this_patch_params[:4])
        return patch_verts
    if this_patch_type == "Cylinder":
        # patch_verts[:, 2] = cylinder_evalZ(
        #     x=patch_verts[:, 0], y=patch_verts[:, 1], z=patch_verts[:, 2],
        #     c=this_patch_params[:3], w=this_patch_params[3:6], r2=this_patch_params[6], debug=True,
        # )
        new_z = cylinder_find_closest_alongZ(
            x=patch_verts[:, 0], y=patch_verts[:, 1], z=patch_verts[:, 2],
            c=this_patch_params[:3], w=this_patch_params[3:6], r2=this_patch_params[6],
            debug=False,
        )
        patch_verts[:, 2] = new_z
        return patch_verts
    warnings.warn(f"Not implemented projection for patch {this_patch_type}")
    return patch_verts


def get_patch(
        allV,
        allF,
        patch_idx: int,
        patches_to_junctions: dict,
        patches_to_internals: dict,
):
    list_of_patch_vertex_idx = patches_to_junctions[patch_idx].tolist()
    list_of_patch_vertex_idx.extend(patches_to_internals[patch_idx].tolist())

    map_old_v_idx_to_new = {list_of_patch_vertex_idx[i]: i for i in range(len(list_of_patch_vertex_idx))}

    set_of_patch_vertex_idx = set(list_of_patch_vertex_idx)

    newvertices = allV[list_of_patch_vertex_idx, :]
    newfaces = list()
    for f in allF:
        if (f[0] in set_of_patch_vertex_idx) and (f[1] in set_of_patch_vertex_idx) and (f[2] in set_of_patch_vertex_idx):
            newfaces.append(
                [map_old_v_idx_to_new[f[0]], map_old_v_idx_to_new[f[1]], map_old_v_idx_to_new[f[2]], ]
            )
    newfaces = np.array(newfaces)

    return newvertices, newfaces, list_of_patch_vertex_idx


def make_smooth_projections(
        pngname: str,
):
    data_params = np.load(f"results/{pngname}/npz/edgeresult_{pngname}_improved_params.npz")
    patch_params_array = data_params["params"]
    patch_types_array = data_params["patches"]
    print(patch_params_array)
    n_patches = patch_params_array.shape[0]
    patch_to_type = {
        i: patch_types_array[i - 2]
        for i in range(2, n_patches)
    }

    with open(f"results/{pngname}/pkl/vertex_opt_logs.pkl", "rb") as f:
        triangulation, \
            input_triang_x, \
            input_triang_y, \
            input_triang_z, \
            optX, \
            optY, \
            optZ, \
            input_dict_region_to_junction_triangulation_vertices_idx, \
            input_dict_region_to_internal_triangulation_vertices_idx, = pickle.load(f)

    init_allV = np.stack(
        (
            input_triang_x,
            input_triang_y,
            input_triang_z,
        ), axis=1,
    )

    # igl.write_obj(filename=f"reports/test.obj", v=init_allV, f=triangulation.faces)

    init_allV[:len(optX), 0] = optX
    init_allV[:len(optX), 1] = optY
    init_allV[:len(optX), 2] = optZ

    better_z = np.copy(init_allV[:, 2])

    for selected_patch_id in range(2, n_patches):
        print(f"- ADMM patch {selected_patch_id}, {patch_to_type[selected_patch_id]}")
        if patch_to_type[selected_patch_id] not in ["Cylinder", "Plane", "Other"]:
            continue
        patchv, patchf, patch_v_idx = get_patch(
            allV=init_allV,
            allF=triangulation.faces,
            patch_idx=selected_patch_id,
            patches_to_internals=input_dict_region_to_internal_triangulation_vertices_idx,
            patches_to_junctions=input_dict_region_to_junction_triangulation_vertices_idx,
        )
        if len(patchf) == 0:
            print(f"Patch {selected_patch_id} has no faces, skip")
            continue
        # TODO: this_patch_fixed_boundaries can be massively improved.
        # TODO: 1) if a boundary was created after cut, it is free to move
        # TODO: 2) if a free boundary (contour) belongs to "Other" patch it is free to move
        n_fixed_points = len(input_dict_region_to_junction_triangulation_vertices_idx[selected_patch_id])
        this_patch_fixed_boundaries = np.arange(n_fixed_points)  # preserve Z on these points
        patch_v_internal_idx = patch_v_idx[n_fixed_points:]  # these ids will be modified in the original mesh
        # igl.write_obj(filename=f"reports/init_{selected_patch_id}.obj", v=patchv, f=patchf)

        n_junction_vertices = len(input_dict_region_to_junction_triangulation_vertices_idx[selected_patch_id])
        patch_params = patch_params_array[selected_patch_id]

        patch_v_flat = np.copy(patchv)
        patch_v_flat[:, 2] = 0

        patch_laplacian = igl.cotmatrix(patch_v_flat, patchf)
        patch_LtL = patch_laplacian.transpose().dot(patch_laplacian)

        if patch_to_type[selected_patch_id] == "Other":
            better_z[patch_v_idx] = solve_smoothness(
                LtL=patch_LtL,
                init_z=patchv[:, 2],
                weight_close_to_init=0.01,
                keepz_idx=this_patch_fixed_boundaries,
            )
            continue

        proj_v = project_points_on_patch(
            init_verts=patchv,
            idx_points_to_project=np.arange(len(patchv)),
            this_patch_type=patch_to_type[selected_patch_id],
            this_patch_params=patch_params,
            keepxy=True,
        )
        # igl.write_obj(filename=f"reports/proj_{selected_patch_id}.obj", v=proj_v, f=patchf)

        if patch_to_type[selected_patch_id] == "Plane":
            better_z[patch_v_idx] = proj_v[:, 2]
            continue

        for i_iter in range(10):
            proj_z = proj_v[:, 2]
            smooth_z = solve_smoothness(
                LtL=patch_LtL,
                init_z=proj_z,
                weight_close_to_init=0.01,
                keepz_idx=this_patch_fixed_boundaries,
            )
            proj_v[:, 2] = smooth_z
            # igl.write_obj(filename=f"reports/{i_iter}_1smooth_p{selected_patch_id}.obj", v=proj_v, f=patchf)
            proj_v = project_points_on_patch(
                init_verts=proj_v,
                idx_points_to_project=np.arange(len(proj_v)),
                this_patch_type=patch_to_type[selected_patch_id],
                this_patch_params=patch_params,
                keepxy=True,
            )
            diff = smooth_z - proj_v[:, 2]
            print(f">--- iter {i_iter} diff : {np.linalg.norm(diff)}")
            # igl.write_obj(filename=f"reports/{i_iter}_2proj_p{selected_patch_id}.obj", v=proj_v, f=patchf)

        better_z[patch_v_internal_idx] = proj_v[n_fixed_points:, 2]

    init_allV[:, 2] = better_z
    # igl.write_obj(filename=f"reports/admm.obj", v=init_allV, f=triangulation.faces)
    igl.write_obj(filename=f"results/{pngname}/admm.obj", v=init_allV, f=triangulation.faces)


if __name__ == "__main__":
    myname = "p5_tubes_view1"
    # myname = "assorted_Posts_008_1"
    make_smooth_projections(pngname=myname)
