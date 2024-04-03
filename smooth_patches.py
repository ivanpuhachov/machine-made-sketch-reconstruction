import warnings
import numpy as np
import igl
import scipy
import scipy.optimize as spo
import pathlib
from plane_fit import fit_plane_euclidean, plane_evalZ, plane_eval_dist, plane_eval_dist2, plane_eval_distZ2
from test_cylinder_fit import cylinder_projectZ_or_closest, cylinder_find_closest
from sphere_fit import sphere_find_closest


def split_var(var, n_v):
    xx = var[:n_v]
    yy = var[n_v:2*n_v]
    zz = var[2*n_v:]
    return xx, yy, zz


def objectiveE(
        var,
        init_x,
        init_y,
        init_z,
        LtL,
        junction_idx: np.array,
        boundary_idx: np.array,
        weight_disp=0.01,
        weight_junction=1000,
        weight_boundary=1000,
        debug=False
):
    """

    :param boundary_idx:
    :param var:
    :param init_x:
    :param init_y:
    :param init_z:
    :param LtL: L^T . L - Laplacian matrix with zeroed elements to prevent smoothing info leaking between patches
    :param n_j: n of junction vertices (first vertices)
    :param weight_disp:
    :param weight_junction:
    :param debug:
    :return:
    """
    n_v = LtL.shape[0]
    x, y, z = split_var(var, n_v=n_v)
    # smooth_term = np.einsum("i,ij,j->", x, LtL, x) + \
    #               np.einsum("i,ij,j->", y, LtL, y) +  \
    #               np.einsum("i,ij,j->", z, LtL, z)
    smooth_term = x.transpose().dot(LtL.dot(x)) + \
                  y.transpose().dot(LtL.dot(y)) + \
                  z.transpose().dot(LtL.dot(z))
    disp_term = np.sum((x - init_x) ** 2) + \
                np.sum((y - init_y) ** 2) + \
                np.sum((z - init_z) ** 2)
    if debug:
        y_displ = (y - init_y) ** 2
        print(f"max Y-displacement {np.max(y_displ)} at {np.argmax(y_displ)}")
    junction_penalty_term = np.sum((x[junction_idx] - init_x[junction_idx]) ** 2) + \
                            np.sum((y[junction_idx] - init_y[junction_idx]) ** 2) + \
                            np.sum((z[junction_idx] - init_z[junction_idx]) ** 2)
    boundary_penalty_term = np.sum((x[boundary_idx] - init_x[boundary_idx]) ** 2) + \
                            np.sum((y[boundary_idx] - init_y[boundary_idx]) ** 2) + \
                            np.sum((z[junction_idx] - init_z[junction_idx]) ** 2)
    value = smooth_term + \
            weight_disp * disp_term + \
            weight_junction * junction_penalty_term + \
            weight_boundary * boundary_penalty_term
    if debug:
        print("smooth_term: ", smooth_term)
        print("disp_term: ", disp_term)
        print("junction_penalty_term: ", junction_penalty_term)
        print("boundary_penalty_term: ", boundary_penalty_term)
        print("value: ", value)
        # print("junctions: ", junction_idx)
        # print("boundaries: ", boundary_idx)

    return value


def jac_objectiveE(
        var,
        init_x,
        init_y,
        init_z,
        LtL,
        junction_idx: np.array,
        boundary_idx: np.array,
        weight_disp=0.01,
        weight_junction=1000,
        weight_boundary=1000,
        debug=False
):
    n_v = LtL.shape[0]
    x, y, z = split_var(var, n_v=n_v)
    # LtL = L_zeroed.transpose().dot(L_zeroed)
    smooth_term = np.hstack((
        2 * LtL.dot(x),
        2 * LtL.dot(y),
        2 * LtL.dot(z)
    ))
    disp_term = np.hstack((
        2 * (x - init_x),
        2 * (y - init_y),
        2 * (z - init_z),
    ))
    junction_penalty_term = np.zeros_like(disp_term)
    junction_penalty_term[junction_idx] = disp_term[junction_idx]
    junction_penalty_term[n_v + junction_idx] = disp_term[n_v + junction_idx]
    junction_penalty_term[2 * n_v + junction_idx] = disp_term[2 * n_v + junction_idx]

    boundary_penalty_term = np.zeros_like(junction_penalty_term)
    boundary_penalty_term[boundary_idx] = disp_term[boundary_idx]
    boundary_penalty_term[n_v + boundary_idx] = disp_term[n_v + boundary_idx]
    boundary_penalty_term[2 * n_v + boundary_idx] = disp_term[2 * n_v + boundary_idx]

    grad = smooth_term + \
           weight_disp * disp_term + \
           weight_junction * junction_penalty_term + \
           weight_boundary * boundary_penalty_term
    return grad


def build_local_laplacians(
        allfaces: np.array,
        all_vertices_for_laplacian: np.array,
        dict_patch_to_interior_idx: dict,
):
    adjacency_list = igl.adjacency_list(allfaces)
    dict_of_laplacians = dict()
    for i_patch in dict_patch_to_interior_idx.keys():
        if i_patch < 2:
            continue
        if len(dict_patch_to_interior_idx) == 0:
            dict_of_laplacians[i_patch] = scipy.sparse.csr_matrix(all_vertices_for_laplacian.shape[0], all_vertices_for_laplacian.shape[0])
            continue
        set_patch_adjacent_vertices = set()
        for v in dict_patch_to_interior_idx[i_patch]:
            for newv in adjacency_list[v]:
                set_patch_adjacent_vertices.add(newv)
        patch_faces = list()
        for ff in allfaces:
            v_is_adjacent = True
            i = 0
            while v_is_adjacent and (i < 3):
                v_is_adjacent = ff[i] in set_patch_adjacent_vertices
                i += 1
            if v_is_adjacent:
                patch_faces.append(ff.tolist())
        patch_faces = np.array(patch_faces)
        if len(patch_faces) == 0:
            dict_of_laplacians[i_patch] = scipy.sparse.csr_matrix((all_vertices_for_laplacian.shape[0], all_vertices_for_laplacian.shape[0]))
            continue
        patchL = igl.cotmatrix(all_vertices_for_laplacian, patch_faces)
        dict_of_laplacians[i_patch] = patchL
    return dict_of_laplacians


def smooth_vertices(
        init_vertices,
        LtL,
        junction_idx: np.array,
        boundary_idx: np.array,
        do_grad_check=True,
):
    print("========== SMOOTH STEP ===========")
    n_vertices = LtL.shape[0]
    init_x = init_vertices[:, 0]
    init_y = init_vertices[:, 1]
    init_z = init_vertices[:, 2]
    assert LtL.shape[0] == init_vertices.shape[0]
    x0 = np.hstack(
        (
            init_x,
            init_y,
            init_z,
        ),
    )
    w_displacement = 0.1
    w_junctions = 10000
    w_boundary = 10000
    print("--- before opt")
    objectiveE(
        x0,
        debug=True,
        LtL=LtL,
        junction_idx=junction_idx,
        boundary_idx=boundary_idx,
        init_x=init_x,
        init_y=init_y,
        init_z=init_z,
        weight_disp=w_displacement,
        weight_junction=w_junctions,
        weight_boundary=w_boundary,
    )
    myobjective = lambda x: objectiveE(x, LtL=LtL, junction_idx=junction_idx,
                                       boundary_idx=boundary_idx,
                                       init_x=init_x, init_y=init_y, init_z=init_z,
                                       weight_disp=w_displacement,
                                       weight_junction=w_junctions,
                                       weight_boundary=w_boundary,
                                       )
    mygrad = lambda x: jac_objectiveE(x, LtL=LtL, junction_idx=junction_idx,
                                      boundary_idx=boundary_idx,
                                      init_x=init_x, init_y=init_y, init_z=init_z,
                                      weight_disp=w_displacement,
                                      weight_junction=w_junctions,
                                      weight_boundary=w_boundary,
                                      )
    if do_grad_check:
        grad_error = spo.check_grad(
            func=myobjective,
            grad=mygrad,
            x0=x0,
        )
        print("Check gradient diff: ", grad_error)
        if grad_error > 0.01:
            warnings.warn(f"Gradient error: {grad_error}")
    # assert grad_error < 0.01

    result = spo.minimize(
        myobjective,
        x0=x0,
        # method="L-BFGS-B",
        method="CG",
        jac=mygrad,
        options={
            "maxiter": 1000,
            "maxfun": 1000000,
            "disp": True,
        }
    )

    print("--- after opt")
    objectiveE(
        result.x,
        debug=True,
        LtL=LtL,
        junction_idx=junction_idx,
        boundary_idx=boundary_idx,
        init_x=init_x,
        init_y=init_y,
        init_z=init_z,
        weight_disp=w_displacement,
        weight_junction=w_junctions,
        weight_boundary=w_boundary,
    )

    resx, resy, resz = split_var(var=result.x, n_v=n_vertices)
    res_vertices = np.stack((resx, resy, resz), axis=1)
    print("========== SMOOTH END ===========")
    return res_vertices


def project_points(
        init_vertices,
        patch_params_array,
        patch_to_type,
        patch_to_internals_idx: dict,
):
    n_patches = patch_params_array.shape[0]
    projectedx = np.copy(init_vertices[:, 0])
    projectedy = np.copy(init_vertices[:, 1])
    projectedz = np.copy(init_vertices[:, 2])
    for i_patch in range(2, n_patches):
        array_params = patch_params_array[i_patch]
        internal_points_idx = patch_to_internals_idx[i_patch]
        triang_patchx = projectedx[internal_points_idx]
        triang_patchy = projectedy[internal_points_idx]
        triang_patchz = projectedz[internal_points_idx]
        if patch_to_type[i_patch] == "Plane":
            patch_params = array_params[:4].tolist()
            projectedz[internal_points_idx] = plane_evalZ(triang_patchx, triang_patchy, *patch_params)
        if patch_to_type[i_patch] == "Cylinder":
            patch_params = array_params[:7]
            c = np.array([patch_params[0], patch_params[1], patch_params[2]])
            w = np.array([patch_params[3], patch_params[4], patch_params[5]])
            r2 = patch_params[6]
            new_triang_internal_x, new_triang_internal_y, new_triang_internal_z = cylinder_find_closest(
                triang_patchx, triang_patchy, triang_patchz,
                c=c, w=w, r2=r2,
                debug=True,
            )
            projectedx[internal_points_idx] = new_triang_internal_x
            projectedy[internal_points_idx] = new_triang_internal_y
            projectedz[internal_points_idx] = new_triang_internal_z
    #     if dict_trapregion_to_type[i_patch] == "Cone":
    #         v = np.array([patch_params[0], patch_params[1], patch_params[2]])
    #         u = np.array([patch_params[3], patch_params[4], patch_params[5]])
    #         theta = patch_params[6]
    #         improvedz_internal = cone_evalZ(patchx, patchy, patchz, v=v, u=u, theta=theta, debug=False)
    #         improvedz_junct = cone_evalZ(patch_junction_x, patch_junction_y, patch_junction_z, v=v, u=u, theta=theta,
    #                                      debug=False)
    #         improved_internal_triangle = cone_evalZ(triang_patchx, triang_patchy, triang_patchz, v=v, u=u, theta=theta,
    #                                                 debug=False)
        if patch_to_type[i_patch] == "Sphere":
            patch_params = array_params[:4]
            c = np.array([patch_params[0], patch_params[1], patch_params[2]])
            r2 = patch_params[3]
            new_triang_internal_x, new_triang_internal_y, new_triang_internal_z = sphere_find_closest(
                x=triang_patchx,
                y=triang_patchy,
                z=triang_patchz,
                c=c,
                r2=r2,
            )
            projectedx[internal_points_idx] = new_triang_internal_x
            projectedy[internal_points_idx] = new_triang_internal_y
            projectedz[internal_points_idx] = new_triang_internal_z
    proj_vertices = np.stack((projectedx, projectedy, projectedz), axis=1)
    return proj_vertices


def do_smoothing_iterations(
        pngname="Pulley_Like_Parts_007_1",
        n_iterations=10,
):
    workdir = pathlib.Path(f"results/{pngname}/")
    flat_v, _ = igl.read_triangle_mesh(str(workdir / f"{pngname}_triang_cut_flat.obj"))
    init_v, _ = igl.read_triangle_mesh(str(workdir / f"1_{pngname}_triang_init.obj"))
    improved_v, f = igl.read_triangle_mesh(str(workdir / f"3_{pngname}_triang_improved.obj"))
    adjacency_list = igl.adjacency_list(f)
    adjacency_matrix = igl.adjacency_matrix(f)

    n_vertices = improved_v.shape[0]
    n_faces = f.shape[0]
    print(f"We have {n_vertices} vertices for {n_faces}")

    data_params = np.load(f"results/{pngname}/data_{pngname}_improved_params.npz")
    data_internals = np.load(f"results/{pngname}/data_{pngname}_triang_internal.npz")
    data_junctions = np.load(f"results/{pngname}/data_{pngname}_triang_junction.npz")

    patch_params_array = data_params["params"]
    patch_types_array = data_params["patches"]
    n_patches = patch_params_array.shape[0]

    patch_to_type = {
        i: patch_types_array[i-2]
        for i in range(2, n_patches)
    }

    patch_to_internals_idx = {
        i: data_internals[f"internal_{i}"]
        for i in range(2, n_patches)
    }

    patch_to_junctions_idx = {
        i: data_junctions[f"junction_{i}"]
        for i in range(2, n_patches)
    }

    make_smooth_surface(
        improved_v=improved_v,
        f=f,
        patch_to_type=patch_to_type,
        patch_to_junctions_idx=patch_to_junctions_idx,
        patch_to_internals_idx=patch_to_internals_idx,
        patch_params_array=patch_params_array,
        n_iterations=n_iterations,
        pngname=pngname,
    )


def make_smooth_surface(
        improved_v,
        f,
        patch_to_type,
        patch_to_junctions_idx,
        patch_to_internals_idx,
        patch_params_array,
        n_iterations,
        pngname="Pulley_Like_Parts_007_1",
):
    workdir = pathlib.Path(f"results/{pngname}/")
    n_patches = max(patch_to_type.keys()) + 1
    flat_v = np.copy(improved_v)
    flat_v[:, 2] = 1
    set_all_junctions = set()
    for k in range(2, n_patches):
        v = patch_to_junctions_idx[k]
        print(k, " ", patch_to_type[k], " junctions: ", v)
        for x in v:
            set_all_junctions.add(x)

    print("set_all_junctions: ", np.array(set_all_junctions))
    print("number of junctions = ", len(set_all_junctions))

    set_boundary_vertices = set()
    EV, FE, EF = igl.edge_topology(v=improved_v, f=f)
    for i in range(EF.shape[0]):
        if (EF[i][0] == -1) or (EF[i][1] == -1):
            set_boundary_vertices.add(EV[i][0])
            set_boundary_vertices.add(EV[i][1])
    print("all boundary vertices: ", np.array(set_boundary_vertices))
    print("number of boundaries: ", len(set_boundary_vertices))
    print("boundaries that are junctions too: ", set_boundary_vertices.intersection(set_all_junctions))
    set_boundary_vertices = set_boundary_vertices.intersection(set_all_junctions)
    set_all_junctions = set_all_junctions - set_boundary_vertices
    # print("boundaries after removing junctions: ", np.array(set_boundary_vertices))
    print("junctions after removing boundaries: ", set_all_junctions)

    print("Unknown junctions on Cone and others")
    set_unknown_junctions = set()
    for k in range(2, n_patches):
        v = patch_to_junctions_idx[k]
        if patch_to_type[k] not in ["Plane", "Cylinder", "Sphere"]:
            for x in v:
                set_unknown_junctions.add(x)
    print("Unknown junctions: ", set_unknown_junctions)

    L = igl.cotmatrix(flat_v, f)
    lhasnan = np.isnan(np.sum(L))
    print("L has NaN: ", lhasnan)
    if lhasnan:
        nz_rows, nz_cols = L.nonzero()
        for i in range(len(nz_rows)):
            if np.isnan(L[nz_rows[i], nz_cols[i]]):
                print(nz_rows[i], nz_cols[i])
        print("here")
        raise Exception("L has NaN")

    M = igl.massmatrix(flat_v, f)
    Minv = scipy.sparse.diags(1 / M.diagonal())

    # set_pure_junctions.discard(set_unknown_junctions)
    set_pure_junctions = set_all_junctions
    # set_pure_junctions = set_all_junctions - set_unknown_junctions
    set_boundary_vertices = set_boundary_vertices.union(set_unknown_junctions)
    L_zeroed = scipy.sparse.csr_matrix.copy(L)
    tempdiag = scipy.sparse.eye(L_zeroed.shape[0]).tolil()
    for x in set_all_junctions:
        tempdiag[x, x] = 0
    L_zeroed = tempdiag.dot(L_zeroed)
    # LtL = L.transpose().dot(L)
    LtL = L_zeroed.transpose().dot(L_zeroed)

    patch_laplacians = build_local_laplacians(
        allfaces=f,
        all_vertices_for_laplacian=flat_v,
        dict_patch_to_interior_idx=patch_to_internals_idx,
    )

    L_patchbased = scipy.sparse.csr_matrix((L.shape[0], L.shape[1]))
    LtL = scipy.sparse.csr_matrix((L.shape[0], L.shape[1]))
    for i_patch in range(2, n_patches):
        patch_l = patch_laplacians[i_patch]
        patch_ltl = patch_l.transpose().dot(patch_l)
        patch_l_zeroed = tempdiag.dot(patch_l)
        patch_ltl_zeroed = patch_l_zeroed.transpose().dot(patch_l_zeroed)
        w_this_patch = 1.0
        if patch_to_type[i_patch] in ["Sphere", "Cylinder", "Cone", "Plane"]:
            w_this_patch = 0.05
        LtL = LtL + w_this_patch * patch_ltl
        L_patchbased = L_patchbased + w_this_patch * patch_l

    # set_pure_junctions = set_pure_junctions - set_unknown_junctions
    # set_boundary_vertices = set_boundary_vertices.union(set_unknown_junctions)
    pure_junctions = np.array(sorted(list(set_pure_junctions)), dtype=int)
    n_junctions = len(pure_junctions)
    print("pure_junctions 1: ", pure_junctions)
    print("n_junctions = ", n_junctions)

    boundary_vertices = np.array(sorted(list(set_boundary_vertices)))
    print("boundary_vertices: ", boundary_vertices)

    print("boundary ")

    if n_iterations == -1:
        smooth_v = smooth_vertices(
            init_vertices=improved_v,
            # LtL=LtL,
            LtL=-L_patchbased,
            junction_idx=pure_junctions,
            boundary_idx=boundary_vertices,
            do_grad_check=True,
        )
        return smooth_v

    def iteration(
            start_vertices,
            grad_check=True,
    ):

        smooth_v = smooth_vertices(
            init_vertices=start_vertices,
            # LtL=LtL,
            LtL=-L_patchbased,
            junction_idx=pure_junctions,
            boundary_idx=boundary_vertices,
            do_grad_check=grad_check,
        )
        proj_v = project_points(
            init_vertices=smooth_v,
            patch_to_type=patch_to_type,
            patch_params_array=patch_params_array,
            patch_to_internals_idx=patch_to_internals_idx,
        )
        return smooth_v, proj_v

    sv0, pv0 = improved_v, improved_v

    for i in range(n_iterations):
        # pv0[boundary_vertices, 0] = improved_v[boundary_vertices, 0]
        # pv0[boundary_vertices, 1] = improved_v[boundary_vertices, 1]
        sv1, pv1 = iteration(
            start_vertices=pv0,
            grad_check=True if (i == 0) or (i == n_iterations-1) else False,
        )
        # triang_mesh3d = MyMesh3D(vertices=sv1, faces=f, vertex_markers=np.array([]),
        #                          holes=[])
        # triang_mesh3d.export_obj(file_path=workdir / f"newsmooth{i}.obj")
        # triang_mesh3d.vertices = pv1
        # triang_mesh3d.export_obj(file_path=workdir / f"newproj{i}.obj")
        # pv1[boundary_vertices, 0] = improved_v[boundary_vertices, 0]
        # pv1[boundary_vertices, 1] = improved_v[boundary_vertices, 1]

        sv0 = sv1
        pv0 = pv1
    return pv0


if __name__ == "__main__":
    # do_smoothing_iterations(pngname="Nuts_014_1")
    # do_smoothing_iterations(pngname="Nuts_013_1")
    # do_smoothing_iterations(pngname="Pulley_Like_Parts_007_1")
    # do_smoothing_iterations(pngname="Cylindrical_Parts_011_1", n_iterations=10)
    # do_smoothing_iterations(pngname="6_freestyle_288_01")
    do_smoothing_iterations(pngname="Round_Change_At_End_011_1")
    # do_smoothing_iterations(pngname="Bearing_Like_Parts_002_1")
    # do_smoothing_iterations(pngname="Bolt_Like_Parts_009_1")
    # do_smoothing_iterations(pngname="assorted_Posts_008_1", n_iterations=1)
    # do_smoothing_iterations()
