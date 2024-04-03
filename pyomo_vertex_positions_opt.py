import argparse
import pickle
import warnings
from pyomo_distances import evaluate_distance, get_patch_axis
from plane_fit import plane_eval_distZ2, plane_evalZ
from test_cylinder_fit import cylinder_evalZ, CylinderFit
import pyomo.environ as pyo
import numpy as np
from improve_me import log3d_mesh
from pathlib import Path
from smooth_patches import make_smooth_surface
from poisson_inflate import inflate_2d_mesh
import igl
from admm_smooth_projection import make_smooth_projections
import matplotlib
import matplotlib.pyplot as plt


def pyomo_opt_problem(
        triang_x,
        triang_y,
        triang_z,
        junctions_idx,
        patch_to_junctions: dict,
        junctions_to_patch: dict,
        patch_to_type,
        patch_params: np.array,
        paths_edges: list,
        weight_z_forshortening=0.1,
):
    model = pyo.ConcreteModel()
    model.set_of_junctions_idx = pyo.Set(initialize=junctions_idx)
    model.set_of_patches = pyo.Set(initialize=patch_to_type.keys())
    model.x = pyo.Var(
        model.set_of_junctions_idx,
        domain=pyo.Reals,
        # bounds=(-3, 3),
        initialize=triang_x[junctions_idx]
    )  # contains junctions x positions
    model.y = pyo.Var(
        model.set_of_junctions_idx,
        domain=pyo.Reals,
        # bounds=(-3, 3),
        initialize=triang_y[junctions_idx]
    )  # contains junctions y positions
    model.z = pyo.Var(
        model.set_of_junctions_idx,
        domain=pyo.Reals,
        # bounds=(-20, 20),
        initialize=triang_z[junctions_idx]
    )  # contains junctions z positions
    model.xx = pyo.Var(initialize=1.5)

    def rosenbrock(m):
        return (1.6 - m.xx) ** 2

    model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)

    for p in model.set_of_patches:
        for j in patch_to_junctions[p]:
            model.obj += evaluate_distance(
                x=model.x[j],
                y=model.y[j],
                z=model.z[j],
                params=patch_params[p,:],
                type=patch_to_type[p],
            )

    for j in junctions_idx:
        model.obj += 10 * ((model.x[j] - triang_x[j])**2 + (model.y[j] - triang_y[j])**2)

    # if weight_z_forshortening != 0:
    #     for p in paths_edges:
    #         for e in p:
    #         # for i_e in range(1, len(p)-1):
    #         #     e = p[i_e]
    #             model.obj += weight_z_forshortening * (model.z[e[0]] - model.z[e[1]]) ** 2

    solver = pyo.SolverFactory('ipopt')
    status = solver.solve(model, tee=True, report_timing=True)
    pyo.assert_optimal_termination(status)
    print(status)

    def var_to_np_array(vv, id_set=model.set_of_junctions_idx):
        l = list()
        for i in id_set:
            l.append(pyo.value(vv[i]))
        return np.array(l)

    opt_x = var_to_np_array(model.x, id_set=model.set_of_junctions_idx)
    opt_y = var_to_np_array(model.y, id_set=model.set_of_junctions_idx)
    opt_z = var_to_np_array(model.z, id_set=model.set_of_junctions_idx)

    def report_vertex(iv: int):
        print(f"\n====\nVertex {iv}")
        print(f"OLD xyz: {triang_x[iv]:.3f}, {triang_y[iv]:.3f}, {triang_z[iv]:.3f}")
        x, y, z = pyo.value(model.x[iv]), pyo.value(model.y[iv]), pyo.value(model.z[iv])
        print(f"NEW xyz: {x:.3f}, {y:.3f}, {z:.3f}")
        print(f"Connected Regions: ", junctions_to_patch[iv])
        for ii in junctions_to_patch[iv]:
            dist = evaluate_distance(x, y, z, params=patch_params[p],
                                     type=patch_to_type[p], )
            print(f"Distance to patch {ii}: {dist:.4f}")

    return opt_x, opt_y, opt_z


def vertexOpt(
        pngname: str,
):
    with open(f"results/{pngname}/pkl/trappedball_logs.pkl", "rb") as f:
        _, \
            _, \
            _, \
            fillmap, \
            thin_fillmap, = pickle.load(f)

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
            input_free_boundary_vertices, \
            input_dict_trapregion_to_type, \
            input_dict_region_to_junction_triangulation_vertices_idx, \
            input_dict_region_to_internal_triangulation_vertices_idx, = pickle.load(f)

    improved_triang_x = np.copy(input_triang_x)
    improved_triang_y = np.copy(input_triang_y)
    improved_triang_z = np.copy(edge_opt_z)

    input_n_patches = max(input_dict_trapregion_to_type) + 1
    region_params = np.zeros(shape=(input_n_patches, 7))
    for p in input_dict_patch_to_params.keys():
        for iv in range(len(input_dict_patch_to_params[p])):
            region_params[p, iv] = input_dict_patch_to_params[p][iv]

    list_of_paths_edges = svg_paths_edges
    for i_chain in range(len(list_of_chains)):
        dup1 = list_of_duplicates[2 * i_chain]
        dup2 = list_of_duplicates[2 * i_chain + 1]
        # here we check that dup1 contains freshly created vertices (that went "left")
        if min(dup1) < triangulation.n_svg_points:
            warnings.warn(f"chain {dup1} has original vertices!")
        left_path_edges = list()
        for i_v in range(len(dup1) - 1):
            left_path_edges.append([dup1[i_v], dup1[i_v + 1]])
            # v1, v2 = dup1[i_v], dup1[i_v + 1]
            # if (v1 > triangulation.n_svg_points) and (v2 > triangulation.n_svg_points):
            #     left_path_edges.append([v1, v2])
        list_of_paths_edges.append(left_path_edges)

    for i_patch in range(2, input_n_patches):
        this_patch_params = region_params[i_patch]
        print(f"patch {i_patch} - {input_dict_trapregion_to_type[i_patch]}, params: ", this_patch_params)
        this_patch_triang_internal_idx = input_dict_region_to_internal_triangulation_vertices_idx[i_patch]
        this_patch_triang_junction_idx = input_dict_region_to_junction_triangulation_vertices_idx[i_patch]
        triang_patchx = improved_triang_x[this_patch_triang_internal_idx]
        triang_patchy = improved_triang_y[this_patch_triang_internal_idx]
        triang_patchz = improved_triang_z[this_patch_triang_internal_idx]
        if input_dict_trapregion_to_type[i_patch] == "Plane":
            improved_triang_z[this_patch_triang_internal_idx] = plane_evalZ(triang_patchx, triang_patchy,
                                                                            *this_patch_params[:4])
        if input_dict_trapregion_to_type[i_patch] == "Cylinder":
            improved_triang_z[this_patch_triang_internal_idx] = cylinder_evalZ(
                triang_patchx, triang_patchy, triang_patchz,
                c=this_patch_params[:3], w=this_patch_params[3:6], r2=this_patch_params[6], debug=False,
            )

    adjacency_list = igl.adjacency_list(f=triangulation.faces)
    set_junctions = set(input_pure_junction_vertices)
    for i_b in input_pure_junction_vertices:
        if triangulation.vertex_markers[i_b] == 1:
            if i_b not in input_free_boundary_vertices:
                # if vertex not on the original boundary, then it means it comes from a cut we made
                # move this cut vertex closer to its neighbors
                ring1 = set(adjacency_list[i_b])
                ring1_nojunctions = ring1 - set_junctions
                if len(ring1_nojunctions) > 0:
                    newz = 0
                    for x in ring1_nojunctions:
                        newz += improved_triang_z[x] / len(ring1_nojunctions)
                    improved_triang_z[i_b] = newz

    log3d_mesh(
        v_x=improved_triang_x,
        v_y=improved_triang_y,
        v_z=improved_triang_z,
        faces=np.copy(triangulation.faces),
        name=f"vertex_{pngname}_input",
        saveto=Path(f"results/{pngname}/"),
    )

    svg_edges = svg_paths_edges[0]
    for i in range(1, len(svg_paths_edges)):
        svg_edges.extend(svg_paths_edges[i])

    try:
        v_colors = triangulation.get_vertex_class_from_segmentation(segm=thin_fillmap)
    except:
        v_colors = 2*np.ones_like(np.ones_like(triangulation.vertices[:, 0], dtype=int))
    v_colors[input_pure_junction_vertices] = 0

    data_params = np.load(f"results/{pngname}/npz/edgeresult_{pngname}_improved_params.npz")
    patch_params_array = data_params["params"]

    map_junction_to_patches = {x: set() for x in input_pure_junction_vertices}
    for p in range(2, input_n_patches):
        for x in input_dict_region_to_junction_triangulation_vertices_idx[p]:
            map_junction_to_patches[x].add(p)

    optX, optY, optZ = pyomo_opt_problem(
        triang_x=input_triang_x,
        triang_y=input_triang_y,
        triang_z=improved_triang_z[input_pure_junction_vertices],
        junctions_idx=input_pure_junction_vertices,
        patch_to_junctions=input_dict_region_to_junction_triangulation_vertices_idx,
        junctions_to_patch=map_junction_to_patches,
        patch_to_type=input_dict_trapregion_to_type,
        patch_params=region_params,
        paths_edges=list_of_paths_edges,
        weight_z_forshortening=0,
    )

    cmap = matplotlib.cm.get_cmap('tab20')
    np.random.seed(42)
    plt.figure()
    minz, maxz = np.min(optZ), np.max(optZ)
    minw, maxw = 1.5, 5
    scaler = lambda x: (x - minz) / (maxz - minz)
    getwidth = lambda x: minw + (maxw - minw) * scaler(x)
    cmap = matplotlib.cm.get_cmap('binary', )
    for i_e in range(len(svg_paths_edges)):
        e_l = svg_paths_edges[i_e]
        edges = np.array(e_l)
        # thiscolor = cmap(np.random.randint(20) / 20)
        for segm in e_l:
            lw = getwidth(optZ[segm[0]])
            plt.plot([optX[segm[0]], optX[segm[1]]],
                     [optY[segm[0]], optY[segm[1]]],
                     color=cmap(min(0.5 + 1.0 * scaler(optZ[segm[0]]), 1.0)),
                     linewidth=lw,
                     zorder=4,
                     alpha=1,
                     )
    plt.axis("equal")
    plt.axis("off")
    # plt.legend()
    origpath = Path(f"results/{pngname}") / "reports/"
    plt.savefig(origpath / "opt_edges.svg")
    plt.close()

    improved_triang_x[input_pure_junction_vertices] = optX
    improved_triang_y[input_pure_junction_vertices] = optY
    improved_triang_z[input_pure_junction_vertices] = optZ

    with open(Path(f"results/{pngname}/") / "pkl/vertex_opt_logs.pkl", 'wb') as f:
        pickle.dump(
            [
                triangulation,
                input_triang_x,
                input_triang_y,
                input_triang_z,
                optX,
                optY,
                optZ,
                input_dict_region_to_junction_triangulation_vertices_idx,
                input_dict_region_to_internal_triangulation_vertices_idx,
            ],
            f,
            protocol=-1,
        )

    log3d_mesh(
        v_x=improved_triang_x,
        v_y=improved_triang_y,
        v_z=improved_triang_z,
        faces=np.copy(triangulation.faces),
        name=f"vertex_{pngname}_boundaries_done",
        saveto=Path(f"results/{pngname}/"),
    )

    improved_v = np.stack(
        (
            improved_triang_x,
            improved_triang_y,
            improved_triang_z,
        ), axis=1,
    )

    smooth_vertices = make_smooth_surface(
        improved_v=improved_v,
        f=triangulation.faces,
        patch_to_type=input_dict_trapregion_to_type,
        patch_to_junctions_idx=input_dict_region_to_junction_triangulation_vertices_idx,
        patch_to_internals_idx=input_dict_region_to_internal_triangulation_vertices_idx,
        patch_params_array=patch_params_array,
        n_iterations=10,
        pngname=pngname,
    )

    log3d_mesh(
        v_x=smooth_vertices[:, 0],
        v_y=smooth_vertices[:, 1],
        v_z=smooth_vertices[:, 2],
        faces=np.copy(triangulation.faces),
        name=f"smooth_{pngname}_triang_imp",
        saveto=Path(f"results/{pngname}/"),
        v_colors=v_colors,
    )

    try:
        make_smooth_projections(pngname=pngname)
    except Exception as e:
        warnings.warn(f"Error in admm for {pngname}")
        print("=========\nERROR in ADMM \n")
        print(e)

    for i_patch in range(2, input_n_patches):
        int_vertices = input_dict_region_to_internal_triangulation_vertices_idx[i_patch]
        if input_dict_trapregion_to_type[i_patch] not in ["Plane", "Cylinder"]:
            improved_v[int_vertices, :] = smooth_vertices[int_vertices, :]
        if input_dict_trapregion_to_type[i_patch] == "Cylinder":
            if len(int_vertices) > 200:
                improved_v[int_vertices, :] = smooth_vertices[int_vertices, :]

    final_faces = np.copy(triangulation.faces)

    print("---- CONNECT CUTS ----")
    stitch_depth_thr = 10
    stitching_faces = []
    for i_cut in range(len(list_of_chains)):
        chain_verts = list_of_chains[i_cut]
        cut1_verts = list_of_duplicates[2*i_cut]
        cut2_verts = list_of_duplicates[2*i_cut + 1]
        print(cut1_verts)
        print(cut2_verts)
        print()
        # stitch only short vertices
        newfaces = []
        if chain_verts[0] != cut2_verts[0]:
            a, c = cut2_verts[0], cut1_verts[0]
            if np.abs(improved_v[a, 2] - improved_v[c, 2]) < stitch_depth_thr:
                newfaces.append([chain_verts[0], cut2_verts[0], cut1_verts[0]])
        for i in range(len(cut1_verts) - 1):
            a, b = cut1_verts[i], cut1_verts[i+1]
            c, d = cut2_verts[i], cut2_verts[i+1]
            if np.abs(improved_v[a, 2] - improved_v[c, 2]) < stitch_depth_thr:
                newfaces.append([a, c, d])
                newfaces.append([a, d, b])
        if chain_verts[-1] != cut1_verts[-1]:
            a, c = cut2_verts[-1], cut1_verts[-1]
            if np.abs(improved_v[a, 2] - improved_v[c, 2]) < stitch_depth_thr:
                newfaces.append([chain_verts[-1], cut2_verts[-1], cut1_verts[-1]])
        stitching_faces.extend(newfaces)
        newfaces = np.array(newfaces)
        if len(newfaces) > 0:
            final_faces = np.vstack((final_faces, newfaces))

    log3d_mesh(
        v_x=improved_v[:, 0],
        v_y=improved_v[:, 1],
        v_z=improved_v[:, 2],
        faces=final_faces,
        name=f"final_{pngname}_triang_imp",
        saveto=Path(f"results/{pngname}/"),
        v_colors=v_colors,
    )

    if len(stitching_faces) > 0:
        stitches_max_vertex = np.max(stitching_faces) + 1
        log3d_mesh(
            v_x=improved_v[:stitches_max_vertex, 0],
            v_y=improved_v[:stitches_max_vertex, 1],
            v_z=improved_v[:stitches_max_vertex, 2],
            faces=stitching_faces,
            name=f"stitches_{pngname}",
            saveto=Path(f"results/{pngname}/"),
            v_colors=v_colors,
        )

    # inflate_2d_mesh(
    #     vertices2d=np.vstack((improved_v[:, 0], improved_v[:, 1])).transpose(),
    #     vertex2d_markers=triangulation.vertex_markers.astype(int),
    #     faces2d=np.copy(triangulation.faces),
    #     holes=triangulation.holes,
    #     vertex_z_values=improved_v[:, 2],
    #     vertex_classes=np.ones_like(improved_triang_x).astype(int),
    #     name=f"{pngname}_triang_inflated",
    #     saveto=Path(f"results/{pngname}/"),
    # )


if __name__ == "__main__":
    pngname = "bin115"
    parser = argparse.ArgumentParser(description="Eval model")
    parser.add_argument(
        "--pngname",
        type=str,
        default=pngname,
        help="name of image",
    )
    args = parser.parse_args()
    vertexOpt(
        pngname=args.pngname
    )
