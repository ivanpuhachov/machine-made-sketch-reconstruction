import pathlib
import pickle
import pyomo.environ as pyo
import numpy as np
from plane_fit import plane_eval_distZ2, plane_evalZ
from test_cylinder_fit import cylinder_evalZ, CylinderFit
from improve_me import log3d_mesh
from pathlib import Path
from improve_global_optimization import find_aligned_ortho_pairs, patches_to_axis_dict
from collections import Counter
from scipy.sparse import csr_matrix
import matplotlib
import matplotlib.pyplot as plt
import warnings
from cutTriangulation import cutTriangulation, cutExistingTriangulation
from arap_triangle import deform_triangle
from camera_transformations import get_pixel_coordinates, coordinates_to_pixels
from myTriangulation import myTraingulation
from pyomo_distances import evaluate_distance, distance_between_patches, get_patch_axis
import igl
from vis_preparations import write_patches_triang, cut_mesh_by_points
from admm_smooth_projection import project_points_on_patch
import argparse
import json
import os


def edge_objective_function(
        set_of_patches,
        set_of_junctions,
        mbb,
        mp,
        mx,
        my,
        mz,
        junction_edges,
        init_x,
        init_y,
        init_z,
        junction_edge_to_patch: dict,
        patch_to_type: dict,
        patch_to_internal_idx: dict,
        boundary_vertices: np.array,
        junction_vertices: np.array,
        svg_vertex_to_patches: dict,
        aligned_pairs: tuple,
        orthogonal_pairs: tuple,
        edges_along_path: list,
        debug=False,
        weight_junction_alignment=1000,
        weight_boundary_alignment=1000,
        weight_binary=0.005,
        weight_junction_displacement=10,
        weight_path_smoothness=100,
        weight_path_foreshortening=1,
):
    energy_distance_at_edge = 0
    energy_weight_is_one = 0
    energy_patch_regularization = 0
    energy_internal_to_patch = 0
    energy_patch_alignment = 0
    energy_patch_orthogonality = 0
    energy_junction_to_init = 0
    energy_boundary_to_init = 0
    energy_junctions_mid_to_init = 0
    energy_boundary_to_patch = 0
    energy_path_smoothness = 0
    energy_path_foreshortening = 0

    n_junction_edges = len(junction_edge_to_patch.keys())
    for ie in range(n_junction_edges):
        v1, v2 = junction_edges[ie][0], junction_edges[ie][1]
        midx = 0.5 * mx[v1] + 0.5 * mx[v2]
        midy = 0.5 * my[v1] + 0.5 * my[v2]
        midz = 0.5 * mz[v1] + 0.5 * mz[v2]
        ra, rb = junction_edge_to_patch[ie]
        # if (patch_to_type[ra] != "Plane") or (patch_to_type[rb] != "Plane"):
        #     continue
        energy_distance_at_edge += mbb[ie] * distance_between_patches(
        # energy_distance_at_edge += distance_between_patches(
            type1=patch_to_type[ra],
            type2=patch_to_type[rb],
            params1=[mp[ra, 0], mp[ra, 1], mp[ra, 2], mp[ra, 3], mp[ra, 4], mp[ra, 5], mp[ra, 6],],
            params2=[mp[rb, 0], mp[rb, 1], mp[rb, 2], mp[rb, 3], mp[rb, 4], mp[rb, 5], mp[rb, 6],],
            x=midx,
            y=midy,
            z=midz,
            # x=midpoints_x[ie],
            # y=midpoints_y[ie],
            # z=midpoints_z[ie],
        )[0] * weight_junction_alignment / n_junction_edges

        init_midx = 0.5 * init_x[v1] + 0.5 * init_x[v2]
        init_midy = 0.5 * init_y[v1] + 0.5 * init_y[v2]
        # init_midz = 0.5 * init_z[v1] + 0.5 * init_z[v2]

        energy_junctions_mid_to_init += weight_junction_displacement * (
            (midx - init_midx) ** 2 + (midy - init_midy) ** 2
        ) / n_junction_edges

    for ie in range(n_junction_edges):
        energy_weight_is_one += weight_binary * (1 - mbb[ie]) / n_junction_edges

    for p in set_of_patches:
        if patch_to_type[p] == "Plane":
            energy_patch_regularization += (1 - mp[p, 0] ** 2 - mp[p, 1] ** 2 - mp[p, 2] ** 2) ** 2
        if patch_to_type[p] == "Cylinder":
            energy_patch_regularization += (1 - mp[p, 3] ** 2 - mp[p, 4] ** 2 - mp[p, 5] ** 2) ** 2

    n_all_junction_vertices = len(set_of_junctions)
    for v in junction_vertices:
        energy_junction_to_init += weight_junction_displacement * (
                    (mx[v] - init_x[v]) ** 2 + (my[v] - init_y[v]) ** 2) / n_all_junction_vertices

    n_all_boundary_vertices = len(boundary_vertices)
    for v in boundary_vertices:
        energy_boundary_to_init += 0 * (
                    (mx[v] - init_x[v]) ** 2 + (my[v] - init_y[v]) ** 2) / n_all_junction_vertices
        p = svg_vertex_to_patches[v][0]
        assert len(svg_vertex_to_patches[v]) == 1
        energy_boundary_to_patch += weight_boundary_alignment * evaluate_distance(
                x=mx[v], y=my[v], z=mz[v],
                params=[mp[p, 0], mp[p, 1], mp[p, 2], mp[p, 3], mp[p, 4], mp[p, 5], mp[p, 6], ],
                type=patch_to_type[p],
        ) / n_all_boundary_vertices

    for p in edges_along_path:
        # energy_path_smoothness += (mx[p[0][0]] - init_x[p[0][0]] - 1) ** 2
        for i_e in range(len(p)):
            v1, v2 = p[i_e][0], p[i_e][1]
            energy_path_smoothness += weight_path_smoothness * (
                    ((mx[v2] - mx[v1]) - (init_x[v2] - init_x[v1]))**2 +
                    ((my[v2] - my[v1]) - (init_y[v2] - init_y[v1]))**2
            ) / n_all_junction_vertices
            # energy_path_foreshortening += weight_path_foreshortening * (
            #     (mz[v2] - mz[v1])**2
            # ) / n_all_junction_vertices
        if len(p) >= 2:
            for i_e in range(1, len(p)):
                edgea, edgeb = p[i_e - 1], p[i_e]
                vert_1x, vert_x = edgea[0], edgea[1]
                vert_x1 = edgeb[1]
                energy_path_smoothness += weight_path_smoothness * (
                        (
                                (2 * (mx[vert_x] - init_x[vert_x]) - (mx[vert_x1] - init_x[vert_x1]) - (mx[vert_1x] - init_x[vert_1x]))
                        ) ** 2 + (
                                (2 * (my[vert_x] - init_y[vert_x]) - (my[vert_x1] - init_y[vert_x1]) - (my[vert_1x] - init_y[vert_1x]))
                        ) ** 2
                ) / n_all_junction_vertices
                # if (i_e > 1) and (i_e < len(p) - 1):
                #     # TODO: exclude stroke ends from foreshortening as they might be at cut
                #     energy_path_foreshortening += weight_path_foreshortening * (2 * mz[vert_x] - mz[vert_x1] - mz[vert_1x]) ** 2 / n_all_junction_vertices
                energy_path_foreshortening += weight_path_foreshortening * (2 * mz[vert_x] - mz[vert_x1] - mz[vert_1x]) ** 2 / n_all_junction_vertices

    expression = energy_distance_at_edge + energy_weight_is_one + energy_patch_regularization + \
                 energy_path_smoothness +\
                 energy_path_foreshortening +\
                 energy_boundary_to_patch +\
                 energy_junction_to_init #+ \
                 # energy_internal_to_patch + \
                 # energy_patch_alignment + energy_patch_orthogonality

    if debug:
        print("Energy term patch to patch distance at edges: ", energy_distance_at_edge)
        print("Energy term mww -> 1 : ", energy_weight_is_one)
        print("Energy term patch param regularization: ", energy_patch_regularization)
        print("Energy term internal points patch eval: ", energy_internal_to_patch)
        print("Energy term patch orthogonality: ", energy_patch_orthogonality)
        print("Energy term patch alignment: ", energy_patch_alignment)
        print("Energy term junction displacement: ", energy_junction_to_init)
        print("Energy term junction midpoint displacement: ", energy_junctions_mid_to_init)
        print("Energy term boundary displacement: ", energy_boundary_to_init)
        print("Energy term boundary to patch: ", energy_boundary_to_patch)
        print("Energy term path smoothing: ", energy_path_smoothness)
        print("Energy term path foreshortening: ", energy_path_foreshortening)
        print("TOTAL: ", expression)

    return expression


def pyomo_edge_optimization(
        svg_vertices_idx: np.array,
        joint_edges_to_patch: dict,
        junction_edges,
        n_patches: int,
        patch_to_init_params: dict,
        init_triang_x: np.array,
        init_triang_y: np.array,
        init_triang_z: np.array,
        patch_to_type: dict,
        patch_to_internal_idx: dict,
        boundary_vertices_idx: np.array,
        svg_vertex_to_patch: dict,
        aligned_pairs: tuple,
        orthogonal_pairs: tuple,
        edges_along_path: list,
        max_iterations: int,
        opt_weights: dict,
):
    model = pyo.ConcreteModel()
    model.set_of_junction_edges = pyo.Set(initialize=joint_edges_to_patch.keys())
    model.set_of_junctions = pyo.Set(initialize=svg_vertices_idx)
    model.set_of_patches = pyo.Set(initialize=range(2, n_patches))

    model.x = pyo.Var(
        model.set_of_junctions,
        domain=pyo.Reals,
        # bounds=(-3, 3),
        initialize=init_triang_x[svg_vertices_idx],
    )  # contains junctions x positions
    model.y = pyo.Var(
        model.set_of_junctions,
        domain=pyo.Reals,
        # bounds=(-3, 3),
        initialize=init_triang_y[svg_vertices_idx],
    )  # contains junctions y positions
    model.z = pyo.Var(
        model.set_of_junctions,
        domain=pyo.Reals,
        # bounds=(-20, 20),
        initialize=init_triang_z[svg_vertices_idx],
    )  # contains junctions z positions

    def params_set_rule(m):
        return ((k, v) for k in m.set_of_patches for v in range(7))

    model.set_of_params_id = pyo.Set(dimen=2, initialize=params_set_rule)
    model.p = pyo.Var(model.set_of_params_id, domain=pyo.Reals)
    for i in model.set_of_patches:
        for j in range(len(patch_to_init_params[i])):
            model.p[i, j] = patch_to_init_params[i][j]
    model.p.display()

    def getcos(m, a, b):
        a1, a2, a3 = get_patch_axis(mp=m.p, p_patch=a, p_type=patch_to_type[a])
        b1, b2, b3 = get_patch_axis(mp=m.p, p_patch=b, p_type=patch_to_type[b])
        dotprod = a1 * b1 + a2 * b2 + a3 * b3
        norma = pyo.sqrt(a1 * a1 + a2 * a2 + a3 * a3)
        normb = pyo.sqrt(b1 * b1 + b2 * b2 + b3 * b3)
        cosab = dotprod / (norma * normb)
        return cosab

    def axis_aligned_rule(m, a, b):
        cosab = getcos(m, a, b)
        return (1 - cosab ** 2) <= 0.04
    model.axisAlignedConstraint = pyo.Constraint(aligned_pairs, rule=axis_aligned_rule)

    def axis_orthogonal_rule(m, a, b):
        cosab = getcos(m, a, b)
        return (cosab ** 2) <= 0.04
    model.axisOrthogonalConstraint = pyo.Constraint(orthogonal_pairs, rule=axis_orthogonal_rule)

    # def softmax_distance_constraint(m, p, debug=False):
    #     n_internal_points = len(patch_to_internal_idx[p])
    #     if patch_to_type[p] != "Plane":
    #         return pyo.Constraint.Skip
    #     list_of_distances = list()
    #     for v in patch_to_internal_idx[p]:
    #         distance = evaluate_distance(
    #             x=triang_x[v], y=triang_y[v], z=triang_z[v],
    #             params=[m.p[p, 0], m.p[p, 1], m.p[p, 2], m.p[p, 3], ],
    #             type="Plane",
    #         )
    #         list_of_distances.append(distance)
    #     list_of_exp_distances = [pyo.exp(x) for x in list_of_distances]
    #     softmax_denominator = sum(list_of_exp_distances)
    #     softmax_values = [pyo.exp(x) * x / softmax_denominator for x in list_of_distances]
    #     return

    model.upperbound = pyo.Var(initialize=1.0)

    def distance_constraint_rule(m, p, i_v):
        if i_v < len(svg_vertices_idx):
            return pyo.Constraint.Skip
        if i_v not in patch_to_internal_idx[p]:
            return pyo.Constraint.Skip
        if patch_to_type[p] == "Plane":
            distance = evaluate_distance(
                x=init_triang_x[i_v], y=init_triang_y[i_v], z=init_triang_z[i_v],
                params=[m.p[p, 0], m.p[p, 1], m.p[p, 2], m.p[p, 3], ],
                type=patch_to_type[p],
            )
            return distance <= m.upperbound
        if patch_to_type[p] == "Cylinder":
            distance = evaluate_distance(
                x=init_triang_x[i_v], y=init_triang_y[i_v], z=init_triang_z[i_v],
                params=[m.p[p, 0], m.p[p, 1], m.p[p, 2], m.p[p, 3], m.p[p, 4], m.p[p, 5], m.p[p, 6], ],
                type=patch_to_type[p],
            )
            return distance <= m.upperbound
        return pyo.Constraint.Skip


    set_of_internal_vertices = set()
    for k, v in patch_to_internal_idx.items():
        set_of_internal_vertices = set_of_internal_vertices.union(set(v))

    model.distanceConstraint = pyo.Constraint(model.set_of_patches, set_of_internal_vertices, rule=distance_constraint_rule)

    model.bbcont = pyo.Var(model.set_of_junction_edges, within=pyo.NonNegativeReals,
                           bounds=[0, 1 + 1e-07], initialize=0.5)

    junction_vertices_idx = np.array(list(set(svg_vertices_idx) - set(boundary_vertices_idx)))

    def eval_objective(
            mbb, mp, mx, my, mz, debug=False,
    ):
        return edge_objective_function(
            set_of_patches=model.set_of_patches,
            set_of_junctions=model.set_of_junctions,
            mbb=mbb,
            mp=mp,
            mx=mx,
            my=my,
            mz=mz,
            init_x=init_triang_x,
            init_y=init_triang_y,
            init_z=init_triang_z,
            junction_edges=junction_edges,
            edges_along_path=edges_along_path,
            junction_edge_to_patch=joint_edges_to_patch,
            boundary_vertices=boundary_vertices_idx,
            junction_vertices=junction_vertices_idx,
            svg_vertex_to_patches=svg_vertex_to_patch,
            patch_to_type=patch_to_type,
            patch_to_internal_idx=patch_to_internal_idx,
            aligned_pairs=aligned_pairs,
            orthogonal_pairs=orthogonal_pairs,
            debug=debug,
            weight_binary=opt_weights["binary"],
            weight_junction_displacement=opt_weights["displacement"],
            weight_path_smoothness=opt_weights["smooth"],
            weight_path_foreshortening=opt_weights["foreshortening"],
        )

    model.xx = pyo.Var(initialize=1.5)

    def rosenbrock(m):
        return (1.6 - m.xx) ** 2 + opt_weights["upperbound"] * m.upperbound

    model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)

    model.obj += eval_objective(mbb=model.bbcont, mp=model.p, mx=model.x, my=model.y, mz=model.z, debug=False)

    # model.obj.pprint()
    solver = pyo.SolverFactory('ipopt')
    solver.options['max_iter'] = max_iterations
    print(f"-- SOLVE for {max_iterations} iterations")
    status = solver.solve(model, tee=True, report_timing=True)
    # pyo.assert_optimal_termination(status)
    print(status)

    print("========= UPPER BOUND")
    model.upperbound.display()
    print("=========")

    # model.bbcont.display()

    opt_p = np.zeros((n_patches, 7), dtype=float)
    for p in model.set_of_patches:
        newp = list()
        for i in range(len(patch_to_init_params[p])):
            newp.append(pyo.value(model.p[p, i]))
            opt_p[p, i] = pyo.value(model.p[p, i])

    def var_to_np_array(vv, id_set=model.set_of_junction_edges):
        l = list()
        for i in id_set:
            l.append(pyo.value(vv[i]))
        return np.array(l)

    opt_bb = var_to_np_array(model.bbcont)
    # print(opt_bb)
    opt_x = var_to_np_array(model.x, id_set=model.set_of_junctions)
    opt_y = var_to_np_array(model.y, id_set=model.set_of_junctions)
    opt_z = var_to_np_array(model.z, id_set=model.set_of_junctions)

    eval_objective(
        mp=opt_p,
        mbb=opt_bb,
        mx=opt_x,
        my=opt_y,
        mz=opt_z,
        debug=True,
    )

    def report_edge(ie: int):
        print(f"\n====\nEdge {ie}: ({junction_edges[ie]})")
        v1, v2 = junction_edges[ie][0], junction_edges[ie][1]
        print(f"v{v1} xyz: {opt_x[v1]:.3f}, {opt_y[v1]:.3f}, {opt_z[v1]:.3f}")
        print(f"v{v2} xyz: {opt_x[v2]:.3f}, {opt_y[v2]:.3f}, {opt_z[v2]:.3f}")
        midx, midy, midz = 0.5 * (opt_x[v1] + opt_x[v2]), 0.5 * (opt_y[v1] + opt_y[v2]), 0.5 * (opt_z[v1] + opt_z[v2])
        print(f"Connected patches: {joint_edges_to_patch[ie]}")
        print(f"Binary decision: {opt_bb[ie]}")
        ra, rb = joint_edges_to_patch[ie]
        dist, midpatch = distance_between_patches(
            type1=patch_to_type[ra],
            type2=patch_to_type[rb],
            params1=opt_p[ra, :],
            params2=opt_p[rb, :],
            x=midx,
            y=midy,
            z=midz,
        )
        distA = evaluate_distance(
            x=midx,
            y=midy,
            z=midz,
            params=opt_p[ra, :],
            type=patch_to_type[ra],
        )
        distB = evaluate_distance(
            x=midx,
            y=midy,
            z=midz,
            params=opt_p[rb, :],
            type=patch_to_type[rb],
        )
        print(f"Distance between patches: {dist}")
        print(f"Distance to patch {ra} {patch_to_type[ra]}: {distA}")
        print(f"Distance to patch {rb} {patch_to_type[rb]}: {distB}")
        print(f"Mid-patch point Z: {midpatch}")

    # report_edge(10)
    try:
        # report_edge(49)
        # report_edge(17)
        report_edge(1)
        report_edge(2)
        report_edge(3)
        # report_edge(48)
        # report_edge(71)
        # report_edge(75)
    except:
        warnings.warn(f"Failed to report edge some edge")

    return opt_bb, opt_p, opt_x, opt_y, opt_z


def edgeCutOptimization(
        pngname: str,
        weights_opt=None,
        max_iterations=40000,
):
    with open(f"results/{pngname}/pkl/triang_regions_logs.pkl", "rb") as f:
        input_svg_vertices, \
            input_dict_region_to_junction_triangulation_vertices_idx, \
            input_dict_region_to_internal_triangulation_vertices_idx, \
            map_junction_to_patches, \
            map_vertex_to_patches = pickle.load(f)

    with open(f"results/{pngname}/pkl/triangulation_logs.pkl", "rb") as f:
        triangulation, \
            svg_points, \
            svg_edges, \
            svg_paths_edges, \
            input_triang_x, \
            input_triang_y, \
            input_triang_z, = pickle.load(f)

    with open(f"results/{pngname}/pkl/localfit_logs.pkl", 'rb') as f:
        refined_pixelsZ, \
            input_dict_trapregion_to_type, \
            input_dict_patch_to_params, = pickle.load(f)

    input_n_patches = max(input_dict_trapregion_to_type.keys()) + 1
    log3d_mesh(
        v_x=input_triang_x,
        v_y=input_triang_y,
        v_z=input_triang_z,
        faces=np.copy(triangulation.faces),
        name=f"0_{pngname}_triang",
        saveto=Path(f"results/{pngname}/"),
    )
    triangulation_vertices = np.stack(
        (
            input_triang_x,
            input_triang_y,
            input_triang_z,
        ),
        axis=1,
    )
    write_patches_triang(
        verts=triangulation_vertices,
        faces=triangulation.faces,
        patch_to_internal_vert=input_dict_region_to_internal_triangulation_vertices_idx,
        patch_to_junction_vert=input_dict_region_to_junction_triangulation_vertices_idx,
        saveto=Path(f"results/{pngname}/predicted_patches")
    )

    for p in range(2, input_n_patches):
        patch_params = input_dict_patch_to_params[p]
        patch_type = input_dict_trapregion_to_type[p]
        patch_v_idx = input_dict_region_to_junction_triangulation_vertices_idx[p].tolist()
        patch_v_idx.extend(input_dict_region_to_internal_triangulation_vertices_idx[p].tolist())
        patch_verts, patch_faces = cut_mesh_by_points(oldV=triangulation_vertices, oldF=triangulation.faces, list_of_vertices=patch_v_idx)
        if len(patch_faces) > 0:
            # little hack for cylinders
            patch_verts[:, 2] = np.mean(patch_verts[:, 2])
            projected_verts = project_points_on_patch(
                init_verts=patch_verts,
                idx_points_to_project=np.arange(len(patch_verts)),
                this_patch_type=patch_type,
                this_patch_params=patch_params,
            )
            igl.write_obj(f"results/{pngname}/predicted_patches/localfit_{p}.obj", projected_verts, patch_faces)

    # map_junction_to_patches = {x: set() for x in input_pure_junction_vertices}
    # for p in range(2, input_n_patches):
    #     for x in input_dict_region_to_junction_triangulation_vertices_idx[p]:
    #         map_junction_to_patches[x].add(p)

    joint_edges = list()  # [[v1, v2], [vA, vB], ...] contains only the edges on joint boundary
    map_region_to_joint_edge = {x: [] for x in range(2, input_n_patches)}
    map_joint_edge_to_regions = dict()
    boundary_edges = list()  # [[v1, v2], [vA, vB], ...] contains only the edge on the boundary
    map_region_to_boundary_edge = {x: [] for x in range(2, input_n_patches)}
    map_boundary_edge_to_region = dict()
    joint_edges_idx = list()  # idx of joint edges from svg_edges list
    boundary_edges_idx = list()  # idx of boundary edges from svg_edges list
    for i_e in range(len(svg_edges)):
        e = svg_edges[i_e]
        ra = set(map_junction_to_patches[e[0]])
        rb = set(map_junction_to_patches[e[1]])
        rab = ra.intersection(rb)
        if len(rab) == 0:
            # the edge is not connected to any region
            warnings.warn(f"edge {i_e} {svg_edges[i_e]} is not connected to any region {e[0]}({ra}) {e[1]}({rb}")
            continue
        if len(rab) > 1:
            if len(rab) > 2:
                warnings.warn(f"edge {e} belongs to 3 patches {rab}, we skip it")
                continue
            for r in rab:
                map_region_to_joint_edge[r].append(len(joint_edges))
            map_joint_edge_to_regions[len(joint_edges)] = tuple(rab)
            joint_edges.append(e.tolist())
            joint_edges_idx.append(i_e)
        else:
            # then we have a boundary edge
            r = rab.pop()
            map_region_to_boundary_edge[r].append(len(boundary_edges))
            map_boundary_edge_to_region[len(boundary_edges)] = r
            boundary_edges.append(e.tolist())
            boundary_edges_idx.append(i_e)

    m_edges = svg_edges.shape[0]
    rowdata = np.arange(m_edges * 2) // 2
    coldata = svg_edges.flatten()
    midpoint_matrix = csr_matrix(
        (
            0.5 * np.ones(2*m_edges),
            (
                rowdata,
                coldata
            )
        ),
        shape=(m_edges, triangulation.n_svg_points),
    )
    edge_midpoints = midpoint_matrix.dot(triangulation_vertices[:triangulation.n_svg_points, :])

    joint_edge_midpoints = edge_midpoints[joint_edges_idx, :]
    print(f"Joint edges {len(joint_edges)}: ", joint_edges)

    init_patch_axis = patches_to_axis_dict(
        dict_patch_type=input_dict_trapregion_to_type,
        dict_patch_params=input_dict_patch_to_params,
    )

    aligned_axis_pairs, orthogonal_axis_pairs = find_aligned_ortho_pairs(
        patch_to_axis_dict=init_patch_axis,
        angle_threshold=0.25,
        patch_to_int_triang_vertices=input_dict_region_to_internal_triangulation_vertices_idx,
        patch_to_junct_triang_vertices=input_dict_region_to_junction_triangulation_vertices_idx,
    )

    print(f"Found aligned patches: ", aligned_axis_pairs)
    print(f"Found orthogonal patches: ", orthogonal_axis_pairs)

    set_vertices_on_boundary = set()
    EV, FE, EF = igl.edge_topology(v=triangulation_vertices, f=triangulation.faces)
    for i in range(EF.shape[0]):
        if (EF[i][0] == -1) or (EF[i][1] == -1):
            set_vertices_on_boundary.add(EV[i][0])
            set_vertices_on_boundary.add(EV[i][1])
    set_lonely_vertices = {x for x, v in map_junction_to_patches.items() if len(v) == 1}
    # TODO: this "intersection" seems a bit redundant, if vertex is lonely the same logic applies as for boundary
    set_boundary_vertices = set_vertices_on_boundary.intersection(set_lonely_vertices)
    boundary_vertices = np.array(sorted(list(set_boundary_vertices)))

    if weights_opt is None:
        weights_opt = dict()
        weights_opt["upperbound"] = 1.0
        weights_opt["binary"] = 0.005
        weights_opt["displacement"] = 10
        weights_opt["smooth"] = 100
        weights_opt["foreshortening"] = 1

    optBB, optParams, optX, optY, optZ = pyomo_edge_optimization(
        junction_edges=joint_edges,
        svg_vertices_idx=input_svg_vertices,
        joint_edges_to_patch=map_joint_edge_to_regions,
        n_patches=input_n_patches,
        patch_to_init_params=input_dict_patch_to_params,
        init_triang_x=input_triang_x,
        init_triang_y=input_triang_y,
        init_triang_z=input_triang_z,
        patch_to_type=input_dict_trapregion_to_type,
        patch_to_internal_idx=input_dict_region_to_internal_triangulation_vertices_idx,
        aligned_pairs=aligned_axis_pairs,
        orthogonal_pairs=orthogonal_axis_pairs,
        boundary_vertices_idx=np.array(list(set_boundary_vertices)),
        svg_vertex_to_patch=map_junction_to_patches,
        edges_along_path=svg_paths_edges,
        max_iterations=max_iterations,
        opt_weights=weights_opt,
    )

    print("Diff X: ", np.linalg.norm(optX - input_triang_x[:len(input_svg_vertices)]))
    print("Diff Y: ", np.linalg.norm(optY - input_triang_y[:len(input_svg_vertices)]))
    print("Diff Z: ", np.linalg.norm(optZ - input_triang_z[:len(input_svg_vertices)]))

    optParams_dict = dict()
    for p in input_dict_patch_to_params.keys():
        if len(input_dict_patch_to_params[p]) > 0:
            print(f"\nPatch {p} ({input_dict_trapregion_to_type[p]}):")
            print(input_dict_patch_to_params[p])
            print(optParams[p])
            optParams_dict[p] = optParams[p]

    for p1 in range(2, input_n_patches-1):
        if input_dict_trapregion_to_type[p1] not in ["Plane", "Cylinder"]:
            continue
        for p2 in range(p1+1, input_n_patches):
            if input_dict_trapregion_to_type[p2] not in ["Plane", "Cylinder"]:
                continue
            print(f"---{p1}-{p2}")
            a1, a2, a3 = get_patch_axis(optParams, p1, p_type=input_dict_trapregion_to_type[p1])
            b1, b2, b3 = get_patch_axis(optParams, p2, p_type=input_dict_trapregion_to_type[p2])
            print(f"Axis {p1} ({input_dict_trapregion_to_type[p1]}) -> {a1}, {a2}, {a3}")
            print(f"Axis {p2} ({input_dict_trapregion_to_type[p2]}) -> {b1}, {b2}, {b3}")
            norma = pyo.sqrt(a1**2 + a2**2 + a3**2)
            normb = pyo.sqrt(b1**2 + b2**2 + b3**2)
            print(f"|| A{p1} || = {norma}")
            print(f"|| A{p2} || = {normb}")
            dotprod = (a1 * b1 + a2 * b2 + a3 * b3) / (norma * normb)
            acos = np.arccos(dotprod)
            print(f"A{p1} . A{p2} = {dotprod}")
            print(f"arccos (A{p1} . A{p2}) = {acos} == {acos * 180 / 3.14}")

    cmap = matplotlib.cm.get_cmap('tab20')
    np.random.seed(42)
    plt.figure()
    plt.title("flat edges after edge optimization")
    plt.scatter(optX, optY, color='white', edgecolors='k', s=3, linewidths=0.2, zorder=5, label="opt positions")
    for i_e in range(len(svg_paths_edges)):
        e_l = svg_paths_edges[i_e]
        edges = np.array(e_l)
        thiscolor = cmap(np.random.randint(20) / 20)
        plt.plot([optX[edges[:, 0]], optX[edges[:, 1]]],
                 [optY[edges[:, 0]], optY[edges[:, 1]]],
                 color=thiscolor,
                 linewidth=1.2,
                 zorder=4,
                 alpha=0.4,
                 )
        plt.plot([input_triang_x[edges[:, 0]], input_triang_x[edges[:, 1]]],
                 [input_triang_y[edges[:, 0]], input_triang_y[edges[:, 1]]],
                 color=thiscolor,
                 linewidth=0.8,
                 zorder=3,
                 linestyle='dotted',
                 )
    plt.axis("equal")
    plt.legend()
    origpath = pathlib.Path(f"results/{pngname}") / "reports/"
    plt.savefig(origpath / "edgeOpt_edges.svg")
    plt.close()

    np.random.seed(42)
    plt.figure()
    for i_e in range(len(svg_paths_edges)):
        e_l = svg_paths_edges[i_e]
        edges = np.array(e_l)
        thiscolor = cmap(np.random.randint(20) / 20)
        plt.plot([input_triang_x[edges[:, 0]], input_triang_x[edges[:, 1]]],
                 [input_triang_y[edges[:, 0]], input_triang_y[edges[:, 1]]],
                 color=thiscolor,
                 linewidth=0.8,
                 zorder=3,
                 )
    plt.axis("equal")
    plt.axis("off")
    origpath = pathlib.Path(f"results/{pngname}") / "reports/"
    plt.savefig(origpath / "paper_edges.svg")
    plt.close()

    # raise NotImplemented

    # print(optBB)

    dislocated_vertices = np.stack(
        (
            optX,
            optY,
            optZ,
        ),
        axis=1,
    )
    new_edge_midpoints = midpoint_matrix.dot(dislocated_vertices[:triangulation.n_svg_points, :])
    joint_edge_midpoints = new_edge_midpoints[joint_edges_idx, :]

    cmap = matplotlib.cm.get_cmap('tab20')
    np.random.seed(42)
    plt.figure()
    plt.scatter(optX[:triangulation.n_svg_points], optY[:triangulation.n_svg_points],
                color="white", edgecolors='k', s=6, linewidths=0.2, zorder=5)
    plt.scatter(joint_edge_midpoints[:, 0], joint_edge_midpoints[:, 1], marker='x', s=6, c=optBB, zorder=6,
                vmin=0, vmax=1,)
    plt.colorbar()
    for idx in range(triangulation.n_svg_points):
        v1 = [optX[idx], optY[idx]]
        plt.text(v1[0], v1[1], s=f"v{idx}",
                 fontsize=1,
                 zorder=8,
                 color='gray',
                 )
    for idx in range(len(joint_edge_midpoints)):
        v1 = joint_edge_midpoints[idx]
        plt.text(v1[0], v1[1], s=f"e{idx}",
                 fontsize=1,
                 zorder=12,
                 color='red',
                 )
    for j in range(len(svg_paths_edges)):
        e_l = svg_paths_edges[j]
        edges = np.array(e_l)
        plt.plot([optX[edges[:, 0]], optX[edges[:, 1]]],
                 [optY[edges[:, 0]], optY[edges[:, 1]]],
                 color=cmap(np.random.randint(20) / 20),
                 linewidth=0.2,
                 )
    for ie in range(len(joint_edges)):
        ee = joint_edges[ie]
        color = "yellow" if round(optBB[ie]) == 1 else "k"
        plt.plot([optX[ee[0]], optX[ee[1]]],
                 [optY[ee[0]], optY[ee[1]]],
                 color=color,
                 linewidth=2,
                 label="nocut" if round(optBB[ie]) == 1 else "cut",
                 )
    plt.axis("equal")
    plt.title(f"Edges")
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    plt.legend(*zip(*unique))
    plt.savefig(f"results/{pngname}/reports/edgeOpt_cuts.svg")
    plt.close()

    plt.figure()
    for j in range(len(svg_paths_edges)):
        e_l = svg_paths_edges[j]
        edges = np.array(e_l)
        plt.plot([optX[edges[:, 0]], optX[edges[:, 1]]],
                 [optY[edges[:, 0]], optY[edges[:, 1]]],
                 color='gray',
                 linewidth=1,
                 )
    for ie in range(len(joint_edges)):
        ee = joint_edges[ie]
        if round(optBB[ie]) == 1:
            continue
        plt.plot([optX[ee[0]], optX[ee[1]]],
                 [optY[ee[0]], optY[ee[1]]],
                 color="red",
                 linewidth=1,
                 )
    plt.axis("equal")
    plt.axis("off")
    plt.savefig(f"results/{pngname}/reports/paper_cuts.svg")
    plt.close()

    np.savez_compressed(
        file=Path(f"results/{pngname}/npz/") / f"edgeresult_{pngname}_improved_params.npz",
        patches=np.array([v for key, v in input_dict_trapregion_to_type.items()]),
        params=optParams,
    )

    improved_triang_z = np.copy(input_triang_z)  # update with projection

    init_triang_verts = np.stack(
        (input_triang_x, input_triang_y, input_triang_z),
        axis=1,
    )

    for i_patch in range(2, input_n_patches):
        this_patch_params = optParams[i_patch]
        print(f"patch {i_patch} - {input_dict_trapregion_to_type[i_patch]}, params: ", this_patch_params)
        this_patch_triang_internal_idx = input_dict_region_to_internal_triangulation_vertices_idx[i_patch]
        projected_points = project_points_on_patch(
            init_verts=init_triang_verts,
            idx_points_to_project=this_patch_triang_internal_idx,
            this_patch_type=input_dict_trapregion_to_type[i_patch],
            this_patch_params=this_patch_params,
        )
        improved_triang_z[this_patch_triang_internal_idx] = projected_points[:, 2]

    for i_vertex in input_svg_vertices:
        if len(map_junction_to_patches[i_vertex]) == 1:
            if i_vertex < len(optZ):
                improved_triang_z[i_vertex] = optZ[i_vertex]

    temp_x, temp_y = np.copy(input_triang_x), np.copy(input_triang_y)
    temp_x[input_svg_vertices] = optX
    temp_y[input_svg_vertices] = optY

    log3d_mesh(
        v_x=temp_x,
        v_y=temp_y,
        v_z=improved_triang_z,
        faces=np.copy(triangulation.faces),
        name=f"edge_{pngname}_after_opt",
        saveto=Path(f"results/{pngname}/"),
    )

    adjacency_list = igl.adjacency_list(f=triangulation.faces)
    set_junctions = set(input_svg_vertices)

    def cut_condition(
            id_v,
    ):
        ring1set = set(adjacency_list[id_v]) - set_junctions
        ring1list = list(ring1set)
        if len(ring1list) < 2:
            return False
        minz = np.min(improved_triang_z[ring1list])
        maxz = np.max(improved_triang_z[ring1list])
        return maxz - minz > 0.3

    cylinder_plane_potential_cuts = []
    for i_e in range(len(joint_edges)):
        v1, v2 = joint_edges[i_e]
        r1, r2 = map_joint_edge_to_regions[i_e]
        t1, t2 = input_dict_trapregion_to_type[r1], input_dict_trapregion_to_type[r2]
        if ((t1, t2) == ("Plane", "Cylinder")) or ((t1, t2) == ("Cylinder", "Plane")) or ((t1, t2) == ("Cylinder", "Cylinder")):
            print(f"edge {v1, v2} is a candidate")
            if cut_condition(v1) and cut_condition(v2):
                print("add this to cuts!")
                cylinder_plane_potential_cuts.append(i_e)

    print(cylinder_plane_potential_cuts)

    with open(Path(f"results/{pngname}/pkl/") / "edge_opt_res.pkl", 'wb') as f:
        pickle.dump(
            [
                optX,
                optY,
                optZ,
                joint_edges,
                boundary_vertices,  # free vertices on original svg boundary
                optBB,
                optParams,
                optParams_dict,
                cylinder_plane_potential_cuts,
                input_dict_trapregion_to_type,
            ],
            f,
            protocol=-1,
        )


def edgeCutTriangulation(
        pngname: str,
        triang_flags='YYqpa50',
):
    predicted_depth = np.load(f"results/{pngname}/npz/{pngname}_depth.npz")["depth"]
    with open(f"results/{pngname}/pkl/triangulation_logs.pkl", "rb") as f:
        triangulation, \
            svg_points, \
            svg_edges, \
            svg_paths_edges, \
            input_triang_x, \
            input_triang_y, \
            input_triang_z, = pickle.load(f)

    with open(f"results/{pngname}/pkl/edge_opt_res.pkl", "rb") as f:
        optX, \
            optY, \
            optZ, \
            joint_edges, \
            svg_free_boundary_vertices, \
            optBB, \
            optParams, \
            optParams_dict, \
            cylinder_plane_potential_cuts, \
            input_dict_trapregion_to_type = pickle.load(f)

    with open(f"results/{pngname}/pkl/trappedball_logs.pkl", "rb") as f:
        _, \
            _, \
            _, \
            fillmap, \
            thin_fillmap, = pickle.load(f)

    with open(f"results/{pngname}/pkl/triang_regions_logs.pkl", "rb") as f:
        input_pure_junction_vertices, \
            input_dict_region_to_junction_triangulation_vertices_idx, \
            input_dict_region_to_internal_triangulation_vertices_idx, \
            map_junction_to_patches, \
            map_vertex_to_patches = pickle.load(f)

    # add more edges to the cut or suppress noise
    input_n_patches = max(optParams_dict.keys()) + 1
    edges_to_cut = list()
    set_edge_cuts = set()
    for i_e in range(len(joint_edges)):
        je = joint_edges[i_e]
        print("joint edge: ", je)
        if (round(optBB[i_e]) != 1) or (i_e in cylinder_plane_potential_cuts):
            edges_to_cut.append(je)
            set_edge_cuts.add(tuple(je))

    for i_p in range(len(svg_paths_edges)):
        n_cuts_here = 0
        set_cuts_here = set()
        for e in svg_paths_edges[i_p]:
            if tuple(e) in set_edge_cuts:
                n_cuts_here += 1
                set_cuts_here.add(tuple(e))
        if (n_cuts_here > 0.9 * len(svg_paths_edges[i_p])) or \
                ((n_cuts_here >= len(svg_paths_edges[i_p]) - 1) and (len(svg_paths_edges[i_p])) > 2) or \
                ((n_cuts_here <= 2) and (n_cuts_here >= 0.5 * len(svg_paths_edges[i_p]))):
            print(f"We should add entire path {i_p} to cut edges")
            set_edge_cuts = set_edge_cuts.union(set(tuple(x) for x in svg_paths_edges[i_p]))
        if (n_cuts_here <= 2) and (n_cuts_here / len(svg_paths_edges[i_p]) < 0.25):
            print(f"Cut is too short, avoid it?")
            set_edge_cuts = set_edge_cuts - set_cuts_here

    edges_to_cut = list(set_edge_cuts)
    print(edges_to_cut)

    with open(Path(f'results/{pngname}/edges_to_cut.json'), 'w') as f:
        dd = [[int(x[0]), int(x[1])] for x in edges_to_cut]
        json.dump(dd, f)
    
    if os.path.exists(Path(f'results/{pngname}/MANUAL_edges_to_cut.json')):
        with open(Path(f'results/{pngname}/MANUAL_edges_to_cut.json')) as f:
            json_data = json.load(f)
            edges_to_cut = [(int(x[0]), int(x[1])) for x in json_data]
    
    plt.figure()
    plt.scatter(optX[:triangulation.n_svg_points], optY[:triangulation.n_svg_points],
                color="white", edgecolors='k', s=6, linewidths=0.2, zorder=5)
    plt.colorbar()
    for idx in range(triangulation.n_svg_points):
        v1 = [optX[idx], optY[idx]]
        plt.text(v1[0], v1[1], s=f"v{idx}",
                 fontsize=2,
                 zorder=8,
                 color='gray',
                 )
    cmap = matplotlib.cm.get_cmap('tab20')
    for thisedge in edges_to_cut:
        plt.plot([optX[thisedge[0]], optX[thisedge[1]]],
                 [optY[thisedge[0]], optY[thisedge[1]]],
                 color=cmap(np.random.randint(20) / 20),
                 linewidth=1,
                 )
    plt.axis("equal")
    plt.savefig(f"results/{pngname}/reports/edge_cuts.svg")
    plt.close()

    # run ARAP to get new triangulation
    new_triang_x, new_triang_y = deform_triangle(
        faces=triangulation.faces,
        ind_b=np.arange(triangulation.n_svg_points),
        initX=input_triang_x,
        initY=input_triang_y,
        newX_b=optX,
        newY_b=optY,
    )
    flatPointsPixels = coordinates_to_pixels(orig_image=predicted_depth, coordsX=new_triang_x, coordsY=new_triang_y)
    triangulation.vertices[:, :2] = flatPointsPixels[:, :2]

    triangulation.plot(faces=True, show=False, saveto=pathlib.Path(f"results/{pngname}") / "reports/", name="newtriangulation")

    # cut triangulation
    triangulation, dict_region_to_internal_triangulation_vertices_idx, \
        dict_region_to_junction_triangulation_vertices_idx, junction_vertices, \
        lists_of_duplicates, list_of_chains = cutExistingTriangulation(
            triang=triangulation,
            svg_points=svg_points,
            svg_edges=svg_edges,
            depthimage=predicted_depth,
            saveto=pathlib.Path(f"results/{pngname}") / "reports/",
            edges_to_cut=edges_to_cut,
            dict_region_to_junctions=input_dict_region_to_junction_triangulation_vertices_idx,
            dict_vertex_to_region=map_vertex_to_patches,
            pure_junctions=input_pure_junction_vertices,
        )

    pixels_to_camera_coordinates = get_pixel_coordinates(depth_values=predicted_depth)

    pixelsX = pixels_to_camera_coordinates[..., 0]
    pixelsY = pixels_to_camera_coordinates[..., 1]
    pixelsZ = pixels_to_camera_coordinates[..., 2]

    triang_x = triangulation.interpolate_f_on_vertices(f_grid=pixelsX)
    triang_y = triangulation.interpolate_f_on_vertices(f_grid=pixelsY)
    triang_z_prediction = triangulation.interpolate_f_on_vertices(f_grid=pixelsZ)  # keep depth prediction here
    improved_triang_z = np.copy(triang_z_prediction)   # update with projection

    init_triang_verts = np.stack(
        (triang_x, triang_y, triang_z_prediction),
        axis=1,
    )

    for i_patch in range(2, input_n_patches):
        this_patch_params = optParams[i_patch]
        print(f"patch {i_patch} - {input_dict_trapregion_to_type[i_patch]}, params: ", this_patch_params)
        this_patch_triang_internal_idx = dict_region_to_internal_triangulation_vertices_idx[i_patch]
        projected_points = project_points_on_patch(
            init_verts=init_triang_verts,
            idx_points_to_project=this_patch_triang_internal_idx,
            this_patch_type=input_dict_trapregion_to_type[i_patch],
            this_patch_params=this_patch_params,
        )
        improved_triang_z[this_patch_triang_internal_idx] = projected_points[:, 2]

    improved_triang_z[input_pure_junction_vertices] = optZ

    # this mesh how has cuts, but positions of points on cuts and on joint boundaries are still unknown
    log3d_mesh(
        v_x=triang_x,
        v_y=triang_y,
        v_z=improved_triang_z,
        faces=np.copy(triangulation.faces),
        name=f"edge_{pngname}_triang_after_cut",
        saveto=Path(f"results/{pngname}/"),
    )

    with open(Path(f"results/{pngname}/") / "pkl/edge_opt_logs.pkl", 'wb') as f:
        pickle.dump(
            [
                triangulation,
                svg_points,
                svg_paths_edges,
                optParams_dict,
                triang_x,
                triang_y,
                triang_z_prediction,
                improved_triang_z,
                lists_of_duplicates,
                list_of_chains,
                junction_vertices,
                svg_free_boundary_vertices,
                input_dict_trapregion_to_type,
                dict_region_to_junction_triangulation_vertices_idx,
                dict_region_to_internal_triangulation_vertices_idx,
            ],
            f,
            protocol=-1,
        )


if __name__ == "__main__":
    myname = "machine124a"
    # pngname = "blender_cut"
    # pngname = "npr_1013_60.45_-125.3_1.4"
    # pngname = "npr_1039_62.53_128.64_1.4"
    # pngname = "Cylindrical_Parts_011_1"
    # pngname = "Prismatic_Stock_017_1"

    parser = argparse.ArgumentParser(description="Run all our scripts")
    parser.add_argument(
        "--pngname",
        type=str,
        default=myname,
        help="name of image",
    )
    parser.add_argument(
        "--u",
        type=float,
        default=1.0,
        help="weight on upperbound",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.005,
        help="weight on binary variable",
    )
    parser.add_argument(
        "--d",
        type=float,
        default=10,
        help="weight on svg vertices displacement",
    )
    parser.add_argument(
        "--s",
        type=float,
        default=100,
        help="weight on stroke smoothness",
    )
    parser.add_argument(
        "--f",
        type=float,
        default=1,
        help="weight on stroke foreshortening",
    )
    parser.add_argument("--maxiter", type=int, default=10000, help="number of iterations")
    args = parser.parse_args()

    w_opt = dict()
    w_opt["upperbound"] = args.u
    w_opt["binary"] = args.b
    w_opt["displacement"] = args.d
    w_opt["smooth"] = args.s
    w_opt["foreshortening"] = args.f

    edgeCutOptimization(pngname=args.pngname, weights_opt=w_opt)
    edgeCutTriangulation(pngname=args.pngname)
