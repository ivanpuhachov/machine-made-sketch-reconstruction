import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import scipy.optimize as spo
import scipy.sparse

from test_cylinder_fit import cylinder_alignment_geomfitty, cylinder_alignment_energy_dist, \
    jacobian_params_cylinder_alignment_energy_dist, jacobian_points_cylinder_alignment_energy_dist
from plane_fit import plane_eval_distZ2, grad_params_plane_distZ2, grad_points_plane_distZ2
from sphere_fit import sphere_alignment_energy2, grad_points_sphere_alignment_energy2, grad_params_sphere_alignment_energy2
from axis_alignment_term import patch_params_to_axis, axis_alignment_energy, grad_axis_alignment, \
    axis_othogonality_energy, grad_axis_orthogonality
import warnings
from timeit import default_timer as timer
import pickle


def split_x0_to_patch_params(
        myx,
        init_patch_to_params,
        n_p,
):
    """

    :param myx: np array of params
    :param init_patch_to_params: dict with patch params, used for length only
    :param n_p: number of patches total
    :return:
    """
    split_params = dict()
    last_index = 0
    for i in range(2, n_p):
        len_patch_params = len(init_patch_to_params[i])
        patch_params = myx[last_index:last_index + len_patch_params]
        last_index += len_patch_params
        split_params[i] = patch_params
    return split_params


def split_x0_to_junction_xyz(
        myx,
        number_patch_params,
        number_junction_vertices,
):
    """
    split incoming X into junction positions x y z
    :param myx: incoming flat X from optimization
    :param number_patch_params: patch params to slice from this offset
    :param number_junction_vertices:
    :return: jx, jy, jz of size number_junction_vertices each
    """
    jx = myx[number_patch_params: number_patch_params + number_junction_vertices]
    jy = myx[number_patch_params + number_junction_vertices: number_patch_params + 2 * number_junction_vertices]
    jz = myx[number_patch_params + 2 * number_junction_vertices: number_patch_params + 3 * number_junction_vertices]
    return jx, jy, jz


def patches_to_axis_dict(
        dict_patch_type: dict,
        dict_patch_params: dict,
):
    """
    extracts axis if patch has axis (cylinder, cone, plane)
    :param dict_patch_type: dict of patch types
    :param dict_patch_params: dict of patch parameters, key -> np array
    :return:
    """
    res = dict()
    for i, t in dict_patch_type.items():
        if t not in ["Plane", "Cylinder"]:
            continue
        res[i] = patch_params_to_axis(
            type_of_patch=t,
            params=dict_patch_params[i]
        )
    return res


def build_junction_to_patches(
        patch_to_junctions: dict,
):
    temp = defaultdict(list)
    junctions = set()
    for i, idxs in patch_to_junctions.items():
        for v in idxs:
            temp[v].append(i)
            junctions.add(v)
    res = dict()
    for i in sorted(list(junctions)):
        res[i] = np.array(temp[i])
    return res


def find_aligned_ortho_pairs(
        patch_to_axis_dict: dict,
        angle_threshold: float,
        patch_to_int_triang_vertices: dict,
        patch_to_junct_triang_vertices: dict,
):
    aligned_pairs = list()
    orthogonal_pairs = list()
    for i in patch_to_axis_dict.keys():
        i_axis = patch_to_axis_dict[i]
        # i_axis = np.copy(init_patch_axis[i][:2])
        # i_axis = i_axis / np.linalg.norm(i_axis)
        for j in patch_to_axis_dict.keys():
            if j <= i:
                continue
            if (len(patch_to_junct_triang_vertices[j]) == 0) or (
                    len(patch_to_int_triang_vertices[j]) == 0):
                continue
            j_axis = patch_to_axis_dict[j]
            # j_axis = np.copy(init_patch_axis[j][:2])
            # j_axis = j_axis / np.linalg.norm(j_axis)
            angle = np.arccos(np.abs(np.dot(i_axis, j_axis)))
            print(f"axis {i} ({i_axis}) and {j} ({j_axis}) (angle {angle})")
            if angle < angle_threshold:
                print("aligned!")
                aligned_pairs.append((i, j))
            if angle > np.pi / 2 - angle_threshold:
                print("orthogonal!")
                orthogonal_pairs.append((i, j))
    return aligned_pairs, orthogonal_pairs


def objective_global_optimization(
        params,
        n_patch_patams: int,
        dict_trapregion_to_type: dict,
        init_dict_patch_to_params: dict,
        n_patches: int,
        aligned_axis_pairs,
        orthogonal_axis_pairs,
        init_junction_points_x: np.array,
        init_junction_points_y: np.array,
        jLtL: scipy.sparse.csr_matrix,
        dict_patch_to_x: dict,
        dict_patch_to_y: dict,
        dict_patch_to_z: dict,
        dict_region_to_junction_triangulation_vertices_idx,
        dict_region_to_internal_triangulation_vertices_idx,
        duplicated_vertices: list,
        debug=False,
        weight_alignment=100.0,
        weight_orthogonality=100.0,
        weight_patches=1.0,
        weight_cylinder=1.0,
        weight_cone=0.5,
        weight_sphere=0.5,
        weight_junction_displacement=10,
        weight_junction_smoothness=1000,
        weight_junction_alignment=500,
        return_grad=False,
):
    """
    Computes objcetive function and its gradient
    :param duplicated_vertices: list of (pair tuples) for vertices that were duplicated in the process
    :param weight_sphere: weight for junction and internals terms for sphere
    :param weight_junction_alignment:
    :param params:
    :param n_patch_patams:
    :param dict_trapregion_to_type:
    :param init_dict_patch_to_params: contains initial guess parameters, used  only for structure
    :param n_patches:
    :param aligned_axis_pairs:
    :param orthogonal_axis_pairs:
    :param init_junction_points_x:
    :param init_junction_points_y:
    :param jLtL: sparse matrix (like laplacian) to enforce smoothness between neighboring junction vertices
    :param dict_patch_to_x:
    :param dict_patch_to_y:
    :param dict_patch_to_z:
    :param dict_region_to_junction_triangulation_vertices_idx:
    :param dict_region_to_internal_triangulation_vertices_idx:
    :param debug:
    :param weight_alignment: axis alignment weight
    :param weight_patches: weight for internal points terms
    :param weight_cylinder:
    :param weight_cone:
    :param weight_junction_displacement:
    :param weight_junction_smoothness:
    :param weight_orthogonality:
    :param return_grad:
    :return:
    """
    energy_value = 0.0
    dict_patch_params = split_x0_to_patch_params(
        myx=params,
        init_patch_to_params=init_dict_patch_to_params,
        n_p=n_patches,
    )  # slice input x0 to get patch params
    dict_patch_axis = patches_to_axis_dict(
        dict_patch_type=dict_trapregion_to_type,
        dict_patch_params=dict_patch_params,
    )   # get axis from patch params
    n_junction_vertices = init_junction_points_x.shape[1]
    junctions_x, junctions_y, junctions_z = split_x0_to_junction_xyz(
        myx=params,
        number_patch_params=n_patch_patams,
        number_junction_vertices=n_junction_vertices,
    )  # slice x0 to get junction positions
    list_of_grad_params = list()
    grad_junctions = np.zeros((n_junction_vertices, 3))
    for i_patch in range(2, n_patches):
        this_patch_triang_junction_idx = dict_region_to_junction_triangulation_vertices_idx[i_patch]
        this_patch_triang_internal_idx = dict_region_to_internal_triangulation_vertices_idx[i_patch]
        patch_params = dict_patch_params[i_patch]
        n_internal_pixel_points = len(dict_patch_to_x[i_patch])
        n_junction_points = len(this_patch_triang_junction_idx)
        if debug:
            print(f"Patch {i_patch} type {dict_trapregion_to_type[i_patch]}, params: ", patch_params)
            print(f"n_internal: {n_internal_pixel_points}, n_junctions: {n_junction_points}")
        this_patch_pixels_x = dict_patch_to_x[i_patch]
        this_patch_pixels_y = dict_patch_to_y[i_patch]
        this_patch_pixels_z = dict_patch_to_z[i_patch]
        this_patch_pixels_points = np.vstack(
            (this_patch_pixels_x, this_patch_pixels_y, this_patch_pixels_z)
        )
        # this_patch_triang_x = triang_x[[this_patch_triang_internal_idx]]
        # this_patch_triang_y = triang_y[[this_patch_triang_internal_idx]]
        # this_patch_triang_z = triang_z[[this_patch_triang_internal_idx]]
        internal_energy = 0
        neighbor_junctions_x = junctions_x[this_patch_triang_junction_idx]
        neighbor_junctions_y = junctions_y[this_patch_triang_junction_idx]
        neighbor_junctions_z = junctions_z[this_patch_triang_junction_idx]
        this_patch_junctions_points = np.vstack(
            (neighbor_junctions_x, neighbor_junctions_y, neighbor_junctions_z)
        )
        junction_energy = 0
        this_patch_grad_params = np.zeros_like(patch_params)
        this_patch_grad_points_x = np.zeros_like(neighbor_junctions_x)
        this_patch_grad_points_y = np.zeros_like(neighbor_junctions_y)
        this_patch_grad_points_z = np.zeros_like(neighbor_junctions_z)
        if (len(this_patch_triang_junction_idx) == 0) or (len(this_patch_triang_internal_idx) == 0):
            warnings.warn(
                f"Patch {i_patch} has {this_patch_triang_junction_idx} junction points and {this_patch_triang_internal_idx} internal points"
            )
            list_of_grad_params.append(this_patch_grad_params)
            continue
        if dict_trapregion_to_type[i_patch] == "Plane":
            # internals
            internal_energy = plane_eval_distZ2(
                x=this_patch_pixels_x,
                y=this_patch_pixels_y,
                z=this_patch_pixels_z,
                a=patch_params[0],
                b=patch_params[1],
                c=patch_params[2],
                d=patch_params[3],
            )
            internal_energy_params_grad = grad_params_plane_distZ2(
                x=this_patch_pixels_x,
                y=this_patch_pixels_y,
                z=this_patch_pixels_z,
                a=patch_params[0],
                b=patch_params[1],
                c=patch_params[2],
                d=patch_params[3],
            )
            energy_value += weight_patches * internal_energy / n_internal_pixel_points
            this_patch_grad_params += weight_patches * internal_energy_params_grad / n_internal_pixel_points
            # junctions
            # junction_energy = plane_eval_dist2(
            junction_energy = plane_eval_distZ2(
                x=neighbor_junctions_x,
                y=neighbor_junctions_y,
                z=neighbor_junctions_z,
                a=patch_params[0],
                b=patch_params[1],
                c=patch_params[2],
                d=patch_params[3],
            )
            junction_energy_params_grad = grad_params_plane_distZ2(
                x=neighbor_junctions_x,
                y=neighbor_junctions_y,
                z=neighbor_junctions_z,
                a=patch_params[0],
                b=patch_params[1],
                c=patch_params[2],
                d=patch_params[3],
            )
            junction_energy_points_grad = grad_points_plane_distZ2(
                x=neighbor_junctions_x,
                y=neighbor_junctions_y,
                z=neighbor_junctions_z,
                a=patch_params[0],
                b=patch_params[1],
                c=patch_params[2],
                d=patch_params[3],
            )

            energy_value += weight_junction_alignment * junction_energy / n_junction_points
            this_patch_grad_params += weight_junction_alignment * junction_energy_params_grad / n_junction_points
            this_patch_grad_points_x += weight_junction_alignment * junction_energy_points_grad[0] / n_junction_points
            this_patch_grad_points_y += weight_junction_alignment * junction_energy_points_grad[1] / n_junction_points
            this_patch_grad_points_z += weight_junction_alignment * junction_energy_points_grad[2] / n_junction_points
            if debug:
                nnn = patch_params[:3]
                print("|| N || = ", wnorm := np.linalg.norm(nnn))
                if not np.allclose(
                    wnorm, 1.0, rtol=2e-2, atol=2e-2, equal_nan=False
                ):
                    warnings.warn(
                        f"Patch {i_patch} normal {nnn} has norm {wnorm}"
                    )
        if dict_trapregion_to_type[i_patch] == "Cylinder":
            # params
            c = np.array([patch_params[0], patch_params[1], patch_params[2]])
            w = np.array([patch_params[3], patch_params[4], patch_params[5]])
            if debug:
                print("|| W || = ", wnorm := np.linalg.norm(w))
                if not np.allclose(
                    wnorm, 1.0, rtol=2e-2, atol=2e-2, equal_nan=False
                ):
                    warnings.warn(
                        f"Patch {i_patch} axis {w} has norm {wnorm}"
                    )
            r2 = patch_params[6]
            # internals
            # internal_energy = cylinder_alignment_geomfitty(
            internal_energy = cylinder_alignment_energy_dist(
                c=c,
                w=w,
                # x=this_patch_pixels_x,
                # y=this_patch_pixels_y,
                # z=this_patch_pixels_z,
                points=this_patch_pixels_points,
            )
            internal_energy_params_grad = jacobian_params_cylinder_alignment_energy_dist(
                c=c,
                w=w,
                points=this_patch_pixels_points,
            )
            internal_energy_params_grad = np.concatenate(internal_energy_params_grad)
            internal_energy_params_grad = np.append(internal_energy_params_grad, [0])   # r2 is stored in params, but is not computed in this optimization

            energy_value += weight_cylinder * weight_patches * internal_energy / n_internal_pixel_points
            this_patch_grad_params += weight_cylinder * weight_patches * internal_energy_params_grad / n_internal_pixel_points

            # junctions
            # junction_energy = cylinder_alignment_geomfitty(
            junction_energy = cylinder_alignment_energy_dist(
                c=c,
                w=w,
                # x=neighbor_junctions_x,
                # y=neighbor_junctions_y,
                # z=neighbor_junctions_z,
                points=this_patch_junctions_points,
            )
            junction_energy_params_grad = jacobian_params_cylinder_alignment_energy_dist(
                c=c,
                w=w,
                points=this_patch_junctions_points,
            )
            junction_energy_params_grad = np.concatenate(junction_energy_params_grad)
            junction_energy_params_grad = np.append(junction_energy_params_grad, [
                0])  # r2 is stored in params, but is not computed in this optimization

            junction_energy_points_grad = jacobian_points_cylinder_alignment_energy_dist(
                c=c,
                w=w,
                points=this_patch_junctions_points,
            )
            energy_value += weight_junction_alignment * weight_cylinder * junction_energy / n_junction_points
            this_patch_grad_params += weight_junction_alignment * weight_cylinder * junction_energy_params_grad / n_junction_points
            this_patch_grad_points_x += weight_junction_alignment * weight_cylinder * junction_energy_points_grad[0, :] / n_junction_points
            this_patch_grad_points_y += weight_junction_alignment * weight_cylinder * junction_energy_points_grad[1, :] / n_junction_points
            this_patch_grad_points_z += weight_junction_alignment * weight_cylinder * junction_energy_points_grad[2, :] / n_junction_points

        # if dict_trapregion_to_type[i_patch] == "Cone":
        #     # params
        #     v = np.array([patch_params[0], patch_params[1], patch_params[2]])
        #     u = np.array([patch_params[3], patch_params[4], patch_params[5]])
        #     theta = patch_params[6]
        #     patchpoints = np.vstack((patchx, patchy, patchz))
        #     # internals
        #     internal_energy = cone_alignenment_energy_geomfitty(
        #         v=v,
        #         u=u,
        #         theta=theta,
        #         points=patchpoints,
        #     )
        #     energy_value += weight_cone * weight_patches * internal_energy / n_internal_pixel_points
        #     # junctions
        #     junction_points = np.vstack((neighbor_junctions_x, neighbor_junctions_y, neighbor_junctions_z))
        #     junction_energy = cone_alignenment_energy_geomfitty(
        #         v=v,
        #         u=u,
        #         theta=theta,
        #         points=junction_points,
        #     )
        #     energy_value += weight_cone * junction_energy / n_junction_points

        if dict_trapregion_to_type[i_patch] == "Sphere":
            c = np.array([patch_params[0], patch_params[1], patch_params[2]])
            r2 = patch_params[3]
            internal_energy = sphere_alignment_energy2(
                points=this_patch_pixels_points,
                c=c,
                r2=r2,
            )
            internal_energy_params_grad = grad_params_sphere_alignment_energy2(
                points=this_patch_pixels_points,
                c=c,
                r2=r2,
            )
            internal_energy_params_grad = np.append(internal_energy_params_grad[0], values=internal_energy_params_grad[1])
            energy_value += weight_sphere * weight_patches * internal_energy / n_internal_pixel_points
            this_patch_grad_params += weight_sphere * weight_patches * internal_energy_params_grad / n_internal_pixel_points

            junction_points = np.vstack((neighbor_junctions_x, neighbor_junctions_y, neighbor_junctions_z))
            junction_energy = sphere_alignment_energy2(
                points=junction_points,
                c=c,
                r2=r2,
            )
            junction_energy_params_grad = grad_params_sphere_alignment_energy2(
                points=junction_points,
                c=c,
                r2=r2,
            )
            junction_energy_params_grad = np.append(junction_energy_params_grad[0], values=junction_energy_params_grad[1])
            junction_energy_points_grad = grad_points_sphere_alignment_energy2(
                points=junction_points,
                c=c,
                r2=r2,
            )
            energy_value += weight_junction_alignment * weight_sphere * junction_energy / n_junction_points
            this_patch_grad_params += weight_junction_alignment * weight_sphere * junction_energy_params_grad / n_junction_points
            this_patch_grad_points_x += weight_junction_alignment * weight_sphere * \
                                        junction_energy_points_grad[0, :] / n_junction_points
            this_patch_grad_points_y += weight_junction_alignment * weight_sphere * \
                                        junction_energy_points_grad[1, :] / n_junction_points
            this_patch_grad_points_z += weight_junction_alignment * weight_sphere * \
                                        junction_energy_points_grad[2, :] / n_junction_points

        if debug:
            print("Internal points energy: ", internal_energy)
            print("Junction points energy: ", junction_energy)
            print("objective value now: ", energy_value)

        list_of_grad_params.append(this_patch_grad_params)
        grad_junctions[this_patch_triang_junction_idx, 0] += this_patch_grad_points_x
        grad_junctions[this_patch_triang_junction_idx, 1] += this_patch_grad_points_y
        grad_junctions[this_patch_triang_junction_idx, 2] += this_patch_grad_points_z

        # print(energy_value)

    # ALIGNMENT AND ORTHOGONALITY
    for al_pair in aligned_axis_pairs:
        axis_alignment_term = axis_alignment_energy(
            a_type=dict_trapregion_to_type[al_pair[0]],
            a_params=dict_patch_params[al_pair[0]],
            b_type=dict_trapregion_to_type[al_pair[1]],
            b_params=dict_patch_params[al_pair[1]],
        )
        if debug:
            ax1 = dict_patch_axis[al_pair[0]]
            ax2 = dict_patch_axis[al_pair[1]]
            dotprod = np.dot(ax1, ax2)
            crossprod = np.linalg.norm(np.cross(ax1, ax2))
            angle = np.arctan2(crossprod, dotprod)
            old_angle_energy = angle ** 2
            print(f"aligned al_pair {al_pair}, {ax1} {ax2}, angle {angle}, OLD energy {old_angle_energy}")
            print(f"aligned al_pair {al_pair}, alignment_term energy value {axis_alignment_term}")
            if np.abs(dotprod) > 1:
                warnings.warn(f"aligned al_pair {al_pair}: ax1.ax2 = {ax1} . {ax2} = {dotprod}")
            if not np.allclose(
                ax1norm := np.linalg.norm(ax1), 1, rtol=2e-2, atol=2e-2,
            ):
                warnings.warn(f"(patch {al_pair[0]}) ax1 {ax1} norm: {ax1norm}")
            if not np.allclose(
                ax2norm := np.linalg.norm(ax2), 1, rtol=2e-2, atol=2e-2,
            ):
                warnings.warn(f"(patch {al_pair[1]}) ax2 {ax2} norm: {ax2norm}")
        energy_value += weight_alignment * axis_alignment_term
        grad1, grad2 = grad_axis_alignment(
            a_type=dict_trapregion_to_type[al_pair[0]],
            a_params=dict_patch_params[al_pair[0]],
            b_type=dict_trapregion_to_type[al_pair[1]],
            b_params=dict_patch_params[al_pair[1]],
        )
        list_of_grad_params[
            al_pair[0] - 2  # this offset happens because region id starts from 2
        ] += weight_alignment * grad1
        list_of_grad_params[
            al_pair[1] - 2  # this offset happens because region id starts from 2
        ] += weight_alignment * grad2

    if debug:
        print("objective value now: ", energy_value)

    for or_pair in orthogonal_axis_pairs:
        ta = dict_trapregion_to_type[or_pair[0]]
        pa = dict_patch_params[or_pair[0]]
        tb = dict_trapregion_to_type[or_pair[1]]
        pb = dict_patch_params[or_pair[1]]
        axis_orthogonality_term = axis_othogonality_energy(
            a_type=dict_trapregion_to_type[or_pair[0]],
            a_params=dict_patch_params[or_pair[0]],
            b_type=dict_trapregion_to_type[or_pair[1]],
            b_params=dict_patch_params[or_pair[1]],
        )
        grad_orthogonality_term = grad_axis_orthogonality(
            a_type=dict_trapregion_to_type[or_pair[0]],
            a_params=dict_patch_params[or_pair[0]],
            b_type=dict_trapregion_to_type[or_pair[1]],
            b_params=dict_patch_params[or_pair[1]],
        )
        energy_value += weight_orthogonality * axis_orthogonality_term
        list_of_grad_params[
            or_pair[0] - 2  # this offset happens because region id starts from 2
            ] += weight_orthogonality * grad_orthogonality_term[0]
        list_of_grad_params[
            or_pair[1] - 2  # this offset happens because region id starts from 2
            ] += weight_orthogonality * grad_orthogonality_term[1]

        if debug:
            print("orthogonal pair: ", or_pair)
            print(f"params {or_pair[0]}: ", pa)
            print(f"params {or_pair[1]}: ", pb)
            print("orthogonality energy: ", axis_orthogonality_term)
            ax1 = dict_patch_axis[or_pair[0]]
            ax2 = dict_patch_axis[or_pair[1]]
            dotprod = np.dot(ax1, ax2)
            crossprod = np.linalg.norm(np.cross(ax1, ax2))
            angle = np.arctan2(crossprod, dotprod)
            print("angle: ", angle)
            if not np.allclose(
                ax1norm := np.linalg.norm(ax1), 1, rtol=2e-2, atol=2e-2,
            ):
                warnings.warn(f"(patch {or_pair[0]}) ax1 {ax1} norm: {ax1norm}")
            if not np.allclose(
                ax2norm := np.linalg.norm(ax2), 1, rtol=2e-2, atol=2e-2,
            ):
                warnings.warn(f"(patch {or_pair[1]}) ax2 {ax2} norm: {ax2norm}")

    # JUNCTION POINTS ARE CLOSE TO INIT POSITIONS
    junction_displacement = np.sum((junctions_x - init_junction_points_x) ** 2) + \
                            np.sum((junctions_y - init_junction_points_y) ** 2)
    energy_value += weight_junction_displacement * junction_displacement / n_junction_vertices
    grad_junctions[:, 0] += 2 * weight_junction_displacement * \
                            (junctions_x - init_junction_points_x.reshape(n_junction_vertices)) / n_junction_vertices
    grad_junctions[:, 1] += 2 * weight_junction_displacement * \
                            (junctions_y - init_junction_points_y.reshape(n_junction_vertices)) / n_junction_vertices

    weight_duplication_overlap = 0
    # DUPLICATED JUNCTIONS OVERLAP IN XY
    for pair in duplicated_vertices:
        # TODO: this should be done in parallel
        vi, vj = pair[0], pair[1]
        energy_value += weight_duplication_overlap * (junctions_x[vi] - junctions_x[vj])**2 + \
                        weight_duplication_overlap * (junctions_y[vi] - junctions_y[vj])**2
        grad_junctions[vi, 0] += 2 * weight_duplication_overlap * (junctions_x[vi] - junctions_x[vj])
        grad_junctions[vj, 0] -= 2 * weight_duplication_overlap * (junctions_x[vi] - junctions_x[vj])
        grad_junctions[vi, 1] += 2 * weight_duplication_overlap * (junctions_y[vi] - junctions_y[vj])
        grad_junctions[vj, 1] -= 2 * weight_duplication_overlap * (junctions_y[vi] - junctions_y[vj])

    if debug:
        print("junction displacement ^2 : ", junction_displacement)
        disp_ = (junctions_x - init_junction_points_x) ** 2 + \
                            (junctions_y - init_junction_points_y) ** 2
        print(f"max XY-displacement is {np.max(disp_)} at {np.argmax(disp_)}")
        print("objective value now: ", energy_value)

    junction_smoothness = 0.1 * junctions_x.dot(jLtL.dot(junctions_x)) + \
                          0.1 * junctions_y.dot(jLtL.dot(junctions_y)) + \
                          junctions_z.dot(jLtL.dot(junctions_z))
    energy_value += weight_junction_smoothness * junction_smoothness / n_junction_vertices
    grad_junctions[:, 0] += 0.1 * weight_junction_smoothness * (jLtL + jLtL.transpose()) * junctions_x / n_junction_vertices
    grad_junctions[:, 1] += 0.1 * weight_junction_smoothness * (jLtL + jLtL.transpose()) * junctions_y / n_junction_vertices
    grad_junctions[:, 2] += weight_junction_smoothness * (jLtL + jLtL.transpose()) * junctions_z / n_junction_vertices
    if debug:
        print("junction smoothness: ", junction_smoothness)
        print("objective value final: ", energy_value)

    grad_params = np.concatenate(list_of_grad_params)

    grad_total = np.concatenate(
        (
            grad_params,
            grad_junctions[:, 0],
            grad_junctions[:, 1],
            grad_junctions[:, 2],
        )
    )

    if return_grad:
        return energy_value, grad_total
    return energy_value


def run_global_optimization(
        x0,
        n_patches,
        n_patches_params,
        dict_patch_to_params: dict,
        dict_trapregion_to_type: dict,
        init_junction_points_x,
        init_junction_points_y,
        dict_patch_to_x,
        dict_patch_to_y,
        dict_patch_to_z,
        junction_LtL,
        dict_region_to_junction_triangulation_vertices_idx,
        dict_region_to_internal_triangulation_vertices_idx,
        aligned_axis_pairs,
        orthogonal_axis_pairs,
        duplicated_vertices: list,
        maxiterations: int,
):
    def objectiveE(
            params,
            debug=False,
            return_grad=False,
    ):
        return objective_global_optimization(
            params=params,
            n_patch_patams=n_patches_params,
            dict_trapregion_to_type=dict_trapregion_to_type,
            init_dict_patch_to_params=dict_patch_to_params,
            n_patches=n_patches,
            aligned_axis_pairs=aligned_axis_pairs,
            orthogonal_axis_pairs=orthogonal_axis_pairs,
            init_junction_points_x=init_junction_points_x,
            init_junction_points_y=init_junction_points_y,
            jLtL=junction_LtL,
            dict_patch_to_x=dict_patch_to_x,
            dict_patch_to_y=dict_patch_to_y,
            dict_patch_to_z=dict_patch_to_z,
            dict_region_to_junction_triangulation_vertices_idx=dict_region_to_junction_triangulation_vertices_idx,
            dict_region_to_internal_triangulation_vertices_idx=dict_region_to_internal_triangulation_vertices_idx,
            duplicated_vertices=duplicated_vertices,
            debug=debug,
            return_grad=return_grad,
        )

    valx0, gradx0 = objectiveE(x0, debug=True, return_grad=True)
    print("x0 objective value: ", valx0)

    print("-- GRADIENT check")

    numerical_grad_energy = spo.approx_fprime(
        xk=x0,
        f=lambda x: objectiveE(
            params=x
        ),
        epsilon=1.5e-8,
    )

    print("num grad shape: ", numerical_grad_energy.shape)
    print("num grad params: ", numerical_grad_energy[:n_patches_params])

    print("mygrad shape: ", gradx0.shape)
    print("mygrad params: ", gradx0[:n_patches_params])

    print("diff grad params: ", grad_diff := numerical_grad_energy[:n_patches_params] - gradx0[:n_patches_params])
    print("max(grad_diff), argmax(grad_diff), grad_diff", np.max(grad_diff), np.argmax(grad_diff), grad_diff.shape)
    print("|| grad_diff || = ", np.linalg.norm(grad_diff))

    print("grad vertices: ", numerical_grad_energy[n_patches_params:n_patches_params+10])
    print("mygrad vertices: ", gradx0[n_patches_params:n_patches_params+10])

    print("max(diff grad points): ", np.max(grad_diff := numerical_grad_energy[n_patches_params:] - gradx0[n_patches_params:]))
    print("|| grad_diff || = ", np.linalg.norm(grad_diff))
    # if np.linalg.norm(grad_diff) > 0.0001:
    #     raise ("ERR")

    grad_diff = spo.check_grad(
        x0=x0,
        func=lambda x: objectiveE(x),
        grad=lambda x: objectiveE(x, return_grad=True)[1],
    )

    print("gradient error: ", grad_diff)
    # assert grad_diff < 0.01
    if grad_diff > 0.001:
        warnings.warn(f"GRADIENT ERROR: 0.001 < {grad_diff} < 0.01")
        # raise Exception(f"grad err {grad_diff}")

    print("--- OPTIMIZATION ITERATIONS ---")

    start = timer()

    result = spo.minimize(
        lambda x: objectiveE(x, return_grad=True),
        x0=x0,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": maxiterations,
            "maxfun": 1000000,
            "disp": True,
        }
    )
    end = timer()
    print(f"\n\n==================\n ----> Done global fit <----")
    print("Time elapsed: ", end - start)
    # print(result)
    print("solution primitives: ", result.x[:n_patches_params])
    print("best value: ", result.fun)

    valxfin = objectiveE(params=result.x, debug=True)
    print("x0 objective value: ", valx0)
    print("res objective value: ", valxfin)

    return result


def patches_global_optimization(
        n_patches: int,
        dict_patch_to_params: dict,
        triang_x: np.array,
        triang_y: np.array,
        triang_z: np.array,
        triang_refined_z: np.array,
        pure_junction_vertices: np.array,
        junction_laplacian: scipy.sparse.csr_matrix,
        dict_trapregion_to_type: dict,
        dict_patch_to_x: dict,
        dict_patch_to_y: dict,
        dict_patch_to_z: dict,
        dict_region_to_junction_triangulation_vertices_idx: dict,
        dict_region_to_internal_triangulation_vertices_idx: dict,
        pairs_of_duplicates: list,
        maxiterations=20,
):
    print("\n -> Constuct x0")

    x0 = np.array([])

    # fill x0 with primitive params
    for i_patch in range(2, n_patches):
        x0 = np.append(x0, values=dict_patch_to_params[i_patch])
    print("x0 patches params: ", len(x0), ", ", x0)
    n_patches_params = len(x0)
    # fill x0 with values on the boundary
    # junction_points_z = refined_pixelsZ[junction_points_mask]
    init_junction_points_x = triang_x[[pure_junction_vertices]]
    init_junction_points_y = triang_y[[pure_junction_vertices]]
    init_junction_points_z = triang_refined_z[[pure_junction_vertices]]
    n_junction_vertices = len(pure_junction_vertices)
    print("number of junction vertices: ", n_junction_vertices)
    x0 = np.append(x0, init_junction_points_x)
    x0 = np.append(x0, init_junction_points_y)
    x0 = np.append(x0, init_junction_points_z)
    print("x0 with junction points: ", len(x0))
    print("n_patches: ", n_patches)

    print("jL rowsum is 0? ", np.allclose(junction_laplacian.sum(axis=0), 0))
    print("jL colsum is 0? ", np.allclose(junction_laplacian.sum(axis=1), 0))
    junction_LtL = junction_laplacian.transpose().dot(junction_laplacian)

    init_patch_params = split_x0_to_patch_params(
        myx=x0,
        init_patch_to_params=dict_patch_to_params,
        n_p=n_patches,
    )

    init_patch_axis = patches_to_axis_dict(
        dict_patch_type=dict_trapregion_to_type,
        dict_patch_params=init_patch_params,
    )

    print("\n -> aligned / orthogonal pairs")

    aligned_axis_pairs, orthogonal_axis_pairs = find_aligned_ortho_pairs(
        patch_to_axis_dict=init_patch_axis,
        angle_threshold=0.25,
        patch_to_int_triang_vertices=dict_region_to_internal_triangulation_vertices_idx,
        patch_to_junct_triang_vertices=dict_region_to_junction_triangulation_vertices_idx,
    )

    print("Aligned pairs: ", aligned_axis_pairs)
    print("Orthogonal pairs: ", orthogonal_axis_pairs)

    duplicated_pairs = list()
    for i in range(len(pairs_of_duplicates) // 2):
        cut1 = pairs_of_duplicates[2 * i]
        cut2 = pairs_of_duplicates[2 * i + 1]
        assert len(cut1) == len(cut2)
        for j in range(len(cut1)):
            duplicated_pairs.append((cut1[j], cut2[j]))

    print("\n ------------> GLOBAL FIT")

    improved_result = run_global_optimization(
        x0=x0,
        n_patches=n_patches,
        n_patches_params=n_patches_params,
        dict_patch_to_params=dict_patch_to_params,
        dict_trapregion_to_type=dict_trapregion_to_type,
        init_junction_points_x=init_junction_points_x,
        init_junction_points_y=init_junction_points_y,
        dict_patch_to_x=dict_patch_to_x,
        dict_patch_to_y=dict_patch_to_y,
        dict_patch_to_z=dict_patch_to_z,
        junction_LtL=junction_LtL,
        dict_region_to_junction_triangulation_vertices_idx=dict_region_to_junction_triangulation_vertices_idx,
        dict_region_to_internal_triangulation_vertices_idx=dict_region_to_internal_triangulation_vertices_idx,
        aligned_axis_pairs=aligned_axis_pairs,
        orthogonal_axis_pairs=orthogonal_axis_pairs,
        duplicated_vertices=duplicated_pairs,
        maxiterations=maxiterations,
    )

    improved_junctions_x, improved_junctions_y, improved_junctions_z = split_x0_to_junction_xyz(
        myx=improved_result.x,
        number_patch_params=n_patches_params,
        number_junction_vertices=n_junction_vertices,
    )

    improved_patch_params = split_x0_to_patch_params(
        myx=improved_result.x,
        init_patch_to_params=dict_patch_to_params,
        n_p=n_patches,
    )

    print("n_params: ", n_patches_params)

    print(f"\n\n==================\n ----> Check if we have more aligned / ortho pairs <----")

    improved_patch_axis = patches_to_axis_dict(
        dict_patch_type=dict_trapregion_to_type,
        dict_patch_params=improved_patch_params,
    )

    new_aligned_axis_pairs, new_orthogonal_axis_pairs = find_aligned_ortho_pairs(
        patch_to_axis_dict=improved_patch_axis,
        angle_threshold=0.25,
        patch_to_junct_triang_vertices=dict_region_to_junction_triangulation_vertices_idx,
        patch_to_int_triang_vertices=dict_region_to_internal_triangulation_vertices_idx,
    )

    print("Aligned pairs: ", aligned_axis_pairs)
    print("NEW Aligned pairs: ", new_aligned_axis_pairs)
    print("Orthogonal pairs: ", orthogonal_axis_pairs)
    print("NEW Orthogonal pairs: ", new_orthogonal_axis_pairs)

    if (aligned_axis_pairs != new_aligned_axis_pairs) or (orthogonal_axis_pairs != new_orthogonal_axis_pairs):
        print("NEW AXIS FOUND")
        joint_aligned_pairs = sorted(list(set(aligned_axis_pairs).union(set(new_aligned_axis_pairs))))
        joint_ortho_pairs = sorted(list(set(orthogonal_axis_pairs).union(set(new_orthogonal_axis_pairs))))

        new_improved_result = run_global_optimization(
            x0=x0,
            n_patches=n_patches,
            n_patches_params=n_patches_params,
            dict_patch_to_params=dict_patch_to_params,
            dict_trapregion_to_type=dict_trapregion_to_type,
            init_junction_points_x=init_junction_points_x,
            init_junction_points_y=init_junction_points_y,
            dict_patch_to_x=dict_patch_to_x,
            dict_patch_to_y=dict_patch_to_y,
            dict_patch_to_z=dict_patch_to_z,
            junction_LtL=junction_LtL,
            dict_region_to_junction_triangulation_vertices_idx=dict_region_to_junction_triangulation_vertices_idx,
            dict_region_to_internal_triangulation_vertices_idx=dict_region_to_internal_triangulation_vertices_idx,
            aligned_axis_pairs=joint_aligned_pairs,
            orthogonal_axis_pairs=joint_ortho_pairs,
            duplicated_vertices=duplicated_pairs,
            maxiterations=maxiterations,
        )

        improved_junctions_x, improved_junctions_y, improved_junctions_z = split_x0_to_junction_xyz(
            myx=new_improved_result.x,
            number_patch_params=n_patches_params,
            number_junction_vertices=n_junction_vertices,
        )

        improved_patch_params = split_x0_to_patch_params(
            myx=new_improved_result.x,
            init_patch_to_params=dict_patch_to_params,
            n_p=n_patches,
        )

    return improved_junctions_x, improved_junctions_y, improved_junctions_z, improved_patch_params


if __name__ == "__main__":
    pngname = "3dsketching_chair"
    with open(f"results/{pngname}/global_opt_logs.pkl", "rb") as f:
        input_n_patches, \
        input_dict_patch_to_params, \
        input_triang_x, \
        input_triang_y, \
        input_triang_z, \
        input_triang_refined_z, \
        input_pure_junction_vertices, \
        input_dict_trapregion_to_type, \
        input_dict_patch_to_x, \
        input_dict_patch_to_y, \
        input_dict_patch_to_z, \
        input_dict_region_to_junction_triangulation_vertices_idx, \
        input_dict_region_to_internal_triangulation_vertices_idx, \
        flat_adjacency_junctions, \
        flat_laplacian_junctions, \
        pairs_of_duplicates = pickle.load(f)

    improved_junctions_x, improved_junctions_y, improved_junctions_z, improved_patch_params = \
        patches_global_optimization(
            n_patches=input_n_patches,
            dict_patch_to_params=input_dict_patch_to_params,
            triang_x=input_triang_x,
            triang_y=input_triang_y,
            triang_z=input_triang_z,
            triang_refined_z=input_triang_refined_z,
            pure_junction_vertices=input_pure_junction_vertices,
            junction_laplacian=flat_laplacian_junctions,
            dict_trapregion_to_type=input_dict_trapregion_to_type,
            dict_patch_to_x=input_dict_patch_to_x,
            dict_patch_to_y=input_dict_patch_to_y,
            dict_patch_to_z=input_dict_patch_to_z,
            dict_region_to_junction_triangulation_vertices_idx=input_dict_region_to_junction_triangulation_vertices_idx,
            dict_region_to_internal_triangulation_vertices_idx=input_dict_region_to_internal_triangulation_vertices_idx,
            pairs_of_duplicates=pairs_of_duplicates,
            maxiterations=100,
        )

    plt.figure()
    plt.scatter(improved_junctions_x, improved_junctions_y, color="white", edgecolors='k', linewidths=0.5, zorder=5, s=10)
    for i in range(len(pairs_of_duplicates) // 2):
        p1 = pairs_of_duplicates[2 * i]
        p2 = pairs_of_duplicates[2 * i + 1]
        for j in range(len(p1)):
            plt.plot([improved_junctions_x[p1[j]], improved_junctions_x[p2[j]]],
                     [improved_junctions_y[p1[j]], improved_junctions_y[p2[j]]],
                     color='red', zorder=6)
    plt.axis("equal")
    plt.savefig("reports/plot_with_duplicates.svg")
    plt.close()
