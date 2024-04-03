import pathlib

import matplotlib.pyplot as plt
import numpy as np
import scipy
import igl
from MyMesh import MyMesh, MyMesh3D
import cvxpy as cp
from myTriangulation import myTraingulation


def inflate(
        svg_file,
        use_this_depth_image,
        use_this_class_image,
        show_plt=False,
        name="test",
):
    triang, svg_points, svg_edges = myTraingulation.from_svg(
        path_to_svg=svg_file,
        svg_sampling_distance=10,
        triang_flags='qpa10',
    )

    vertex_depth_values = triang.interpolate_f_on_vertices(f_grid=use_this_depth_image)
    print(vertex_depth_values)

    mm = MyMesh(
        vertices=triang.vertices,
        vertex_markers=triang.vertex_markers,
        faces=triang.faces,
        holes=triang.holes,
    )

    original_vertex_classes = mm.get_class_from_classimage(classimage=use_this_class_image)


    mm.print_dims()
    mm.vertices = np.hstack(
        (
            # triang.get_vertices_to_camera_coords(imshape=(416, 416)),
            mm.vertices,
            4 - vertex_depth_values.reshape(-1, 1),
            original_vertex_classes.reshape(-1, 1)
        )
    )
    mm.reshuffle_triangulation_vertices()
    z_values = np.copy(mm.vertices[:, 2])
    class_values = np.copy(mm.vertices[:, 3].astype(int))
    mm.vertices = mm.vertices[:, :2]
    mm.vertices_to_camera_coords(imsize=use_this_depth_image.shape[0])

    return inflate_2d_mesh(
        vertices=mm.vertices[:, :2],
        vertex_markers=mm.vertex_markers,
        faces=mm.faces,
        holes=mm.holes,
        vertex_z_values=z_values,
        vertex_classes=class_values,
        show_plt=show_plt,
        name=name,
    )


def inflate_2d_mesh(
        vertices2d: np.array,
        vertex2d_markers: np.array,
        faces2d: np.array,
        holes: np.array,
        vertex_z_values: np.array,
        vertex_classes: np.array,
        show_plt=False,
        name="test",
        saveto=pathlib.Path("results/"),
):
    print(" == LOADING DATA == ")

    mm = MyMesh(
        vertices=vertices2d,
        vertex_markers=vertex2d_markers,
        faces=faces2d,
        holes=holes,
    )

    plt.scatter(mm.vertices[:, 0], mm.vertices[:, 1], c=np.arange(mm.vertices.shape[0]))
    plt.title("scatter plot of vertices")
    plt.colorbar()
    if show_plt:
        plt.show()
    else:
        plt.close()

    plt.scatter(mm.vertices[:, 0], mm.vertices[:, 1], c=vertex_classes, cmap="tab10", vmin=-0.5, vmax=9.5)
    plt.title("scatter plot of vertices")
    plt.colorbar()
    if show_plt:
        plt.show()
    else:
        plt.close()

    m3d = MyMesh3D.fromMyMesh2d(mm)
    m3d.vertices[:, 2] = vertex_z_values

    mm.reshuffle_triangulation_vertices()
    m3d.reshuffle_triangulation_vertices()

    # m3d.plot_html("enflate_2.html")

    print(" == NORMALS == ")
    EV, FE, EF = igl.edge_topology(v=mm.vertices, f=mm.faces)

    boundary_edges = []
    boundary_edges_midpoint = []
    boundary_faces = []
    boundary_normals = []
    for edge in range(len(EF)):
        e_to_f = EF[edge]
        if min(e_to_f) == -1:
    #         print(f"edge {i} is at boundary: {e_to_f}")
            boundary_faces.append(max(e_to_f))
            boundary_edges.append(edge)
            edge_vertices = EV[edge]
            edge_vector = mm.vertices[edge_vertices[0]] - mm.vertices[edge_vertices[1]]
            edge_normal = np.array([-edge_vector[1], edge_vector[0]])
            edge_normal /= np.linalg.norm(edge_normal)
            boundary_normals.append(edge_normal)
            boundary_edges_midpoint.append(0.5*(mm.vertices[edge_vertices[0]] + mm.vertices[edge_vertices[1]]))

    n_faces = len(mm.faces)
    n_boundary_faces = len(boundary_faces)
    n_boundary_edges = len(boundary_edges)
    assert n_boundary_faces == len(boundary_edges)
    n_vertices = len(mm.vertices)
    n_boundary_vertices = np.sum(mm.vertex_markers)
    n_internal_vertices = n_vertices - n_boundary_vertices
    print(mm.vertex_markers[:n_boundary_vertices])
    assert np.sum(mm.vertex_markers[:n_boundary_vertices]) == n_boundary_faces

    print(f"n vertices: {n_vertices}")
    print(f"n boundary vertices: {n_boundary_vertices}")
    print(f"n boundary edges: {n_boundary_edges}")
    print(f"n faces: {n_faces}")
    print(f"n boundary faces: {n_boundary_faces}")

    print(" == PLOTTING NORMALS ==")
    if n_boundary_faces < 100:
        set_of_boundary_faces = set(boundary_faces)
        plt.figure(figsize=(6,6))

        # plt.scatter(mm.vertices[:,0], mm.vertices[:,1], c=np.arange(len(mm.vertices)),zorder=5)

        for i_f in range(len(mm.faces)):
            f = mm.faces[i_f]
            a, b, c = mm.vertices[f[0]], mm.vertices[f[1]], mm.vertices[f[2]]
            color = 'k'

            if i_f in set_of_boundary_faces:
                color = 'red'
                i_in_boundary_list = boundary_faces.index(i_f)
                edge = boundary_edges[i_in_boundary_list]
                normal = boundary_normals[i_in_boundary_list]
                midpoint = boundary_edges_midpoint[i_in_boundary_list]
                plt.plot([midpoint[0], midpoint[0]+normal[0]], [midpoint[1], midpoint[1]+normal[1]], c='blue')

            plt.plot([a[0],b[0]], [a[1],b[1]], color=color, zorder=2, linewidth=0.5)
            plt.plot([a[0],c[0]], [a[1],c[1]], color=color, zorder=2, linewidth=0.5)
            plt.plot([c[0],b[0]], [c[1],b[1]], color=color, zorder=2, linewidth=0.5)
        plt.axis("equal")
        plt.savefig("normals.svg")

        plt.show()

    print(" == COMPUTING FEOM MATRICES ==")
    if len(mm.vertices.shape) == 2:
        mm.vertices = np.hstack((
            mm.vertices,
            np.zeros((mm.vertices.shape[0],1))
        ))

    L = igl.cotmatrix(mm.vertices, mm.faces)
    print(f"L shape: {L.shape}")
    print("L has NaN: ", np.isnan(np.sum(L)))

    M = igl.massmatrix(mm.vertices, mm.faces)
    Minv = scipy.sparse.diags(1 / M.diagonal())
    print(f"M shape: {M.shape}")

    LtML = L.transpose() @ (Minv @ L)

    G = igl.grad(mm.vertices, mm.faces)
    print("grad shape: ", G.shape)
    print("flat vertices: ", mm.vertices.shape)
    print("flat faces: ", mm.faces.shape)
    print(f"boundary faces: {n_boundary_faces}")

    m3d.make_back_surface()
    # m3d.plot_html(shading={"wireframe": True, "colormap": "PuRd"})

    print(" == CONSTRUCTING OPTIMIZATION PROBLEM ==")

    temp_data = np.ones(n_boundary_vertices)
    temp_row = np.arange(n_boundary_vertices)
    temp_col = np.arange(n_boundary_vertices)
    known_points_selector = scipy.sparse.csr_matrix(
        (temp_data, (temp_row, temp_col)),
        shape=(n_boundary_vertices, n_vertices)
    )
    known_points_solution = m3d.vertices[:n_boundary_vertices, 2]


    temp_row = np.arange(2*n_boundary_faces) // 2
    temp_col = np.arange(2*n_boundary_faces)
    temp_col[0::2] = boundary_faces
    temp_col[1::2] = boundary_faces
    temp_col[1::2] += len(mm.faces)

    temp_data = np.array(boundary_normals).flatten()

    boundary_faces_dot_n_selector = scipy.sparse.csr_matrix(
        (temp_data, (temp_row, temp_col)),
        shape=(n_boundary_faces, 3*mm.faces.shape[0])
    )
    print(boundary_faces_dot_n_selector.nnz)

    normals_bc_rhs = - boundary_faces_dot_n_selector @ G @ m3d.vertices[:n_vertices,2]
    lam = 0.1
    cc = 0.0001

    x = cp.Variable(n_boundary_vertices + n_internal_vertices)
    prob = cp.Problem(
        cp.Minimize(
            cp.sum_squares((L@x - M @ np.ones(n_boundary_vertices + n_internal_vertices) * cc)) + lam * cp.sum_squares(boundary_faces_dot_n_selector @ G @ x - normals_bc_rhs)
        ),
        [
            known_points_selector @ x == known_points_solution,
    #         boundary_faces_dot_n_selector @ G @ x == - boundary_faces_dot_n_selector @ G @ m3d.vertices[:n_vertices,2],
            x[n_boundary_vertices:] <= m3d.vertices[n_boundary_vertices:n_vertices,2]
        ]
    )
    prob.solve()

    m3d.vertices[n_vertices:, 2] = x.value[n_boundary_vertices:]
    # m3d.plot_html(name=str(saveto / f"{name}.html"))
    m3d.export_obj(saveto / f"{name}.obj", front_surface_count=n_faces)
    m3d.export_colored_ply(saveto / f"{name}.ply", vertex_color_index=vertex_classes)

    return m3d


if __name__ == "__main__":
    itemidx = 407
    itemangle = 288
    depth_prediction_data = np.load(f"data/{itemidx}_{itemangle}_01_depth.npz")
    segm_prediction_data = np.load(f"data/{itemidx}_{itemangle}_01_segm.npz")

    use_this_depth_image = depth_prediction_data["depth"][8:-8, 8:-8]
    use_this_class_image = segm_prediction_data["classes"][8:-8, 8:-8]

    inflate(
        use_this_class_image=use_this_class_image,
        use_this_depth_image=use_this_depth_image,
        svg_file=f"clean_svgs/{itemidx}_freestyle_{itemangle}_01_vector.svg",
        name=f"a2_refined"
    )
