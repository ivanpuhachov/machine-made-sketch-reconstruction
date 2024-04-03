import pickle
import numpy as np
import igl


def process_item(
        pngname: str,
):
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
            boundary_vertices, \
            optBB, \
            optParams, \
            optParams_dict, \
            cylinder_plane_potential_cuts, \
            input_dict_trapregion_to_type = pickle.load(f)

    newX, newY = deform_triangle(
        faces=triangulation.faces,
        initX=input_triang_x,
        initY=input_triang_y,
        ind_b=np.arange(triangulation.n_svg_points),
        newX_b=optX,
        newY_b=optY,
    )

    init_verts = np.stack((input_triang_x, input_triang_y, np.ones_like(input_triang_y)), axis=1)
    new_verts = np.stack((newX, newY, np.ones_like(newX)), axis=1)
    igl.write_obj("reports/init.obj", init_verts, triangulation.faces)
    igl.write_obj("reports/new.obj", new_verts, triangulation.faces)


def deform_triangle(
        faces,
        initX,
        initY,
        ind_b,
        newX_b,
        newY_b,
):
    init_verts = np.stack((initX, initY, np.ones_like(initX)), axis=1)

    bc = np.stack((
        newX_b,
        newY_b,
        np.ones_like(newY_b)), axis=1)

    arap = igl.ARAP(init_verts, faces, 3, ind_b)
    vn = arap.solve(bc, init_verts)

    newX, newY = vn[:, 0], vn[:, 1]
    return newX, newY


if __name__ == "__main__":
    myname = "tun1"
    process_item(pngname=myname)