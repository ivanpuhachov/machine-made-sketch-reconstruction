import numpy as np
import matplotlib.pyplot as plt
import skimage
from pathlib import Path

from MyMesh import MyMesh, MyMesh3D
from cone_fit import ConeFit, cone_alignment_energy2_vutheta, cone_alignment_energy2, cone_evalZ, plot_cone
from test_cylinder_fit import CylinderFit, cylinder_alignment_geomfitty_xyz, cylinder_evalZ, plot_cylinder, cylinder_estimate_r
from sphere_fit import SphereFit, sphere_evalZ, sphere_alignment_energy2
from plane_fit import fit_plane_euclidean, plane_evalZ, plane_eval_dist, plane_eval_dist2, plane_eval_distZ2, fit_plane_singlestep
from camera_transformations import get_camera_matrices, get_pixel_coordinates, transform_normals_global_to_camera
from refine_segmentation import refine_segmentation
from utils_fillmap import dict_patch_template, build_region_to_junctions_dicts


def log3d(
        background_mask,
        depth_camera,
        segmentation,
        normals_camera=None,
        name="test",
        saveto=Path("results/"),
):
    """
    given background mask logs the results with pixelwise triangulation
    :param normals_camera:
    :param background_mask: True when background
    :param depth_camera:
    :param segmentation:
    :param name:
    :return:
    """
    print(f"\n\n==================\n ----> log3d {name} <----")
    mm = MyMesh.from_background_mask_pixelwise(background_mask=background_mask)
    mm.reshuffle_triangulation_vertices()
    m3d = MyMesh3D.fromMyMesh2d(mm)
    m3d.setZ_from_depth(depth_camera)
    vertex_classes = mm.get_class_from_classimage(classimage=segmentation)
    mm.vertices_to_camera_coords(imsize=depth_camera.shape[0])
    m3d.vertices_to_camera_coords(imsize=depth_camera.shape[0])
    m3d.export_colored_ply(file_path=str(saveto / f"{name}.ply"), vertex_color_index=vertex_classes)
    # m3d.plot_html(name=f"results/{name}.html")


def report_values(
        img: np.array,
        depth: np.array,
        segmentation: np.array,
        normals: np.array,
        name="test",
):
    """
    does savefig for provided data
    :param img:
    :param depth:
    :param segmentation:
    :param normals:
    :param name:
    :return:
    """
    label_keys = [
        "Background", "Plane", "Cylinder", "Cone", "Sphere",
        "Torus", "Revolution", "Extrusion", "BSpline", "Other",
    ]

    plt.figure()
    plt.imshow(img, interpolation="nearest", cmap="gray_r")
    plt.colorbar()
    plt.title("image")
    plt.savefig(f"reports/{name}_sketch.png", bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(normals, interpolation="nearest")
    plt.title("normals")
    plt.savefig(f"reports/{name}_normals.png", bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(depth, interpolation="nearest")
    plt.title("depth prediction")
    plt.colorbar()
    plt.savefig(f"reports/{name}_depth_pred.png", bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(segmentation, cmap="tab10", interpolation="nearest", vmin=-0.5, vmax=9.5)
    cbar = plt.colorbar(ticks=np.arange(len(label_keys)))
    cbar.ax.set_yticklabels(label_keys)  # vertically oriented colorbar
    plt.axis("off")
    plt.title("segmentation")
    plt.savefig(f"reports/{name}_segmentation.png", bbox_inches="tight", dpi=150)
    plt.close()


def build_fillmap_correspondance(
        fillmap,
        gridvalues,
):
    dict_correspondance = dict()
    for i in range(2, np.max(fillmap) + 1):
        dict_correspondance[i] = gridvalues[fillmap == i]
    return dict_correspondance


def refine(
        image,
        depth,
        camera_normals,
        segmentation,
        plot=False,
        save3d=False,
        name="test",
        saveto=Path("results/"),
        savenpz=False,
):
    """

    :param saveto:
    :param savenpz:
    :param image: np array (h,w)
    :param depth: np array (h,w)
    :param camera_normals: np array CAMERA normals (h,w,3)
    :param segmentation: np array (h,w)
    :param plot:
    :param save3d:
    :param name:
    :return:
    """
    print(image.shape)
    print(depth.shape)
    print(segmentation.shape)
    assert image.shape == depth.shape == segmentation.shape
    refined_segm, dict_trapregion_to_type, fillmap, thin_fillmap = refine_segmentation(
        image=image,
        raw_segmentation=segmentation,
        plot=True,
        name=name,
        saveto=saveto,
    )

    pixels_to_camera_coordinates = get_pixel_coordinates(depth_values=depth)

    return do_local_fits(
        fillmap=fillmap,
        pixels_to_camera_coordinates=pixels_to_camera_coordinates,
        camera_normals=camera_normals,
        dict_trapregion_to_type=dict_trapregion_to_type,
        plot=plot,
        name=name,
        saveto=saveto,
    )


def do_local_fits(
        fillmap,
        pixels_to_camera_coordinates,
        camera_normals,
        dict_trapregion_to_type: dict,
        plot=True,
        name="local_fit_test",
        saveto=Path("reports/"),
):
    refined_depth = -pixels_to_camera_coordinates[..., 2].copy()

    pixelsX = pixels_to_camera_coordinates[..., 0]
    pixelsY = pixels_to_camera_coordinates[..., 1]
    pixelsZ = pixels_to_camera_coordinates[..., 2]
    refined_pixelsZ = np.copy(pixelsZ)

    print(np.unique(fillmap))
    n_patches = np.max(fillmap) + 1

    dict_patch_to_params = dict_patch_template(n_patches)  # contains params for each primitive fit
    dict_patch_to_x = build_fillmap_correspondance(fillmap, gridvalues=pixelsX)  # contain X coordinates for points in a patch
    dict_patch_to_y = build_fillmap_correspondance(fillmap, gridvalues=pixelsY)  # contain Y coordinates for points in a patch
    dict_patch_to_z = build_fillmap_correspondance(fillmap, gridvalues=pixelsZ)  # contain Z coordinates for points in a patch

    dict_patch_to_image_junction_mask, dict_patch_to_image_junction_idx = build_region_to_junctions_dicts(
        fillmap=fillmap,
        name=name,
        plot=plot,
    )
    print(" --- define processing order --- ")
    array_idx = np.arange(2, n_patches)
    array_types = np.array([dict_trapregion_to_type[x] for x in array_idx])
    list_planes = array_idx[array_types == "Plane"].tolist()
    list_cylinders = array_idx[array_types == "Cylinder"].tolist()
    list_spheres = array_idx[array_types == "Sphere"].tolist()
    list_cones = array_idx[array_types == "Cone"].tolist()
    list_others = list(set(array_idx) - set(list_planes) - set(list_cylinders) - set(list_spheres) - set(list_cones))
    regions_processing_order = list()
    regions_processing_order.extend(list_planes)
    regions_processing_order.extend(list_spheres)
    regions_processing_order.extend(list_cylinders)
    regions_processing_order.extend(list_cones)
    regions_processing_order.extend(list_others)
    print("new processing order: ", regions_processing_order)
    assert set(regions_processing_order) == set(range(2, n_patches))

    print("\n ------------> LOCAL PATCH FIT")

    axis_found_before = list()  # will contain a list of axis we found

    for processing_id in range(len(regions_processing_order)):
        regionidx = regions_processing_order[processing_id]
        print(f"\n\n==================\n ----> \tpatch {regionidx} {dict_trapregion_to_type[regionidx]}<----")
        region_type = dict_trapregion_to_type[regionidx]
        x, y = dict_patch_to_x[regionidx], dict_patch_to_y[regionidx]
        z = dict_patch_to_z[regionidx]
        normals0, normals1, normals2 = camera_normals[..., 0][fillmap == regionidx], \
                                       camera_normals[..., 1][fillmap == regionidx], \
                                       camera_normals[..., 2][fillmap == regionidx]
        nearby_junctions_mask = dict_patch_to_image_junction_mask[regionidx]
        jx = pixelsX[nearby_junctions_mask]
        jy = pixelsY[nearby_junctions_mask]
        jz = pixelsZ[nearby_junctions_mask]
        # log_pointcloud(
        #     points=np.stack((x, y, z), axis=1),
        #     normals=np.stack((normals0, normals1, normals2), axis=1),
        #     name=f"reports/patch_{regionidx}",
        #     offline=True,
        # )
        if region_type == "Plane":
            print(" plane - fitting depths")
            planeparams = fit_plane_singlestep(x, y, z)
            print("plane params: ", planeparams)
            planeparams = np.array(planeparams)
            planeparams = planeparams / np.linalg.norm(planeparams[:3])
            print("plane params after normalization: ", planeparams)
            # fit internal points
            newz = plane_evalZ(x, y, a=planeparams[0], b=planeparams[1], c=planeparams[2], d=planeparams[3])
            refined_depth[fillmap == regionidx] = - newz
            refined_pixelsZ[fillmap == regionidx] = newz
            dict_patch_to_params[regionidx] = planeparams
            # fit junction points
            newz = plane_evalZ(jx, jy, a=planeparams[0], b=planeparams[1], c=planeparams[2], d=planeparams[3])
            refined_pixelsZ[dict_patch_to_image_junction_mask[regionidx]] = newz
            refined_depth[dict_patch_to_image_junction_mask[regionidx]] = -newz
            axis_found_before.append(planeparams[:3])
        if region_type == "Cylinder":
            # Cylinder
            mypoints = np.vstack((x, y, z))
            mynormals = np.vstack((normals0, normals1, normals2))
            # mynormals = np.random.rand(3,50)
            # print("mypoints shape: ", mypoints.shape)
            # print("mynormals shape: ", mynormals.shape)
            cylfit = CylinderFit(points=mypoints, normals=mynormals)
            # cylfit.plot_data()
            # cylfit.plot_scatter()
            # cylfit.plot_normals()
            # C, W, R2 = cylfit.fit(plot=False)
            C, W, R2 = cylfit.pyomo_fit()
            energy2 = cylinder_alignment_geomfitty_xyz(
                    c=C,
                    w=W,
                    x=x,
                    y=y,
                    z=z,
                )
            print("Energy2 eval: ", energy2)

            minenergy = energy2
            minC, minW, minR2 = C, W, R2

            print("First attempt: c w r2", C, W, R2)

            for j_processed_before in range(0, processing_id):
                j_patch_before = regions_processing_order[j_processed_before]
                # here we try other patches to find a better axis approximation
                if dict_trapregion_to_type[j_patch_before] == "Plane":
                    thisother_plane_params = dict_patch_to_params[j_patch_before]
                    planeW = np.array([
                        thisother_plane_params[0],
                        thisother_plane_params[1],
                        thisother_plane_params[2]
                    ])
                    C, W, R2 = cylfit.pyomo_fit(estimateW=planeW)
                    energy2 = cylinder_alignment_geomfitty_xyz(c=C, w=W, x=x, y=y, z=z,)
                    if energy2 < minenergy:
                        print("found better axis: c w r2", C, W, R2)
                        minenergy = energy2
                        minC, minW, minR2 = C, W, R2
                # TODO: do the same with other patches that might be there
                if dict_trapregion_to_type[j_patch_before] == "Cylinder":
                    thisother_cyl_params = dict_patch_to_params[j_patch_before]
                    candidateW = np.array([
                        thisother_cyl_params[3],
                        thisother_cyl_params[4],
                        thisother_cyl_params[5]
                    ])
                    C, W, R2 = cylfit.pyomo_fit(estimateW=candidateW)
                    energy2 = cylinder_alignment_geomfitty_xyz(c=C, w=W, x=x, y=y, z=z, )
                    if energy2 < minenergy:
                        print("found better axis: c w r2", C, W, R2)
                        minenergy = energy2
                        minC, minW, minR2 = C, W, R2

            C, W, R2 = minC, minW, minR2
            print(f"for cylinder {regionidx} we settled on: c w r2", C, W, R2)
            dict_patch_to_params[regionidx] = np.array([C[0], C[1], C[2], W[0], W[1], W[2], R2])
            print("Cylinder params: ", C, W, R2)
            print("||W|| = ", np.linalg.norm(W))
            print("- projecting points")
            # fit internal points
            newz = cylinder_evalZ(x, y, z, c=C, w=W, r2=R2, debug=False)
            refined_depth[fillmap == regionidx] = - newz
            refined_pixelsZ[fillmap == regionidx] = newz
            # fit junction points
            newz = cylinder_evalZ(
                jx, jy, jz,
                c=C, w=W, r2=R2, debug=False,
            )
            axis_found_before.append(W)
            refined_pixelsZ[dict_patch_to_image_junction_mask[regionidx]] = newz
            refined_depth[dict_patch_to_image_junction_mask[regionidx]] = -newz

        if region_type == "Cone":
            continue
        #     # Cone
        #     # unstable. try commenting my_line_thinner to make it slightly better
            mypoints = np.vstack((x, y, z))
            mynormals = np.vstack((normals0, normals1, normals2))
            conefit = ConeFit(points=mypoints, normals=mynormals, method="lm")
            optV, optU, optTheta = conefit.pyomo_fit()
            minenergy = cone_alignment_energy2(v=optV, u=optU, theta=optTheta, points=mypoints, weight_norm_u=0)
            print("Cone init energy: ", minenergy)
            for aa in axis_found_before:
                print("Cone: try axis ", aa)
                coneV, coneU, coneTheta = conefit.pyomo_fit(estimateU=aa)
                energy = cone_alignment_energy2(v=coneV, u=coneU, theta=coneTheta, points=mypoints, weight_norm_u=0)
                print("This attempt energy: ", energy)
                if energy < minenergy:
                    minenergy = energy
                    optV, optU, optTheta = coneV, coneU, coneTheta
                    print("CONE - gotcha")

            energy2 = cone_alignment_energy2(v=coneV, u=coneU, theta=coneTheta, points=mypoints, weight_norm_u=0)
            print("Energy eval: ", energy)
            print("Energy2 eval: ", energy2)
            dict_patch_to_params[regionidx] = np.array([coneV[0], coneV[1], coneV[2], coneU[0], coneU[1], coneU[2], coneTheta])
            print("Cone params: ", optV, optU, optTheta)
            print("||U|| = ", np.linalg.norm(coneU))
            print("- projecting points")
            newz = cone_evalZ(x, y, z, v=coneV, u=coneU, theta=coneTheta, debug=False)
        #
        #     # fig = plt.figure()
        #     # ax = plt.axes(projection='3d')
        #     # plt.title("Cylinder fit")
        #     # # ax.scatter3D(data[0, :], data[1, :], data[2, :], c="green")
        #     # ax.scatter3D(coneV[0], coneV[1], coneV[2], c='blue', marker="^", label="trueV")
        #     # ax.plot3D(
        #     #     [coneV[0], (coneV + coneU)[0]],
        #     #     [coneV[1], (coneV + coneU)[1]],
        #     #     [coneV[2], (coneV + coneU)[2]],
        #     #     c="green", label="trueU"
        #     # )
        #     # ax.scatter3D(x, y, z, c='black', marker="*", label="p")
        #     # ax.scatter3D(x, y, newz, c='red', marker="*", label="p")
        #     #
        #     # plot_cone(ax, v=coneV, u=coneU, theta=coneTheta, h0=0.5, h1=2.5)
        #     #
        #     # ax.set_xlabel('X')
        #     # ax.set_ylabel('Y')
        #     # ax.set_zlabel('Z')
        #     #
        #     # ax.set_aspect('equal')
        #     # plt.legend()
        #     # plt.show()
        #
            refined_depth[fillmap == regionidx] = - newz
            refined_pixelsZ[fillmap == regionidx] = newz
        #     # fit junction points
        #     newz = cone_evalZ(
        #         jx, jy, jz,
        #         v=coneV, u=coneU, theta=coneTheta, debug=False,
        #     )
        #     refined_pixelsZ[dict_patch_to_image_junction_mask[regionidx]] = newz
        #     refined_depth[dict_patch_to_image_junction_mask[regionidx]] = -newz

        if region_type == "Sphere":
            # Sphere
            mypoints = np.vstack((x, y, z))
            mynormals = np.vstack((normals0, normals1, normals2))
            spherefit = SphereFit(points=mypoints, normals=mynormals)
            # sphereC, sphereR2 = spherefit.fit(plot=False, show=True)
            sphereC, sphereR2 = spherefit.pyomo_fit()
            dict_patch_to_params[regionidx] = np.array(
                [sphereC[0], sphereC[1], sphereC[2], sphereR2]
            )
            newz = sphere_evalZ(x, y, z, c=sphereC, r2=sphereR2)
            refined_depth[fillmap == regionidx] = -newz
            refined_pixelsZ[fillmap == regionidx] = newz
            # fit junction points
            newz = sphere_evalZ(
                jx, jy, jz,
                c=sphereC, r2=sphereR2,
            )
            refined_pixelsZ[dict_patch_to_image_junction_mask[regionidx]] = newz
            refined_depth[dict_patch_to_image_junction_mask[regionidx]] = -newz

    # fillmap = my_line_thinner(fillmap)

    # if plot:
    #     report_values(
    #         img=image,
    #         depth=refined_depth,
    #         segmentation=segmentation,
    #         normals=camera_normals,
    #         name=f"{name}_refined"
    #     )

    # log3d(
    #     background_mask=fillmap == 1,
    #     depth_camera=refined_depth,
    #     segmentation=fillmap,
    #     name=f"{name}_pixel_refined",
    #     saveto=saveto,
    # )

    print(f"\n\n==================\n ----> Done local fit <----")

    return fillmap, refined_pixelsZ, dict_trapregion_to_type, dict_patch_to_params


def abc_experiment():
    np.set_printoptions(precision=3)
    # for i in [3,6,7,32,58,66,78,136,171,249]:
    #     refine(itemidx=i, itemangle=288)
    itemidx = 6
    itemangle = 288
    gt_data = np.load(f"gt_data/{itemidx}_{itemangle}_01.npz")
    pred_depth_data = np.load(f"data/{itemidx}_{itemangle}_01_depth.npz")
    pred_depth = pred_depth_data["depth"]
    pred_segmentation_data = np.load(f"data/{itemidx}_{itemangle}_01_segm.npz")
    pred_segmentation = pred_segmentation_data["classes"]
    pred_normals_data = np.load(f"data/{itemidx}_{itemangle}_01_norms.npz")
    pred_normals = pred_normals_data["normals"]

    gtimage = np.pad(gt_data["sketch"], ((8, 8), (8, 8)), constant_values=gt_data["sketch"][0, 0])
    gtdepth = np.pad(gt_data["depth"], ((8, 8), (8, 8)), constant_values=gt_data["depth"][0, 0])
    gtnormals = np.pad(gt_data["normals"], ((8, 8), (8, 8), (0, 0)), mode="wrap")
    gtsegmentation = np.pad(gt_data["classes"], ((8, 8), (8, 8)), constant_values=gt_data["classes"][0, 0])

    gtimage = skimage.transform.rescale(gtimage, scale=0.5, order=3)
    print("gtimage.shape: ", gtimage.shape)
    pred_depth = skimage.transform.rescale(pred_depth, scale=0.5, order=0)
    print("pred_depth.shape: ", pred_depth.shape)
    gtdepth = skimage.transform.rescale(gtdepth, scale=0.5, order=0)
    print("gtdepth.shape: ", gtdepth.shape)
    gtnormals = skimage.transform.rescale(gtnormals, scale=0.5, order=0, channel_axis=2)  # TODO: fix
    print("gtnormals.shape: ", gtnormals.shape)
    pred_normals = skimage.transform.rescale(pred_normals, scale=0.5, order=0, channel_axis=2)
    print("pred_normals.shape: ", pred_normals.shape)
    # gtsegmentation = skimage.transform.rescale(gtsegmentation, scale=0.5, order=0)
    # print(gtsegmentation.shape)
    # gtimage = gtimage[::2, ::2]
    # gtdepth = gtdepth[::2, ::2]
    # gtnormals = gtnormals[::2, ::2]
    gtsegmentation = gtsegmentation[::2, ::2]
    print("gtsegmentation.shape: ", gtsegmentation.shape)
    pred_segmentation = pred_segmentation[::2, ::2]
    print("pred_segmentation.shape: ", pred_segmentation.shape)
    # depth = depth[::2, ::2]

    log3d(
        background_mask=gtsegmentation == 0,
        depth_camera=gtdepth,
        segmentation=gtsegmentation,
        name=f"{itemidx}_{itemangle}_pixel_gt",
    )

    log3d(
        background_mask=gtsegmentation == 0,
        depth_camera=pred_depth,
        segmentation=pred_segmentation,
        normals_camera=transform_normals_global_to_camera(global_normals=gtnormals),
        name=f"{itemidx}_{itemangle}_pixel_predicted"
    )

    refine(
        image=gtimage,
        segmentation=gtsegmentation,
        depth=pred_depth,
        camera_normals=transform_normals_global_to_camera(global_normals=pred_normals),
        plot=True,
        save3d=True,
        name=f"{itemidx}_{itemangle}",
    )


def png_experiment():
    pngname = "Long_Machine_Elements_015_1"
    workfolder = Path(f"results/{pngname}/")
    pred_depth_data = np.load(str(workfolder / f"{pngname}_depth.npz"))
    pred_depth = pred_depth_data["depth"]
    pred_segmentation_data = np.load(str(workfolder / f"{pngname}_refined_segm.npz"))
    pred_segmentation = pred_segmentation_data["classes"]
    pred_normals_data = np.load(str(workfolder / f"{pngname}_norms.npz"))
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

    report_values(
        img=gtimage,
        depth=pred_depth,
        segmentation=pred_segmentation,
        normals=pred_normals,
        name=f"{pngname}_pred"
    )

    log3d(
        background_mask=pred_segmentation == 0,
        depth_camera=pred_depth,
        segmentation=pred_segmentation,
        normals_camera=transform_normals_global_to_camera(global_normals=pred_normals),
        name=f"{pngname}_pixel_predicted"
    )

    refine(
        image=gtimage,
        segmentation=pred_segmentation,
        depth=pred_depth,
        camera_normals=transform_normals_global_to_camera(global_normals=pred_normals),
        plot=True,
        save3d=True,
        name=pngname,
        saveto=Path(f"results/{pngname}/"),
    )


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    # abc_experiment()
    png_experiment()
