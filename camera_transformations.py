import numpy as np
import matplotlib.pyplot as plt


def plot_normals(normals, suptitle="", show=False):
    plt.figure(figsize=(13, 3))
    plt.suptitle(suptitle)

    plt.subplot(141)
    plt.imshow(normals)

    plt.subplot(142)
    plt.title("channel 0")
    plt.imshow(normals[..., 0], cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar()

    plt.subplot(143)
    plt.title("channel 1")
    plt.imshow(normals[..., 1], cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar()

    plt.subplot(144)
    plt.title("channel 2")
    plt.imshow(normals[..., 2], cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar()

    if show:
        plt.show()


def get_camera_matrices(
        input_shape,
        ortho_param=3.5,
):
    """
    useful matrices for coordinates manipulation. Orthographic camera.
    camera_matrix_world -> X_world = M . X_camera
    camera_matrix_pixel -> X_image = P . X_camera
    :param input_shape: shape of
    :return:
    """
    camera_matrix_world = np.array(
        [
            [-1, 0, 0, 0],
            [0, 0, 1, 5],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]
    )  # this matrix translates camera coordinates to world coordinates X_world = M . X_camera

    res_x, res_y = input_shape
    camera_matrix_pixel = np.array(
        [
            [res_x / ortho_param, 0, res_x / 2],
            [0, -res_y / ortho_param, res_y / 2],
            [0, 0, 1]
        ]
    )  # this matrix translates camera coordinates to pixel coordinates X_image = M . X_camera
    return camera_matrix_world, camera_matrix_pixel


def transform_normals_global_to_camera(global_normals):
    """
    transforms global normals to camera normals
    :param global_normals: np array (h,w,3)
    :return:
    """
    camera_matrix_world, camera_matrix_pixel = get_camera_matrices(
        input_shape=(global_normals.shape[0], global_normals.shape[1])
    )
    normals_camera = np.tensordot(
        camera_matrix_world[:3, :3].transpose(),
        global_normals.transpose(),
        axes=([1], [0]),
    ).transpose()
    return normals_camera


def get_pixel_coordinates(
        depth_values=None,
        shape=None,
):
    """
    returns a grid with pixels in camera coordinates. Inputs used only to determine appropriate shape
    :param depth_values:
    :param shape:
    :return:
    """
    assert (shape is not None) or (depth_values is not None)
    pixels_shape = shape
    if depth_values is not None:
        pixels_shape = depth_values.shape[:2]
    U, V = np.indices(dimensions=pixels_shape, dtype=float)
    camera_matrix_world, camera_matrix_pixel = get_camera_matrices(
        input_shape=pixels_shape
    )
    pixel_coords = np.stack((U, V, np.ones_like(U))).transpose()
    camera_matrix_pixel_inv = np.linalg.inv(camera_matrix_pixel)
    pixels_to_camera_coordinates = np.tensordot(
        camera_matrix_pixel_inv,
        pixel_coords.transpose(),
        axes=([1], [0]),
    ).transpose()
    if depth_values is not None:
        pixels_to_camera_coordinates[..., 2] = - depth_values
    return pixels_to_camera_coordinates


def coordinates_to_pixels(
        orig_image,
        coordsX, coordsY,
):
    """
    given camera coordinates, returns coordinates in pixel space of original image
    :param depth_values:
    :param coordsX:
    :param coordsY:
    :return:
    """
    pixels_shape = orig_image.shape[:2]
    camera_matrix_world, camera_matrix_pixel = get_camera_matrices(
        input_shape=pixels_shape
    )
    camera_coords = np.stack(
        (coordsX, coordsY, np.ones_like(coordsX)), axis=1,
    )
    pixels_coords = np.tensordot(
        camera_matrix_pixel,
        camera_coords.transpose(),
        axes=([1], [0]),
    ).transpose()
    return pixels_coords[:, :2]


if __name__ == "__main__":
    gt_data = np.load(f"gt_data/Sphere.npz")
    gt_depth = np.pad(gt_data["depth"], ((8, 8), (8, 8)), mode="wrap")
    gt_normals = np.pad(gt_data["normals"], ((8, 8), (8, 8), (0, 0)), mode="wrap")

    pixels = get_pixel_coordinates(depth_values=gt_depth)

    plt.figure(figsize=(7, 2))
    plt.subplot(131)
    plt.imshow(pixels[..., 0], interpolation="nearest", cmap="coolwarm")
    plt.title("X")
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(pixels[..., 1], interpolation="nearest", cmap="coolwarm")
    plt.title("Y")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(pixels[..., 2], interpolation="nearest", cmap="coolwarm")
    plt.title("Z")
    plt.colorbar()
    plt.suptitle("pixels")
    plt.show()

    print("gt depth shape: ", gt_depth.shape)
    print("gt normals shape: ", gt_normals.shape)

    gt_normals_camera = transform_normals_global_to_camera(global_normals=gt_normals)

    plot_normals(gt_normals, suptitle="GT normals", show=True)
    plot_normals(gt_normals_camera, suptitle="GT normals to camera coords", show=True)

    pixelsX = pixels[..., 0]
    pixelsY = pixels[..., 1]
    pixelsZ = pixels[..., 2]

    selectedX = pixelsX[200, :]
    yidx = [100 + np.abs(200 - x) for x in range(len(selectedX))]
    selectedY = pixelsY[yidx, 200]

    print(yidx)
    print(selectedX)
    print(selectedY)

    coords = coordinates_to_pixels(
        orig_image=gt_depth,
        coordsX=selectedX,
        coordsY=selectedY,
    )

    print(coords.shape)

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.scatter(selectedX, selectedY, c=range(len(selectedX)))
    plt.title("some points in camera coords")
    plt.axis("equal")

    plt.subplot(122)
    plt.scatter(coords[:, 0], coords[:, 1], c=range(len(selectedX)))
    plt.title("these points in pixel space")
    plt.axis("equal")
    plt.show()

