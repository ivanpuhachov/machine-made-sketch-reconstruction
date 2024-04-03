import numpy as np
import matplotlib.pyplot as plt


def dict_patch_template(
        n_patches: int,
):
    return dict({i: np.array([], dtype=int) for i in range(n_patches)})


def build_region_to_junctions_dicts(
        fillmap,
        name="test",
        plot=False,
):
    """

    :param fillmap:
    :return:
    """
    pixelsU, pixelsV = np.indices(dimensions=fillmap.shape, dtype=int)
    n_patches = np.max(fillmap) + 1

    dict_patch_to_x = dict_patch_template(n_patches)
    dict_patch_to_y = dict_patch_template(n_patches)
    dict_patch_to_z = dict_patch_template(n_patches)  # this dict contains predicted z values (z = - depth)

    junction_points_mask = fillmap == 0
    n_junction_points = np.sum(junction_points_mask)
    junction_points_u = pixelsU[junction_points_mask]
    junction_points_v = pixelsV[junction_points_mask]
    print("Total junction pixels: ", n_junction_points)

    dict_junction_point_to_patch = dict()  # for each junction point store list of neighbor patches
    dict_patch_to_junction_idx = dict_patch_template(n_patches)  # for each patch store np array of idx of junction points
    dict_patch_to_junction_mask = dict({i: np.zeros_like(fillmap, dtype=bool) for i in range(n_patches+1)})
    for i_point in range(len(junction_points_u)):
        pu, pv = junction_points_u[i_point], junction_points_v[i_point]
        window_size = 2
        window = fillmap[pu - window_size:pu + window_size, pv - window_size:pv + window_size]
        uniques = set(np.unique(window))
        uniques.discard(0)
        uniques.discard(1)
        if len(uniques) == 0:
            # print("Empty neighbors!")
            window_size = 4
            window = fillmap[pu - window_size:pu + window_size, pv - window_size:pv + window_size]
            uniques = set(np.unique(window))
            uniques.discard(0)
            uniques.discard(1)
            if len(uniques) == 0:
                print("Super empty neighbors!")
                window_size = 10
                window = fillmap[pu - window_size:pu + window_size, pv - window_size:pv + window_size]
                uniques = set(np.unique(window))
                uniques.discard(0)
                uniques.discard(1)
        dict_junction_point_to_patch[i_point] = list(uniques)

        for i_patch in iter(uniques):
            dict_patch_to_junction_idx[i_patch] = np.append(dict_patch_to_junction_idx[i_patch], i_point)
            dict_patch_to_junction_mask[i_patch][pu, pv] = True

    # if plot:
    #     plt.figure()
    #     plt.imshow(fillmap, cmap="tab20b", interpolation="nearest", vmin=-0.5, vmax=19.5)
    #     plt.title("trapped ball regions")
    #     plt.colorbar()
    #     plt.savefig(f"reports/{name}_trappedball.png", bbox_inches="tight", dpi=150)
    #     plt.close()
    #
    #     for i in range(2, n_patches):
    #         plt.figure()
    #         plt.imshow(dict_patch_to_junction_mask[i])
    #         plt.title(f"junctions {i}")
    #         plt.colorbar()
    #         plt.savefig(f"reports/{name}_junctions_{i}.png", bbox_inches="tight", dpi=150)
    #         plt.close()

    return dict_patch_to_junction_mask, dict_patch_to_junction_idx
