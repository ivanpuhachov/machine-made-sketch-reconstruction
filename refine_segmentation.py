import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import argparse
from pathlib import Path
import warnings

from util_trappedball_fill import my_trapped_ball, my_line_thinner


def plot_segmentation_image(
        segm_image: np.array,
):
    label_keys = [
        "Background", "Plane", "Cylinder", "Cone", "Sphere",
        "Torus", "Revolution", "Extrusion", "BSpline", "Other",
    ]
    plt.imshow(segm_image, cmap="tab10", interpolation="nearest", vmin=-0.5, vmax=9.5)
    cbar = plt.colorbar(ticks=np.arange(len(label_keys)))
    cbar.ax.set_yticklabels(label_keys)  # vertically oriented colorbar
    plt.axis("off")


def report_segmentation_image(
        segm_image: np.array,
        title="some segmentation",
        reportsdir=Path("reports/")
):
    plt.figure()
    plot_segmentation_image(segm_image=segm_image)
    plt.title(title)
    plt.savefig(reportsdir / f"{title.lower().replace(' ', '_')}.png", bbox_inches="tight", dpi=150)
    plt.close()


def refine_segmentation_data(
        pred_segm_data,
        plot=False,
        savenpz=False,
        saveto="data",
        name="test",
):
    image = pred_segm_data["sketch"]
    pred_segm = pred_segm_data["classes"]
    return refine_segmentation(
        image,
        raw_segmentation=pred_segm,
        plot=plot,
        savenpz=savenpz,
        name=name,
        saveto=saveto,
    )


def refine_segmentation(
        image,
        raw_segmentation,
        plot=False,
        savenpz=False,
        name="test",
        saveto="data",
        pixel_threshold=200,
):
    print("\n\n --- REFINE SEGMENTATION ---")
    label_keys = [
        "Background", "Plane", "Cylinder", "Cone", "Sphere",
        "Torus", "Revolution", "Extrusion", "BSpline", "Other",
    ]
    workdir = Path(saveto)
    reportsdir = workdir / "reports"

    image_for_trapped_ball = ((1 - image) * 255).astype(np.uint8)
    fillmap = my_trapped_ball(image_for_trapped_ball, first_ball=4)

    thin_fillmap = my_line_thinner(fillmap)

    refined_segm = np.copy(raw_segmentation)

    print(np.unique(fillmap))

    # raw_segmentation = np.ones_like(raw_segmentation)
    # refined_segm[fillmap == 1] = 0

    dict_region_to_patch_type = dict()

    for regionidx in range(2, np.max(fillmap)+1):
        print(f"\n\n==================\n ----> \tpatch {regionidx} <----")
        # skip 0 (lines) and 1 (background)
        u, count = np.unique(
            raw_segmentation[
                fillmap == regionidx
            ].flatten(),
            return_counts=True)
        mode = u[np.argsort(-count)]
        # mode = scipy.stats.mode(
        #     raw_segmentation[
        #         fillmap == regionidx
        #     ].flatten(), keepdims=False)
        most_popular_mark = mode[0]
        total_pixels = np.sum(fillmap == regionidx)
        print(f"Most popular mark on {total_pixels} pixels: {label_keys[most_popular_mark]} ({most_popular_mark})")
        if total_pixels < pixel_threshold:
            warnings.warn(f"Region {regionidx} has only {total_pixels} pixels -> change label to Other from {label_keys[most_popular_mark]}")
            most_popular_mark = 9
        if most_popular_mark == 0:
            if len(mode) > 1:
                second_popular = mode[1]
                warnings.warn(
                    f"Region {regionidx} has only is classified as {label_keys[most_popular_mark]}, 2nd most popular is {label_keys[second_popular]}")
                most_popular_mark = second_popular
        refined_segm[
            thin_fillmap == regionidx
            ] = most_popular_mark
        dict_region_to_patch_type[regionidx] = label_keys[most_popular_mark]
    # refined_segm[(fillmap == 0) & (refined_segm == 0)] = 1

    if plot:
        plt.figure()
        plt.imshow(image, interpolation="nearest", cmap="gray_r")
        plt.colorbar()
        plt.title("image")
        plt.savefig(reportsdir / f"{name}_image.png", bbox_inches="tight", dpi=150)
        plt.close()

        plt.figure()
        plt.imshow(fillmap, cmap="tab20b", interpolation="nearest", vmin=-0.5, vmax=19.5)
        plt.title("trapped ball regions")
        plt.colorbar(ticks=np.arange(20))
        plt.savefig(reportsdir / f"{name}_trappedball.png", bbox_inches="tight", dpi=150)
        plt.close()

        plt.figure()
        plt.imshow(thin_fillmap, cmap="tab20b", interpolation="nearest", vmin=-0.5, vmax=19.5)
        plt.title("trapped ball regions thin")
        plt.colorbar(ticks=np.arange(20))
        plt.savefig(reportsdir / f"{name}_trappedball_thin.png", bbox_inches="tight", dpi=150)
        plt.close()

        report_segmentation_image(
            segm_image=raw_segmentation,
            title="predicted segmentation",
            reportsdir=reportsdir,
        )

        report_segmentation_image(
            segm_image=refined_segm,
            title="refined segmentation",
            reportsdir=reportsdir,
        )

    if savenpz:
        print(workdir / "npz" / f"{name}_refined_segm.npz")

        np.savez_compressed(
            workdir / "npz" / f"{name}_refined_segm.npz",
            sketch=image,
            classes=refined_segm,
        )
    return refined_segm, dict_region_to_patch_type, fillmap, thin_fillmap


if __name__ == "__main__":
    # np.set_printoptions(precision=3)
    # pred_segmentation_data = np.load("data/predicted_segmentation_6_288.npz")
    # # pred_segmentation_data = np.load("data/Pulley_Like_Parts_007_1.npz")
    # refine_segmentation_data(pred_segmentation_data, plot=True)
    parser = argparse.ArgumentParser(description="Eval model")
    parser.add_argument("--input", type=str, help="path to input segmentation npz file")
    parser.add_argument("--output", type=str, default=None, help="path to output folder")

    args = parser.parse_args()

    predicted_segmentation_data = np.load(args.input)
    if args.output is None:
        outpath = Path(args.input).parents[1]  # assume input path is like "results / name / npz / name_segm.npz"
    else:
        outpath = args.output
    name = Path(args.input).stem.replace("_segm", "")

    refine_segmentation_data(
        pred_segm_data=predicted_segmentation_data,
        name=name,
        savenpz=True,
        plot=True,
        saveto=str(outpath),
    )
