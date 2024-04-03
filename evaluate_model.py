from models import TestModel, SegmentationModel
from dataitem import RenderItem

import numpy as np
import torch
import argparse
import json
from pathlib import Path
import os
import matplotlib.pyplot as plt
from img_utils import read_image
from PIL import Image


def build_predicted_item(
        inp_image,
        preds,
        do_depth=True,
        do_normals=True,
        do_segmentation=True,
        upscale_to=512,
) -> RenderItem:
    item_to_save = RenderItem.empty_item(side=inp_image.shape[0])
    upsampling_linear = torch.nn.Upsample(size=(upscale_to, upscale_to), mode='bilinear', align_corners=True)
    item_to_save.sketch = upsampling_linear(inp_image).cpu().numpy()[0, 0]
    if do_depth:
        item_to_save.depth_image = preds['depth'].cpu().numpy()[0, 0]
        # item_to_save.depth_image = upsampling_linear(preds['depth']).cpu().numpy()[0, 0]
    if do_normals:
        item_to_save.normals = preds['normals'].cpu().numpy()[0]
        # item_to_save.normals = upsampling_linear(
        #     preds['normals'].cpu().permute(0, 3, 1, 2)
        # ).numpy()[0].transpose(1,2,0)
    if do_segmentation:
        # item_to_save.classes = preds['segmentation_class'].cpu().numpy()[0]
        classes_image = preds['segmentation_class'].cpu().numpy()[0]
        img_nn_pil = np.array(Image.fromarray(classes_image.astype(float)).resize((upscale_to, upscale_to), Image.NEAREST)).astype(int)
        item_to_save.classes = img_nn_pil
        # for i in range(len(labelkeys := item_to_save.get_label_keys())):
            # item_to_save.labels[labelkeys[i]] = preds['segmentation_masks'].cpu().numpy()[0][i]
            # item_to_save.labels[labelkeys[i]] = upsampling_nearest(preds['segmentation_masks']).cpu().numpy()[0][i]
    return item_to_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval model")
    parser.add_argument("--image", type=str, default="short_benchmark/ecoffee105a.png", help="path to input image")
    parser.add_argument("--out", type=str, default="item.npz", help="path to input image")
    # parser.add_argument("--image", type=str, default="svgs/6_freestyle_288_01.png", help="path to input image")
    parser.add_argument("--npz", type=str, default=None, help="path to input npz dataitem")
    parser.add_argument('--depth', default=False, action="store_true")
    parser.add_argument('--backdepth', default=False, action="store_true")
    parser.add_argument('--normals', default=False, action="store_true")
    parser.add_argument('--segm', default=False, action="store_true")
    parser.add_argument("--saveto", default="reports/", type=str)
    parser.add_argument("--plots", default=True, action="store_true")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/segm_02_14_top_epoch_35-valid_dataset_iou=0.9972253.ckpt",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not (args.depth or args.normals or args.segm or args.backdepth):
        print("no action specified, use --depth, --normals or --segm")

    myinp = torch.zeros((1, 3, 256, 256)).to(device)

    inputname = "latest"

    if args.npz is not None:
        assert os.path.exists(args.npz)
        inputname = Path(args.npz).stem
        data = np.load(args.npz)
        image = np.pad(data["sketch"], ((8, 8), (8, 8)), constant_values=data["sketch"][0, 0])
        image = np.repeat(image[np.newaxis, :, :], 3, axis=0)
        myinp = torch.from_numpy(image).float().unsqueeze(0).to(device)

    if args.image is not None:
        assert os.path.exists(args.image)
        inputname = Path(args.image).stem
        fixed_width = 416 if args.segm else 512
        img = read_image(im_path=args.image, fixed_width=fixed_width)[..., 0]
        img_repeated = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        myinp = torch.from_numpy(img_repeated.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        img1024 = read_image(im_path=args.image, fixed_width=1024)
        img512 = read_image(im_path=args.image, fixed_width=512)
        plt.imsave(fname=Path(args.saveto).parents[0] / f"{inputname}_resized1024.png", arr=img1024[..., 0],
                   cmap="gray_r")
        plt.imsave(fname=Path(args.saveto).parents[0] / f"{inputname}_resized512.png", arr=img512[..., 0],
                   cmap="gray_r")

    print(img.shape)

    if args.depth or args.normals or args.backdepth:
        mdl = TestModel.load_from_checkpoint(checkpoint_path=args.ckpt).to(device)
    if args.segm:
        mdl = SegmentationModel.load_from_checkpoint(checkpoint_path=args.ckpt).to(device)

    reportsdir = Path(args.saveto)

    mdl.eval()
    with torch.no_grad():
        preds = mdl.eval_predict(myinp)

    preditem = build_predicted_item(
        inp_image=myinp.cpu(),
        preds=preds,
        do_depth=args.depth or args.backdepth,
        do_normals=args.normals,
        do_segmentation=args.segm,
        upscale_to=512,
    )

    preditem.savenpz(path_to_save=Path(f"{args.out}"))

    # preditem.save_depth_to_ply(path_to_ply=Path(f"depth{args.saveto}"))

    if args.plots:
        plt.figure()
        plt.imshow(myinp.cpu().numpy()[0, 0], interpolation='nearest')
        plt.title(f"input tensor: {myinp.shape}")
        plt.colorbar()
        plt.savefig(reportsdir / f"{inputname}_input.png", dpi=150)
        plt.close()

        if args.depth:
            plt.figure()
            plt.imshow(preditem.depth_image, interpolation='nearest')
            plt.title("depth")
            plt.colorbar()
            plt.savefig(reportsdir / f"{inputname}_depth.png")
            plt.close()

            plt.imsave(reportsdir / f"{inputname}_depth_raw.png", preditem.depth_image)

        if args.backdepth:
            plt.figure()
            plt.imshow(preditem.depth_image, interpolation='nearest')
            plt.title("depth")
            plt.colorbar()
            plt.savefig(reportsdir / f"{inputname}_backdepth.png")
            plt.close()

            plt.imsave(reportsdir / f"{inputname}_backdepth_raw.png", preditem.depth_image)

        if args.normals:
            plt.figure()
            plt.imshow(preditem.normals, interpolation='nearest')
            plt.title("normals")
            plt.savefig(reportsdir / f"{inputname}_normals.png")
            plt.close()

            # plt.imsave(reportsdir / f"{inputname}_normals_raw.png", preditem.normals)

        if args.segm:
            plt.figure()
            plt.imshow(preditem.classes, interpolation='nearest', cmap="tab10", vmin=-0.5, vmax=9.5)
            plt.title("classes")
            plt.colorbar()
            plt.savefig(reportsdir / f"{inputname}_segm.png")
            plt.close()

            plt.imsave(reportsdir / f"{inputname}_segm_raw.png", preditem.classes, cmap="tab10", vmin=-0.5, vmax=9.5)
