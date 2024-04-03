from PIL import Image, ImageOps
import numpy as np
from pathlib import Path


def read_image(
        im_path: Path,
        fixed_width=512,
):
    pil_image = Image.open(im_path)
    pil_image = ImageOps.autocontrast(pil_image.convert("RGB"))
    width, height = pil_image.size
    aspect_ratio = height / width
    if aspect_ratio <= 1:
        return scale_and_pad(pil_image, fixed_width=fixed_width)
    else:
        return np.copy(
            scale_and_pad(pil_image.transpose(Image.TRANSPOSE), fixed_width=fixed_width).transpose(1, 0, 2)
        )


def scale_and_pad(
        pil_image,
        fixed_width=512,
):
    width, height = pil_image.size
    aspect_ratio = height / width
    new_height = int(aspect_ratio * fixed_width)
    pad_height = (32 - new_height % 32) % 32
    resized_image = pil_image.resize((fixed_width, new_height), Image.LINEAR)  # this is removed in PIL 10, downgrade
    np_image = np.array(resized_image)  # this flips axis, height is shape[0] now
    if np.max(np_image) > 255:
        raise Exception(f"You are probably using 16bit image, image max: {np.max(np_image)}")
    if len(np_image.shape) == 2:
        np_image = np.repeat(np_image[:, :, np.newaxis], 3, axis=2)
    if np.max(np_image) > 2:
        np_image = 1.0 - np_image / 255.0
    np_image = np.pad(np_image, ((0, pad_height), (0, 0), (0, 0)), mode="constant", constant_values=np_image[0, 0, 0])
    if np_image.shape[0] > np_image.shape[1]:
        diff = np_image.shape[0] - np_image.shape[1]
        np_image = np.pad(np_image, ((0, 0), (0, diff), (0, 0)), mode="constant", constant_values=np_image[0, 0, 0])
        # diff1, diff2 = diff // 2, diff // 2 + diff % 2
        # np_image = np.pad(np_image, ((0, 0), (diff1, diff2), (0, 0)), mode="constant", constant_values=np_image[0, 0, 0])
    if np_image.shape[0] < np_image.shape[1]:
        diff = np_image.shape[1] - np_image.shape[0]
        np_image = np.pad(np_image, ((0, diff), (0, 0), (0, 0)), mode="constant",
                          constant_values=np_image[0, 0, 0])
        # diff1, diff2 = diff // 2, diff // 2 + diff % 2
        # np_image = np.pad(np_image, ((diff1, diff2), (0, 0), (0, 0)), mode="constant",
        #                   constant_values=np_image[0, 0, 0])
    if np_image.shape[2] == 4:
        np_image = np_image[..., :3]  # remove alpha channel
    return np_image
