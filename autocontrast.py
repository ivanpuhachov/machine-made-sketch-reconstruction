from PIL import Image, ImageOps
from pathlib import Path
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autocontrast")
    parser.add_argument("--input", type=str, default=None, help="path to input image")
    parser.add_argument("--output", type=str, default=None, help="save output to")
    args = parser.parse_args()

    im_path = Path(args.input)
    assert os.path.exists(im_path)
    out_path = Path(args.output)

    pil_image = Image.open(im_path)
    pil_image = ImageOps.autocontrast(pil_image.convert("RGB"))
    pil_image.save(out_path)