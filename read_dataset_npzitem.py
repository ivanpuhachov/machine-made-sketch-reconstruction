import numpy as np


def get_label_masks_from_class_image(class_image) -> dict:
    assert len(class_image.shape) == 2
    label_masks = dict()
    label_keys = [
            "Background", "Plane", "Cylinder", "Cone", "Sphere",
            "Torus", "Revolution", "Extrusion", "BSpline", "Other",
    ]
    for i, name in enumerate(label_keys):
        label_masks[name] = class_image == i
    return label_masks
