from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from EXRTools import ExrNormals, ExrDepth, ExrMaterials
import json


class RenderItem:
    """
    Contains a single rendering output: freestyle (sketch) image, surface type labels
    """
    @staticmethod
    def get_class_image_from_masks(label_masks: dict) -> np.array:
        classes = np.zeros(shape=(label_masks["Background"].shape[0], label_masks["Background"].shape[1]), dtype=int)
        for i, name in enumerate(RenderItem.get_label_keys()):
            if name in label_masks.keys():
                classes[label_masks[name]] = i
        return classes

    @staticmethod
    def get_label_masks_from_class_image(class_image) -> dict:
        assert len(class_image.shape) == 2
        label_masks = dict()
        for i, name in enumerate(RenderItem.get_label_keys()):
            label_masks[name] = class_image == i
        return label_masks

    @staticmethod
    def loadimage(path_to_image=Path("renders/00000171/albedo_0000001.png")):
        img = np.array(Image.open(path_to_image))
        # print(img.shape)
        # img = np.pad(img, ((8,8),(8,8), (0,0)))
        # print(img.shape)
        # print(img.dtype)
        # if len(img.shape)==4:
        #     img[img[:,:,3]==0] *=0
        return img

    @staticmethod
    def get_label_keys():
        return [
            "Background", "Plane", "Cylinder", "Cone", "Sphere",
            "Torus", "Revolution", "Extrusion", "BSpline", "Other",
        ]

    @staticmethod
    def get_depth_image_from_depth(
            d,
            closest=3,
            farthest=7,
    ):
        """
        This method takes depth from range (closest, farthest) and maps it to (0,1) where 0 is farthest (background)
        """
        k = farthest - closest
        newd = 1 - (d - closest) / k  # WARNING: these constants are used in loss and post-processing!
        return newd

    @staticmethod
    def get_depth_from_depth_image(
            di,
            closest=3,
            farthest=7,
    ):
        """
        Inverse from get_depth_image_from_depth. This method maps (0,1) to true depth values.
        """
        k = farthest - closest
        d = (1 - di) * k + closest
        # TODO: this assertion fails on items from predicitons
        # assert (RenderItem.get_depth_image_from_depth(d) == di).all()
        return d

    def __init__(
            self,
            sketch: np.array,
            normals: np.array,
            classes: np.array,
            labels: dict,
            depth_image: np.array,
            lines: np.array = None,
    ):
        """
        sketch: (416, 416)
        normals: (416, 416, 3)
        classes: (416, 416)
        label Background: (416, 416)
        label Plane: (416, 416)
        label Cone: (416, 416)
        label Cylinder: (416, 416)
        label BSpline: (416, 416)
        label Revolution: (416, 416)
        label Sphere: (416, 416)
        label Torus: (416, 416)
        label Extrusion: (416, 416)
        label Other: (416, 416)
        lines: (416, 416)
        depth_image: (416, 416)
        normals_image: (416, 416, 3)
        Depth image shape:  (416, 416)
        """
        self.sketch = sketch
        self.normals = normals
        # print(normals.shape)
        assert normals.shape[-1] == 3
        self.classes = classes
        self.labels = labels
        self.depth_image = depth_image

        if lines is None:
            self.lines = sketch
        else:
            self.lines = lines

        self.normals_image = (self.normals + 1) * 0.5  # transforming to RGB range
        self.label_keys = self.get_label_keys()
        self.n_classes = len(self.label_keys)

    @classmethod
    def from_exr_files(
            cls,
            path_to_freestyle=Path("renders/00000877/freestyle_108_01.png"),
            path_to_drawing=Path("renders/00000877/drawing_108_01.png"),
            path_to_label=Path("renders/00000877/exr_material_108_01.exr"),
            path_to_normal=Path("renders/00000877/exr_normal_108_01.exr"),
            path_to_depth=Path("renders/00000877/exr_depth_108_01.exr"),
            path_to_shading=Path("renders/00000877/render_108"),
            use_shading=False,
            use_drawing=False,
    ):
        sketch = RenderItem.loadimage(path_to_image=path_to_freestyle)[:, :, 3].astype("float")
        sketch /= 255.0
        lines = sketch

        if use_drawing:
            sketch = RenderItem.loadimage(path_to_image=path_to_drawing)[...,0].astype("float")
            sketch = 1 - (sketch / 255)

        exr_normals = ExrNormals.from_file(path_to_file=path_to_normal)
        # TODO: depth normalization as preprocessing
        exr_depth = ExrDepth.from_file(path_to_file=path_to_depth)
        exr_depth.clear_background(thr=7, value=7)
        # doing depth normalization when reading, 7 -> 0 (7 is background, see above)
        exr_materials = ExrMaterials.from_file(path_to_file=path_to_label)

        labels = dict()
        label_keys = RenderItem.get_label_keys()
        for k in label_keys:
            labels[k] = exr_materials.get_mask(material_name=k)

        if use_shading:
            pngpath = str(path_to_freestyle).replace("freestyle", "render")[:-7]
            shade = RenderItem.loadimage(path_to_image=path_to_shading)[:, :, 0].astype("float")
            shade /= 255.0
            shade = 1 - shade
            shade *= 1 - labels["Background"]

            sketch = np.maximum(
                shade,
                sketch,
            )

        return cls(
            sketch=sketch,
            lines=lines,
            normals=exr_normals.data,
            classes=exr_materials.data,
            labels=labels,
            depth_image=cls.get_depth_image_from_depth(exr_depth.data),
        )

    @classmethod
    def empty_item(cls, side=400):
        sketch = np.zeros(shape=(side, side))
        depth_image = np.zeros(shape=(side, side))
        # TODO: check that default normals are ok (pointed to camera)
        normals = np.stack(
            (
                np.zeros(shape=(side, side)),
                - np.ones(shape=(side, side)),
                np.zeros(shape=(side, side)),
            ), axis=2,
        )
        # print(normals.shape)
        classes = np.ones(shape=(side, side))  # this way all pixels will be of class "Plane"
        return cls(
            sketch=sketch,
            normals=normals,
            classes=classes,
            labels=cls.get_label_masks_from_class_image(class_image=classes),
            depth_image=depth_image,
        )

    @classmethod
    def default_item(cls):
        idx = 78
        angle = 216
        foldername = f"{idx:08}"
        with open('settings.json', 'r') as f:
            settings = json.load(f)
        return RenderItem.from_freestyle_exr(
            path_to_freestyle=Path(f"{settings['renders_folder']}/{foldername}/freestyle_{angle:03}_01.png")
        )

    def savenpz(self, path_to_save: Path, save_labels=False):
        moreargs = self.labels if save_labels else dict()
        np.savez_compressed(
            path_to_save,
            sketch=self.sketch,
            normals=self.normals,
            depth=self.get_depth_from_depth_image(self.depth_image),
            lines=self.lines,
            classes=self.classes,
            **moreargs,
        )

    def print_dims(self):
        print(f"sketch: {self.sketch.shape}")
        print(f"normals: {self.normals.shape}")
        print(f"classes: {self.classes.shape}")
        for k in self.get_label_keys():
            print(f"label {k}: {self.labels[k].shape}")
        print(f"lines: {self.lines.shape}")
        print(f"depth_image: {self.depth_image.shape}")
        print(f"normals_image: {self.normals_image.shape}")

    def pad(self, pad_w, pad_h):
        self.sketch = np.pad(self.sketch, ((pad_w, pad_w), (pad_h, pad_h)), constant_values=self.sketch[0, 0])
        self.normals = np.pad(self.normals, ((pad_w, pad_w), (pad_h, pad_h), (0, 0)), constant_values=0)
        self.classes = np.pad(self.classes, ((pad_w, pad_w), (pad_h, pad_h)), constant_values=self.classes[0, 0])
        for k in self.get_label_keys():
            self.labels[k] = np.pad(
                self.labels[k],
                ((pad_w, pad_w), (pad_h, pad_h)), constant_values=self.labels[k][0, 0]
            )
        self.depth_image = np.pad(self.depth_image, ((pad_w, pad_w), (pad_h, pad_h)),
                                  constant_values=self.depth_image[0, 0])

        self.normals_image = (self.normals + 1) * 0.5  # transforming to RGB range

    def save_depth_to_ply(self, path_to_ply: Path):
        # PLY file specs http://paulbourke.net/dataformats/ply/
        with open(path_to_ply, 'w') as f:
            lines_to_write = [
                f"ply\nformat ascii 1.0\ncomment object: depth to PLY mesh\n",
                f"element vertex {self.depth_image.shape[0] * self.depth_image.shape[1]}",
                f"\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\n"
                f"property uchar green\nproperty uchar blue\n",
                f"element face {(self.depth_image.shape[0] - 1) * (self.depth_image.shape[1] - 1)}\n",
                f"property list uchar int vertex_index\nend_header\n"
            ]
            maxdim = 40
            scaling_coeff = maxdim / self.depth_image.shape[0]
            for i in range(self.depth_image.shape[0]):
                for j in range(self.depth_image.shape[1]):
                    lines_to_write.append(
                        f"{scaling_coeff * i} {self.depth_image[i, j] * maxdim} {scaling_coeff * j} 100 100 0\n")
            n_faces_counter = 0
            for i in range(self.depth_image.shape[0] - 1):
                for j in range(self.depth_image.shape[1] - 1):
                    """
                    a - b
                    |   |
                    c - d
                    clock-wise
                    """
                    is_foreground = [
                        self.labels["Background"][i, j] < 0.5,
                        self.labels["Background"][i, j + 1] < 0.5,
                        self.labels["Background"][i + 1, j] < 0.5,
                        self.labels["Background"][i + 1, j + 1] < 0.5,
                    ]
                    if sum(is_foreground) < 3:
                        continue
                    n_faces_counter += 1
                    a = i * self.depth_image.shape[1] + j
                    b = a + 1
                    c = (i + 1) * self.depth_image.shape[1] + j
                    d = c + 1
                    # f.write(f"3 {a} {d} {c}\n")
                    # f.write(f"3 {a} {b} {d}\n")
                    lines_to_write.append(
                        f"{sum(is_foreground)} {a if is_foreground[0] else ''} {b if is_foreground[1] else ''} {d if is_foreground[3] else ''} {c if is_foreground[2] else ''}\n")
            lines_to_write[3] = f"element face {n_faces_counter}\n"
            f.writelines(lines_to_write)

    @classmethod
    def from_npz(cls, path_to_load: Path):
        data = np.load(path_to_load)
        return cls(
            sketch=data['sketch'],
            normals=data['normals'],
            classes=data['classes'],
            depth_image=cls.get_depth_image_from_depth(data['depth']),
            labels=cls.get_label_masks_from_class_image(data['classes']),
            lines=data["lines"]
        )

    @classmethod
    def from_freestyle_exr(
            cls,
            path_to_freestyle=Path("renders/00000877/freestyle_108_01.png"),
            use_shading=False,
            use_drawing=False,
    ):
        pf = str(path_to_freestyle)
        pdrw = str(path_to_freestyle).replace("freestyle_", "drawing_")
        pl = str(path_to_freestyle).replace("freestyle_", "exr_material_").replace(".png", ".exr")
        pn = str(path_to_freestyle).replace("freestyle_", "exr_normal_").replace(".png", ".exr")
        pd = str(path_to_freestyle).replace("freestyle_", "exr_depth_").replace(".png", ".exr")
        ps = pf
        if use_shading:
            ps = str(path_to_freestyle).replace("freestyle", "render")[:-7]
        return RenderItem.from_exr_files(
            path_to_freestyle=Path(pf),
            path_to_drawing=Path(pdrw),
            path_to_label=Path(pl),
            path_to_normal=Path(pn),
            path_to_depth=Path(pd),
            path_to_shading=Path(ps),
            use_shading=use_shading,
            use_drawing=use_drawing,
        )

    @staticmethod
    def depth_to_normals(depth_values: np.array):
        """
        This method converts camera depth values to normals in global cs
        """
        blender_ortho_camera_scale = 3.1
        original_image_resolution = 400
        # 3.1 == blender ortho camera param, 400 == original image resolution
        d0, d1 = np.gradient(depth_values, blender_ortho_camera_scale / original_image_resolution, edge_order=2)
        normal = np.dstack((-d1, np.ones_like(depth_values), -d0))
        # print("normal: ", normal.shape)
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n
        # normal += 1
        # normal /= 2
        return normal

    def normals_from_depth(self):
        return self.depth_to_normals(depth_values=-4 * self.depth_image)

    def stack_label_masks(self):
        masks = [self.labels[k].astype(np.float32) for k in self.get_label_keys()]
        stacked_masks = np.stack(masks, axis=0).astype('float')
        return stacked_masks

    def label_masks_to_classes(self):
        stacked_masks = self.stack_label_masks()
        # print(stacked_masks.shape)
        classes = np.argmax(stacked_masks, axis=0)
        return classes


class PredictedItem(RenderItem):
    def __init__(self, inps, preds, depth=None, normals=None):
        """
        inps: sketch; np.array; HxW
        preds: class prediction probabilities; np.array; n_classes x H x W
        """
        temp = RenderItem.default_item()
        super().__init__(sketch=temp.sketch, normals=temp.normals, classes=temp.classes, labels=temp.labels,
                         depth_image=temp.depth_image,)
        self.sketch = inps
        self.n_classes = preds.shape[0]
        self.labels = {self.label_keys[i]: preds[i] for i in range(self.n_classes)}
        self.classes = self.label_masks_to_classes()
        if depth is not None:
            self.depth_image = depth
        if normals is not None:
            assert normals.shape[-1] == 3
            self.normals_image = normals

    @classmethod
    def from_class_image(cls, inps, pred_mask, max_classes=9):
        """
        inps: sketch; np.array; HxW
        pred_mask: image with classes; np.array; HXW
        """
        preds = np.zeros(shape=(max_classes, inps.shape[0], inps.shape[1]), dtype=int)
        for i in range(max_classes):
            preds[i] = (pred_mask == i) * 1
        return cls(input, preds)

    @classmethod
    def from_pred_dict(cls, inps, preds):
        segm_preds = preds['segmentation_masks'].detach().cpu().squeeze(0).numpy()
        depth = preds['depth'].detach().cpu().squeeze(0).squeeze(0).numpy()
        normals = preds['normals'].detach().cpu().squeeze(0).transpose(1,2,0).numpy()
        return cls(inps, segm_preds, depth, normals)

    @classmethod
    def loadnpz(cls, path_to_load: Path):
        data = np.load(path_to_load)
        defaultitem = RenderItem.default_item()
        defaultitem.sketch = data['sketch']
        defaultitem.normals_image = data['normals_image']
        defaultitem.depth_image = data['depth_image']
        for k in defaultitem.get_label_keys():
            defaultitem.labels[k] = data[k]
        defaultitem.classes = np.zeros(shape=(defaultitem.sketch.shape[0], defaultitem.sketch.shape[1]), dtype=int)
        # for i, name in enumerate(defaultitem.label_keys):
        #     defaultitem.classes[defaultitem.labels[name]] = i
        return defaultitem


def getItemByID(
        idx=78,
        angle=108,
        use_shading: bool = None,
        use_drawing: bool = None,
):
    foldername = f"{idx:08}"
    with open('settings.json', 'r') as f:
        settings = json.load(f)
    if (use_shading is not None) or (use_drawing is not None):
        ri = RenderItem.from_freestyle_exr(
            path_to_freestyle=Path(f"{settings['renders_folder']}/{foldername}/freestyle_{angle:03}_01.png"),
            use_shading=use_shading,
            use_drawing=use_drawing,
        )
        return ri
    ri = RenderItem.from_npz(
        path_to_load=Path(f"{settings['renders_folder']}/{foldername}/{angle:03}_01.npz")
    )
    return ri


def getESBItem(
        foldername="Bearing_Blocks_002",
        use_shading=False,
):
    with open('settings.json', 'r') as f:
        settings = json.load(f)

    ri = RenderItem.from_freestyle_exr(
        path_to_freestyle=Path(f"{settings['ESB_renders_folder']}/{foldername}/freestyle_324_00.png"),
        use_shading=use_shading,
    )
    return ri


if __name__ == "__main__":
    # renderitembasictest(407)
    # renderitembasictest(7)
    # renderitembasictest(78)
    ri = getItemByID(idx=78, angle=216, use_shading=True)
    ri = RenderItem.from_freestyle_exr(
        path_to_freestyle="/home/ivan/projects/abcblendersketch/renders/ESB/Long_Pins_009/freestyle_324_00.png",
        use_shading=False,
    )
    # ri.save_depth_to_ply("item_reports/Long_Pins_009.ply")
    ri.print_dims()

