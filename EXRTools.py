import numpy as np
import OpenEXR
import Imath
import json
from pathlib import Path
import matplotlib.pyplot as plt


class MyEXR:
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_file(cls, path_to_file: Path):
        # print(path_to_file)
        pixeltypeFloat = Imath.PixelType(Imath.PixelType.FLOAT)
        file = OpenEXR.InputFile(str(path_to_file))
        dw = file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        # print(file.header())
        img = np.zeros((size[1], size[0], 3))
        cnames = ['R', 'G', 'B']

        for i in range(len(cnames)):
            channelname = cnames[i]
            channelstr = file.channel(channelname, pixeltypeFloat)
            # print(channelstr)
            inp = np.frombuffer(channelstr, dtype=np.float32)
            inp.shape = size[1], size[0]
            img[..., i] = inp
        return cls(data=img)


class ExrDepth(MyEXR):
    def __init__(self, data):
        super(ExrDepth, self).__init__(data=data)
        self.data = self.data[..., 0]
    def clear_background(self, thr=100, value=np.nan):
        self.data[self.data > thr] = value


class ExrNormals(MyEXR):
    def __init__(self, data):
        super(ExrNormals, self).__init__(data=data)

    def get_rgb_img(self):
        return (self.data + 1) * 0.5


class ExrMaterials(MyEXR):
    def __init__(self, data):
        super(ExrMaterials, self).__init__(data=data)
        self.data = self.data[..., 0].astype('int8')
        self.materials_list = ["Background", "Plane", "Cylinder", "Cone", "Sphere", "Torus", "Revolution", "Extrusion",
                               "BSpline", "Other"]
        self.materials_dict = {self.materials_list[i]: i for i in range(len(self.materials_list))}

    def get_mask(self, material_name):
        assert material_name in self.materials_list
        return self.data == self.materials_dict[material_name]


if __name__ == "__main__":
    with open('settings.json', 'r') as f:
        settings = json.load(f)
    exrd = ExrMaterials.from_file(Path(settings['renders_folder']) / "00000004" / "exr_material_072_01.exr")
    print("---------")
    print(exrd.data.shape)
    plt.imshow(exrd.data, cmap='tab10', vmin=0, vmax=10)
    plt.colorbar()
    plt.show()

    plt.imshow(exrd.get_mask("Background"), cmap='tab10', vmin=0, vmax=10)
    plt.colorbar()
    plt.show()
