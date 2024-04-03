import numpy as np
import re


class EleFile:
    def __init__(
            self,
            triangles: np.array,
            attributes: np.array,
    ):
        """
        in triangles, vertices id starts from 1
        http://www.cs.cmu.edu/~quake/triangle.ele.html
        :param triangles: in np array (Nx3)
        """
        self.triangles = triangles
        self.n_elements = triangles.shape[0]
        self.n_nodes_per_triangle = triangles.shape[1]
        self.n_attributes = attributes.shape[1]
        self.attributes = attributes

    @classmethod
    def from_file(
            cls,
            somefile,
    ):
        with open(somefile) as f:
            lines = f.readlines()
            header = lines[0]
            header = re.sub(' +', ' ', header)
            if header[0] == " ":
                header = header[1:]
            hdspl = header.split(" ")
            n_elements = int(hdspl[0])
            n_nodes_per_triangle = int(hdspl[1])
            n_attributes = int(hdspl[2])
            triangles = np.zeros(shape=(n_elements, n_nodes_per_triangle), dtype=int)
            attributes = np.zeros(shape=(n_elements, n_attributes), dtype=float)
            elements_lines = lines[1:1+n_elements]
            for i in range(n_elements):
                l = elements_lines[i]
                l = re.sub(' +', ' ', l)
                if l[0]==" ":
                    l = l[1:]
                lspl = l.split(" ")
                for j in range(n_nodes_per_triangle):
                    triangles[i, j] = int(lspl[1+j])
                if n_attributes > 0:
                    for j in range(n_attributes):
                        attributes[i, j] = float(lspl[3+j])

        return cls(
            triangles=triangles,
            attributes=attributes,
        )

    def to_file(
        self,
        somefile,
    ):
        with open(somefile, 'w') as myfile:
            myfile.write(
                f"{self.triangles.shape[0]} {self.triangles.shape[1]} {self.attributes.shape[1]}\n"
            )
            for i in range(self.n_elements):
                attr = " ".join([str(self.attributes[i, j]) for j in range(self.n_attributes)])
                elem = " ".join([str(self.triangles[i, j]) for j in range(self.n_nodes_per_triangle)])
                myfile.write(
                    f"{i+1} {elem} {attr}\n"
                )


if __name__ == "__main__":
    ef = EleFile.from_file(somefile="Triangle/A.1.ele")
    print(ef.triangles)
    ef.to_file("test.txt")
