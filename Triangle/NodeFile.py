import numpy as np
import re


class NodeFile:
    def __init__(
            self,
            vertices,
            attributes,
            boundary_markers,
    ):
        """
        http://www.cs.cmu.edu/~quake/triangle.node.html
        :param vertices:
        :param attributes:
        :param boundary_markers:
        """
        self.vertices = vertices
        self.n_vertices = vertices.shape[0]
        self.attributes = attributes
        self.n_attributes = attributes.shape[1]
        self.boundary_markers = boundary_markers
        self.n_boundary_markers = boundary_markers.shape[1]

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
            n_vertices = int(hdspl[0])
            n_dims = int(hdspl[1])
            n_attributes = int(hdspl[2])
            n_boundary_markers = int(hdspl[3])
            vertices = np.zeros(shape=(n_vertices, n_dims), dtype=float)
            attributes = np.zeros(shape=(n_vertices, n_attributes), dtype=float)
            boundary_markers = np.zeros(shape=(n_vertices, n_boundary_markers), dtype=int)
            node_lines = lines[1:1+n_vertices]
            for i in range(n_vertices):
                l = node_lines[i]
                l = re.sub(' +', ' ', l)
                if l[0] == " ":
                    l = l[1:]
                lspl = l.split(" ")
                for j in range(n_dims):
                    vertices[i, j] = float(lspl[1+j])
                if n_attributes > 0:
                    for j in range(n_attributes):
                        attributes[i, j] = float(lspl[1+n_dims+j])
                if n_boundary_markers > 0:
                    boundary_markers[i, 0] = int(lspl[1+n_dims+n_attributes])
            return cls(
                vertices=vertices,
                attributes=attributes,
                boundary_markers=boundary_markers,
            )

    def to_file(
            self,
            somefile,
    ):
        with open(somefile, 'w') as myfile:
            myfile.write(
                f"{self.vertices.shape[0]} {self.vertices.shape[1]} {self.n_attributes} {self.n_boundary_markers}\n"
            )
            for i in range(self.n_vertices):
                attr = " ".join([str(self.attributes[i, j]) for j in range(self.n_attributes)])
                elem = " ".join([str(self.vertices[i, j]) for j in range(self.vertices.shape[1])])
                bnd = str(self.boundary_markers[i, 0]) if self.n_boundary_markers > 0 else ""
                myfile.write(
                    f"{i+1} {elem} {attr} {bnd}\n"
                )


if __name__ == "__main__":
    nf = NodeFile.from_file("Triangle/A.1.node")
    print(nf.boundary_markers)
    nf.to_file("test.txt")
