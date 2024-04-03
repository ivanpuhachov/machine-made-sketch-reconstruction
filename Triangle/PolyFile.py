import numpy as np


class PolyFile:
    def __init__(
        self,
        vertices: np.array,
        vertex_atributes: np.array,
        vertex_boundary_markers: np.array,
        segments: np.array,
        segment_boundary_markers: np.array,
        holes: np.array,
    ):
        """
        poly file specs - http://www.cs.cmu.edu/~quake/triangle.poly.html
        :param vertices:
        :param vertex_atributes:
        :param vertex_boundary_markers:
        :param segments:
        :param segment_boundary_markers:
        :param holes:
        """
        self.vertices = vertices
        self.n_vertices = vertices.shape[0]
        self.vertex_attributes = vertex_atributes
        self.vertertex_boundary_markers = vertex_boundary_markers
        self.segments = segments
        self.n_segments = segments.shape[0]
        self.segments_boundary_markers = segment_boundary_markers
        self.holes = holes
        self.n_holes = holes.shape[0]

    @classmethod
    def from_dict(
            cls,
            data: dict,
    ):
        """

        :param data:
        :return:
        """
        vertices = data['vertices']
        n_vertices = vertices.shape[0]
        segments = data['segments']
        n_segments = segments.shape[0]
        vertex_boundary_markers = np.zeros(shape=(n_vertices, 0), dtype=int)
        segments_boundary_markers = np.zeros(shape=(n_segments, 0), dtype=int)
        holes = np.zeros(shape=(0,2), dtype=float)
        if 'vertex_markers' in data.keys():
            vertex_boundary_markers = data['vertex_markers']
            assert vertex_boundary_markers.shape[0] == n_vertices
        if 'segment_markers' in data.keys():
            segments_boundary_markers = data['segment_markers']
            assert segments_boundary_markers.shape[0] == n_segments
        if 'holes' in data.keys():
            holes = data['holes']
        vertex_attributes = np.zeros(shape=(n_vertices, 0), dtype=float)
        return cls(
            vertices=vertices,
            vertex_atributes=vertex_attributes,
            vertex_boundary_markers=vertex_boundary_markers,
            segments=segments,
            segment_boundary_markers=segments_boundary_markers,
            holes=holes,
        )


    @classmethod
    def from_file(
            cls,
            somefile,
    ):
        with open(somefile) as f:
            lines = f.readlines()
            header = lines[0]
            hdspl = header.split(" ")
            n_vertices = int(hdspl[0])
            n_attributes = int(hdspl[2])
            vertices = np.zeros(shape=(n_vertices, 2), dtype=float)
            vertex_attributes = np.zeros(shape=(n_vertices, n_attributes), dtype=float)
            vertex_boundary = np.zeros(shape=(n_vertices, int(hdspl[3])), dtype=int)
            vertex_lines = lines[1:1+n_vertices]
            segments_header = lines[1+n_vertices]
            sspl = segments_header.split(" ")
            n_segments = int(sspl[0])
            segments = np.zeros(shape=(n_segments, 2), dtype=int)
            segment_lines = lines[2+n_vertices:2+n_segments+n_vertices]
            segment_boundary = np.zeros(shape=(n_segments, int(sspl[1])), dtype=int)
            holes_header = lines[2+n_segments+n_vertices]
            hspl = holes_header.split(" ")
            n_holes = int(hspl[0])
            hole_lines = lines[3+n_segments+n_vertices:3+n_vertices+n_segments+n_holes]
            holes = np.zeros(shape=(n_holes, 2), dtype=float)

            for i in range(n_vertices):
                l = vertex_lines[i]
                lspl = l.split(" ")
                vertices[i, 0] = float(lspl[1])
                vertices[i, 1] = float(lspl[2])
                if n_attributes > 0:
                    for j in range(n_attributes):
                        vertex_attributes[i, j] = float(lspl[3+j])
                if vertex_boundary.shape[1] > 0:
                    vertex_boundary[i, 0] = int(l[3+n_attributes])

            for i in range(n_segments):
                l = segment_lines[i]
                lspl = l.split(" ")
                segments[i, 0] = int(lspl[1])
                segments[i, 1] = int(lspl[2])
                if segment_boundary.shape[1] > 0:
                    segment_boundary[i, 0] = int(lspl[3])

            for i in range(n_holes):
                l = hole_lines[i]
                lspl = l.split(" ")
                holes[i, 0] = float(lspl[1])
                holes[i, 1] = float(lspl[2])

            return cls(
                vertices=vertices,
                vertex_atributes=vertex_attributes,
                vertex_boundary_markers=vertex_boundary,
                segments=segments,
                segment_boundary_markers=segment_boundary,
                holes=holes,
            )

    def to_file(
            self,
            somefile,
    ):
        with open(somefile, 'w') as myfile:
            myfile.write(f"{self.vertices.shape[0]} {self.vertices.shape[1]} {self.vertex_attributes.shape[1]} {self.vertertex_boundary_markers.shape[1]}\n")
            for i_vertex in range(self.n_vertices):
                attr = " " + (" ".join(
                    [
                        str(self.vertex_attributes[i_vertex, j])
                        for j in range(self.vertex_attributes.shape[1])
                    ])
                ) if self.vertex_attributes.shape[1] > 0 else ""
                bndr = (" " + str(self.vertertex_boundary_markers[i_vertex])) if self.vertertex_boundary_markers.shape[1] > 0 else ""
                myfile.write(f"{i_vertex+1} {self.vertices[i_vertex, 0]} {self.vertices[i_vertex, 1]}{attr}{bndr}\n")
            myfile.write(f"{self.segments.shape[0]} {self.segments_boundary_markers.shape[1]}\n")
            for i_segment in range(self.n_segments):
                bndr = (" " + str(self.segments_boundary_markers[i_vertex])) if self.segments_boundary_markers.shape[1] > 0 else ""
                myfile.write(f"{i_segment+1} {self.segments[i_segment, 0]} {self.segments[i_segment, 1]}{bndr}\n")
            myfile.write(f"{self.holes.shape[0]}\n")
            for i_hole in range(self.n_holes):
                myfile.write(f"{i_hole+1} {self.holes[i_hole, 0]} {self.holes[i_hole, 1]}")


if __name__ == "__main__":
    # pf = PolyFile.from_file(somefile="Triangle/A.poly")
    dd = dict()
    dd['vertices'] = np.array([
        [0, 0],
        [1, 0],
        [0.3, 0.3],
        [0, 1],
    ])
    dd['segments'] = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
    ])
    pf = PolyFile.from_dict(data=dd)
    pf.to_file("Triangle/test.poly")
