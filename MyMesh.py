import numpy as np
from collections import defaultdict
import triangle
import matplotlib.pyplot as plt
from boundary_opencv import get_boundary, get_boundary_numpy
import igl
from read_dataset_npzitem import get_label_masks_from_class_image
import scipy


class MyMesh:
    def __init__(
            self,
            vertices,
            faces,
            vertex_markers,
            holes
    ):
        self.vertices = vertices
        self.vertex_markers = vertex_markers
        self.faces = faces
        self.holes = holes

    def print_dims(self):
        print(f"Vertices: {self.vertices.shape}")
        print(f"Vertex_markers: {self.vertex_markers.shape}")
        print(f"Faces: {self.faces.shape}")
        print(f"Holes: {self.holes}")


    @classmethod
    def default(cls, triangle_flags='qpa0.7'):
        def circle(N, R):
            i = np.arange(N)
            theta = i * 2 * np.pi / N
            pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R
            seg = np.stack([i, i + 1], axis=1) % N
            return pts, seg

        pts0, seg0 = circle(20, 3)
        # pts1, seg1 = circle(8, 0.6)
        # pts = np.vstack([pts0, pts1])
        # seg = np.vstack([seg0, seg1 + seg0.shape[0]])

        # A = dict(vertices=pts, segments=seg, holes=[[0, 0]])
        A = dict(vertices=pts0, segments=seg0, holes=[[-100, 0]])
        B = triangle.triangulate(A, opts=triangle_flags)
        # print(B.keys())

        return cls(vertices=B['vertices'],
                   faces=B['triangles'],
                   vertex_markers=B['vertex_markers'].flatten(),
                   holes=B['holes']
                   )

    @classmethod
    def from_background(cls, background_file):
        pts, edges = get_boundary(background_file, dist_thr=100)
        i = np.arange(len(pts))
        # seg = np.stack([i, i + 1], axis=1) % len(pts)
        A = dict(vertices=pts, segments=edges, holes=[[0, 0]])
        print(f"we start with {pts.shape[0]} boundary points")
        # A = dict(vertices=pts)  #, segments=seg, holes=[[0, 0]])
        B = triangle.triangulate(A, 'qpa100')
        # print(B.keys())
        return cls(vertices=B['vertices'],
                   faces=B['triangles'],
                   vertex_markers=B['vertex_markers'].flatten(),
                   holes=B['holes']
                   )

    @classmethod
    def from_background_mask_pixelwise(cls, background_mask, background_thr=0.5):
        coordinate_to_index_image = -1 * np.ones_like(background_mask)
        vertices = []
        vertex_markers = []
        for i in range(background_mask.shape[0]):
            for j in range(background_mask.shape[1]):
                if background_mask[i, j] < background_thr:
                    coordinate_to_index_image[i, j] = len(vertices)
                    vertices.append([j, i])

                    if (i == 0) or (i == background_mask.shape[0] - 1) or (j == 0) or (
                            j == background_mask.shape[1] - 1):
                        vertex_markers.append(1)
                        continue

                    if (background_mask[i - 1, j] > background_thr) or (background_mask[i + 1, j] > background_thr):
                        vertex_markers.append(1)
                        continue

                    if (background_mask[i, j - 1] > background_thr) or (background_mask[i, j + 1] > background_thr):
                        vertex_markers.append(1)
                        continue

                    vertex_markers.append(0)
        # print(len(vertices))
        # print(len(vertex_markers))
        # plt.imshow(coordinate_to_index_image)
        # plt.colorbar()
        # plt.show()
        faces = []
        for i in range(background_mask.shape[0] - 1):
            for j in range(background_mask.shape[1] - 1):
                """
                a - b
                |   |
                c - d
                clock-wise
                """
                is_foreground = [
                    background_mask[i, j] < background_thr,  # a
                    background_mask[i + 1, j] < background_thr,  # c
                    background_mask[i + 1, j + 1] < background_thr,  # d
                    background_mask[i, j + 1] < background_thr,  # b
                ]
                if sum(is_foreground) < 3:
                    continue
                a = coordinate_to_index_image[i, j]
                b = coordinate_to_index_image[i, j + 1]
                c = coordinate_to_index_image[i + 1, j]
                d = coordinate_to_index_image[i + 1, j + 1]

                if sum(is_foreground) == 4:
                    faces.append([a, c, d])
                    faces.append([a, d, b])
                else:
                    # TODO change ordering for normal orientation
                    faces.append(np.array([a, c, d, b])[is_foreground].tolist())
        return cls(vertices=np.array(vertices),
                   faces=np.array(faces),
                   vertex_markers=np.array(vertex_markers),
                   holes=np.array([])
                   )

    @classmethod
    def from_background_mask_triangle(cls, background_mask, dist_thr, triang_flags='qpa100'):
        pts, edges = get_boundary_numpy(img=background_mask, dist_thr=dist_thr)
        i = np.arange(len(pts))
        # seg = np.stack([i, i + 1], axis=1) % len(pts)
        A = dict(vertices=pts, segments=edges, holes=[[0, 0]])
        print(f"we start with {pts.shape[0]} boundary points")
        # A = dict(vertices=pts)  #, segments=seg, holes=[[0, 0]])
        B = triangle.triangulate(A, triang_flags)
        B['triangles'] = B['triangles'][:, ::-1]
        # print(B.keys())
        return cls(vertices=B['vertices'],
                   faces=B['triangles'],
                   vertex_markers=B['vertex_markers'].flatten(),
                   holes=B['holes']
                   )

    @classmethod
    def from_npz_pixelwise(cls, path_npz, background_thr=0.5):
        """
        given npz_path loads data and uses 'Background' for dense pixel-to-pixel triangulation
        :param path_npz:
        :param background_thr: is background if value > thr
        :return:
        """
        data = np.load(path_npz)
        background_image = get_label_masks_from_class_image(
            class_image=data['classes']
        )['Background']
        return cls.from_background_mask_pixelwise(
            background_mask=background_image,
            background_thr=background_thr,
        )

    @classmethod
    def from_npz_triangle(cls, path_npz, dist_thr=100, triang_flags='qpa100'):
        data = np.load(path_npz)
        background_image = get_label_masks_from_class_image(
            class_image=data['classes']
        )['Background']
        return cls.from_background_mask_triangle(
            background_mask=background_image,
            dist_thr=dist_thr,
            triang_flags=triang_flags,
        )
    
    @staticmethod
    def get_depth_npz(path_npz):
        data = np.load(path_npz)
        # depth = 4 - 4 * data["depth_image"]
        depth = data['depth']
        return depth

    def reshuffle_triangulation_vertices(self) -> None:
        """
        Move boundary vertices to the beginning of vertex list
        :return:
        """
        shuffle_boundary = dict()
        shuffle_interior = dict()
        n_boundary_found, n_interior_found = 0, 0
        for i in range(len(self.vertices)):
            if self.vertex_markers[i] == 1:
                shuffle_boundary[i] = n_boundary_found
                n_boundary_found += 1
            else:
                shuffle_interior[i] = n_interior_found
                n_interior_found += 1
        n_boundary_points = np.sum(self.vertex_markers)
        assert n_boundary_found == n_boundary_points
        shuffle_f = {
            i: shuffle_boundary[i] if self.vertex_markers[i]==1 else shuffle_interior[i] + n_boundary_found
            for i in range(len(self.vertices))
        }
        # do the shuffling
        self.vertices = np.vstack(
            (
                self.vertices[self.vertex_markers == 1],
                self.vertices[self.vertex_markers == 0]
            )
        )
        new_faces = list()
        for f in self.faces:
            new_faces.append(
                [shuffle_f[f[0]], shuffle_f[f[1]], shuffle_f[f[2]]]
            )
        self.faces = np.array(new_faces)
        self.vertex_markers = np.hstack(
            (
                np.ones(n_boundary_points, dtype=int),
                np.zeros(self.vertices.shape[0]-n_boundary_points, dtype=int)
             )
        )

    def plot2d(self, show=False):
        plt.scatter(self.vertices[:,0], self.vertices[:,1], c=self.vertex_markers)
        if show:
            plt.axis("equal")
            plt.show()

    def export_obj(self, file_path, front_surface_count=-1):
        with open(file_path, "w") as f:
            if front_surface_count > 0:
                f.write("mtllib surface.mtl\n")
            for vertex in self.vertices:
                f.write(
                    f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"
                )
            i = -1
            if front_surface_count > 0:
                f.write("usemtl BSpline\n")
            for face in self.faces:
                i += 1
                if i == front_surface_count:
                    f.write("usemtl Cylinder\n")
                f.write(
                    f"f {face[0]+1} {face[1]+1} {face[2]+1}\n"
                )

    def export_colored_ply(self, file_path, vertex_color_index):

        color_palette = np.stack(
            [(np.array(c)*255).astype(int) for c in plt.cm.tab20b.colors]
        )

        def vertex_index_to_color_str(x: int):
            if x == -1:
                return "255 255 255"
            x = x % 20
            return f"{color_palette[x][0]} {color_palette[x][1]} {color_palette[x][2]}"

        with open(file_path, 'w') as f:
            lines_to_write = [
                f"ply\nformat ascii 1.0\ncomment object: depth to PLY mesh\n",
                f"element vertex {len(self.vertices)}",
                f"\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\n"
                f"property uchar green\nproperty uchar blue\n",
                f"element face {len(self.faces)}\n",
                f"property list uchar int vertex_index\nend_header\n"
            ]
            f.writelines(lines_to_write)
            for i in range(len(self.vertices)):
                vertex = self.vertices[i]
                color = vertex_color_index[i] if i<len(vertex_color_index) else -1
                f.write(
                    f"{vertex[0]} {vertex[1]} {vertex[2]} {vertex_index_to_color_str(color)}\n"
                )
            for i in range(len(self.faces)):
                face = self.faces[i]
                f.write(
                    f"{len(face)} {face[0]} {face[1]} {face[2]}\n"
                )

    def vertices_to_camera_coords(self, imsize):
        """
        Uses camera transformations to make vertices at 400x400 scale to corresponding x y positions
        :param imsize:
        :return:
        """
        res_x, res_y = imsize, imsize
        ortho_scale = 3.5
        A = np.array(
            [
                [res_x/ortho_scale, 0, res_x/2],
                [0, -res_y/ortho_scale, res_y/2],
                [0, 0, 1]
            ]
        )
        Ainv = np.linalg.inv(A)
        temp_vert_matrix = np.vstack(
            (
                self.vertices[:, 0],
                self.vertices[:, 1],
                np.ones(self.vertices.shape[0])
            )
        ).astype(float)
        # print("temp_vert_matrix: ", temp_vert_matrix.shape)
        res = np.matmul(Ainv, temp_vert_matrix)
        # print(res)
        res[0, :] /= res[2, :]
        res[1, :] /= res[2, :]

        self.vertices = self.vertices.astype(float)
        self.vertices[:, 0] = res[0, :]
        self.vertices[:, 1] = res[1, :]

    def get_face_markers(self):
        """
        :return: binary list, 1 for a boundary face
        """
        face_markers = list()
        for f in self.faces:
            v_nmarkers = [self.vertex_markers[i] for i in f]
            if np.sum(v_nmarkers) > 1:
                face_markers.append(1)
            else:
                face_markers.append(0)
        return face_markers

    def plot_html(self, name="test.html", shading=dict(), normals=None, offline=True, c=None):
        raise NotImplementedError

    def get_class_from_classimage(self, classimage):
        classes = np.ones(self.vertices.shape[0], dtype=int)
        print(classimage.shape)
        for i in range(self.vertices.shape[0]):
            v_x, v_y = self.vertices[i][0], self.vertices[i][1]
            xa, xb = int(np.floor(v_x)), int(np.ceil(v_x))
            xa, xb = min(xa, xb), max(xa, xb)
            alpha_x, _ = np.modf(v_x)
            ya, yb = int(np.floor(v_y)), int(np.ceil(v_y))
            ya, yb = min(ya, yb), max(ya, yb)
            alpha_y, _ = np.modf(v_y)
            pixel_values = classimage[ya:yb+1, xa:xb+1]
            c = np.max(pixel_values)
            if self.vertex_markers[i] != 1:
                values, counts = np.unique(pixel_values, return_counts=True)
                c = values[np.argmax(counts)]
            else:
                c = np.max(pixel_values)
            classes[i] = c
        return classes


class MyMesh3D(MyMesh):
    def __init__(self,
            vertices,
            faces,
            vertex_markers,
            holes):
        assert vertices.shape[1]==3
        super(MyMesh3D, self).__init__(vertices, faces, vertex_markers, holes)

    @classmethod
    def fromMyMesh2d(cls, mm: MyMesh):
        return cls(
            vertices=np.hstack((mm.vertices, np.zeros((mm.vertices.shape[0], 1)))),
            faces=mm.faces,
            vertex_markers=mm.vertex_markers,
            holes=mm.holes
        )

    def setZ(self, Z):
        assert Z.shape[0] == self.vertices.shape[0]
        self.vertices[:,2] = Z

    def setZ_from_depth(self, depth):
        xx, yy = np.arange(depth.shape[1]), np.arange(depth.shape[0])
        f = scipy.interpolate.interp2d(xx, yy, depth, kind='linear')
        Z = np.ones(self.vertices.shape[0])
        for i in range(self.vertices.shape[0]):
            v_x, v_y = self.vertices[i][0], self.vertices[i][1]
            # xa, xb = int(np.floor(v_x)), int(np.ceil(v_x))
            # alpha_x, _ = np.modf(v_x)
            # ya, yb = int(np.floor(v_y)), int(np.ceil(v_y))
            # alpha_y, _ = np.modf(v_y)
            # z = depth[ya, xa]
            z = f(v_x, v_y)
            # TODO: linear interpolation on the boundary!
            # pixel_values = depth[yb:ya+1, xb:xa+1]
            # if self.vertex_markers[i] != 1:
            #     z = np.mean(pixel_values)
            # else:
            #     z = np.min(pixel_values)
            # Z[i] = (4-z)
            Z[i] = -z
            # if i < np.sum(self.vertex_markers):
            #     if z >= 4:
            #         print(f"--- ! {i} ! ---")
            #         print(f"boundary {i} vertex ({v_x}, {v_y}) -> z={z}")
        self.setZ(Z)

    def make_back_surface(self):
        self.reshuffle_triangulation_vertices()
        n_boundary_points = np.sum(self.vertex_markers)
        print(f"Boundary points: {n_boundary_points}")
        n_internal_points = self.vertices.shape[0] - n_boundary_points
        print(f"Internal points: {n_internal_points}, double it")
        self.vertices = np.vstack(
            (
                self.vertices,
                self.vertices[n_boundary_points:]
            )
        )
        print(f" - vertices: {self.vertices.shape}")
        self.vertex_markers = np.hstack(
            (
                self.vertex_markers,
                np.zeros(n_internal_points)
            )
        )
        print(f"Faces: {self.faces.shape}")
        new_faces = list()
        for f in self.faces:
            # print(f)
            nf = [
                f[0] + n_internal_points if f[0] >= n_boundary_points else f[0],
                f[1] + n_internal_points if f[1] >= n_boundary_points else f[1],
                f[2] + n_internal_points if f[2] >= n_boundary_points else f[2]
            ]
            if (f[0] < n_boundary_points) and (f[1] < n_boundary_points) and (f[2] < n_boundary_points):
                continue
            nf.reverse()
            new_faces.append(nf)
        print(f"append faces: {len(new_faces)}")

        self.faces = np.vstack(
            (
                self.faces,
                np.array(new_faces)
            )
        )
        print(f" - faces: {self.faces.shape}")

        # self.vertices[n_internal_points+n_boundary_points:, 2] = -20
        self.vertices[n_internal_points+n_boundary_points:, 2] = np.min(
            self.vertices[:n_internal_points+n_boundary_points, 2]
        )


if __name__ == "__main__":
    npzname = "data/00000136.npz"
    # mm = MyMesh.from_npz_pixelwise(path_npz="data/00000007.npz")
    # print("FACES")
    # print(mm.faces[:15])
    # print("MARKERS")
    # print(mm.vertex_markers[:15])
    mm = MyMesh.from_npz_triangle(path_npz=npzname, dist_thr=200)
    depth = mm.get_depth_npz(path_npz=npzname)
    print(depth.shape)

    for f in mm.faces:
        is_boundary = [
            mm.vertex_markers[f[0]],
            mm.vertex_markers[f[1]],
            mm.vertex_markers[f[2]]
        ]
        if np.sum(is_boundary) > 2:
            print(f)

    mm.reshuffle_triangulation_vertices()
    mm.get_face_markers()

    data = np.load(npzname)

    if len(mm.faces) < 500:
        EV, FE, EF = igl.edge_topology(v=mm.vertices, f=mm.faces)
        plt.figure(figsize=(10,10))
        plt.scatter(mm.vertices[:,0], mm.vertices[:,1], c=mm.vertex_markers, cmap='jet', zorder=5)
        plt.imshow(data['Background'], cmap="gray_r", alpha=0.6)
        for e in EV:
            plt.plot([mm.vertices[e[1],0], mm.vertices[e[0],0]], [mm.vertices[e[1],1], mm.vertices[e[0],1]], color='k')
        plt.show()

    m3 = MyMesh3D.fromMyMesh2d(mm)
    m3.plot_html("1.html", shading={"wireframe": True, "width": 1000, "height": 600,})
    m3.get_face_markers()

    m3.reshuffle_triangulation_vertices()
    m3.setZ_from_depth(depth)
    # print(np.array(m3.vertex_markers).flatten())
    # m3.reshuffle_triangulation_vertices()
    # print(m3.vertex_markers)
    # print(np.sum(m3.vertex_markers))
    m3.make_back_surface()
    mm.vertices_to_camera_coords(imsize=400)
    m3.vertices_to_camera_coords(imsize=400)
    m3.plot_html("2.html")
    m3.export_obj("data/m3.obj", front_surface_count=mm.faces.shape[0])

    plt.imshow(depth)
    plt.colorbar()
    plt.show()
