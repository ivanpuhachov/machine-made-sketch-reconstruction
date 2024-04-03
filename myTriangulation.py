import numpy as np
import triangle
import matplotlib.pyplot as plt
from boundary_opencv import get_boundary, get_boundary_numpy
import igl
import scipy
from util_svgs import naive_sampler, clean_svg_sampler
from util_trappedball_fill import my_trapped_ball
from Triangle.PolyFile import PolyFile
from Triangle.NodeFile import NodeFile
from Triangle.EleFile import EleFile
import subprocess
import pathlib
from collections import Counter
from camera_transformations import get_pixel_coordinates
import warnings
import argparse
from pathlib import Path


class myTraingulation:
    def __init__(
            self,
            vertices,
            faces,
            vertex_markers=None,
            holes=None,
            n_svg_points=0,
    ):
        self.vertices = vertices  # has shape (n, 2)
        self.vertex_markers = vertex_markers
        self.faces = faces
        self.holes = holes
        self.n_svg_points = n_svg_points  # contains number of points sampled from strokes. those are first points in vertices list

    @classmethod
    def default(cls, triangle_flags='qpa0.7'):
        def circle(N, R):
            i = np.arange(N)
            theta = i * 2 * np.pi / N
            pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R
            seg = np.stack([i, i + 1], axis=1) % N
            return pts, seg

        pts0, seg0 = circle(20, 3)
        pts1, seg1 = circle(8, 0.6)
        pts = np.vstack([pts0, pts1])
        seg = np.vstack([seg0, seg1 + seg0.shape[0]])

        A = dict(vertices=pts, segments=seg, holes=[[-30, 0], [0, 0]])
        B = triangle.triangulate(A, opts=triangle_flags)
        print("Triangle lib returns: ", B.keys())

        return cls(vertices=B['vertices'],
                   faces=B['triangles'],
                   vertex_markers=B['vertex_markers'].flatten(),
                   holes=B['holes'],
                   n_svg_points=len(pts),
                   )

    @classmethod
    def from_background_mask_triangle(cls, background_mask, dist_thr, triang_flags='qpa100'):
        """
        triangulate foreground with triangle lib
        :param background_mask: np array, 1 - background, 0 - foreground
        :param dist_thr: distance**2 to simplify boundary
        :param triang_flags: flags to pass to traingle lib
        :return:
        """
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
                   holes=B['holes'],
                   n_svg_points=len(pts),
                   )

    @classmethod
    def from_svg(cls, path_to_svg, svg_sampling_distance=40, triang_flags='YYqa200'):
        pts, edges, paths_edges = clean_svg_sampler(
            path_to_svg=path_to_svg,
            sampling_distance=svg_sampling_distance,
        )
        edges = np.array(edges, dtype=int)
        pts = np.array(pts, dtype=float)
        pts = np.around(pts, decimals=3)
        print("points from svg: ", pts.shape)
        print("edges from ", edges.shape)
        return cls.from_sampled_points_edges(
            pts=pts,
            edges=np.array(edges),
            triang_flags=triang_flags,
            edges_by_path=paths_edges,
        )

    @classmethod
    def from_sampled_points_edges(cls, pts: np.array, edges: np.array, edges_by_path: list, triang_flags='YYqa200'):
        assert np.min(edges) == 0
        dd = dict()
        dd['vertices'] = pts
        dd['segments'] = edges.astype(int) + 1
        dd['holes'] = np.array([[-5, -5]])
        pf = PolyFile.from_dict(
            data=dd,
        )
        pf.to_file("Triangle/lastsvg.poly")

        process = subprocess.Popen(
            f"./Triangle/triangle -{triang_flags} Triangle/lastsvg.poly | tee Triangle/lastsvg_logs.txt",
            shell=True,
            stdout=subprocess.PIPE
        )
        process.wait()
        print(" --- > Triangle subprocess exit code: ", process.returncode)
        if int(process.returncode) != 0:
            raise Exception("Triangle bad termination. See Triangle/lastsvg_logs.txt")
        ef = EleFile.from_file("Triangle/lastsvg.1.ele")
        nf = NodeFile.from_file("Triangle/lastsvg.1.node")

        newfaces = np.zeros_like(ef.triangles, dtype=int)
        newfaces[:, 0] = ef.triangles[:, 0]
        newfaces[:, 1] = ef.triangles[:, 2]  # intentionally swapping the axis so the normals are outward
        newfaces[:, 2] = ef.triangles[:, 1]
        newfaces = newfaces - 1  # because Triangle indexes vertices from 1

        # A = dict(vertices=pts.tolist(), holes=[[0, 0]])
        # print(A)
        # triang_flags = 'qa20'
        # B = triangle.triangulate(A, opts=triang_flags)
        # print(B['vertex_markers'])
        return cls(vertices=nf.vertices,
                   faces=newfaces,
                   vertex_markers=nf.boundary_markers.flatten(),
                   # holes=B['holes'],
                   n_svg_points=len(pts),
                   ), \
               pts, \
               edges, \
               edges_by_path

    def reshuffle_triangulation_vertices(
            self,
            use_this_marker=None,
            return_mapping=False,
    ):
        """
        Move vertices marked by 1 to the beginning of vertex list. Shuffle faces and boundary markers correspondingly.
        By default, use self.vertex_markers
        :return:
        """
        if use_this_marker is None:
            use_this_marker = self.vertex_markers
        shuffle_boundary = dict()
        shuffle_interior = dict()
        n_boundary_found, n_interior_found = 0, 0
        for i in range(len(self.vertices)):
            if use_this_marker[i] == 1:
                shuffle_boundary[i] = n_boundary_found
                n_boundary_found += 1
            else:
                shuffle_interior[i] = n_interior_found
                n_interior_found += 1
        n_boundary_points = np.sum(use_this_marker)
        assert n_boundary_found == n_boundary_points
        shuffle_f = {
            i: shuffle_boundary[i] if use_this_marker[i] == 1 else shuffle_interior[i] + n_boundary_found
            for i in range(len(self.vertices))
        }
        # do the shuffling
        self.vertices = np.vstack(
            (
                self.vertices[use_this_marker == 1],
                self.vertices[use_this_marker == 0]
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
                self.vertex_markers[use_this_marker == 1],
                self.vertex_markers[use_this_marker == 0]
            )
        )
        if return_mapping:
            return shuffle_f

    def plot(
            self,
            faces=False,
            ids=True,
            show=False,
            saveto=pathlib.Path("reports/"),
            name="triangulation",
    ):
        plt.figure()
        # plt.scatter(self.vertices[self.n_svg_points:, 0], self.vertices[self.n_svg_points:, 1], c=np.arange(len(self.vertices) - self.n_svg_points), zorder=5)
        if ids:
            plt.scatter(self.vertices[:self.n_svg_points, 0], self.vertices[:self.n_svg_points, 1], color='red', zorder=6, marker="*", s=1)
            for idx in range(self.n_svg_points):
                v1 = self.vertices[idx]
                plt.text(v1[0], v1[1], s=f"{idx}",
                         fontsize=1,
                         zorder=7,
                         )
        else:
            pass
            # plt.scatter(self.vertices[:self.n_svg_points, 0], self.vertices[:self.n_svg_points, 1], color='k',
            #             zorder=6, marker=".", s=3)

        # plt.colorbar()
        if faces:
            EV, FE, EF = igl.edge_topology(v=self.vertices, f=self.faces)
            for e in EV:
                plt.plot([self.vertices[e[1], 0], self.vertices[e[0], 0]],
                         [self.vertices[e[1], 1], self.vertices[e[0], 1]],
                         color='gray', linewidth=0.5)
        plt.axis("equal")
        if show:
            plt.show()
        else:
            plt.savefig(saveto / f"{name}.svg")
            plt.close()

    def plot_on_top_of_image(
            self,
            image,
            saveto=pathlib.Path("reports/"),
            name="triang_image",
            cmap='tab20b',
    ):
        plt.figure()
        plt.imshow(image, interpolation='nearest', cmap='tab20b', zorder=3)
        # plt.scatter(self.vertices[self.n_svg_points:, 0], self.vertices[self.n_svg_points:, 1], c=np.arange(len(self.vertices) - self.n_svg_points), zorder=5)
        plt.scatter(self.vertices[:self.n_svg_points, 0], self.vertices[:self.n_svg_points, 1], color='red', zorder=6,
                    marker="*", s=1)
        for idx in range(self.n_svg_points):
            v1 = self.vertices[idx]
            plt.text(v1[0], v1[1], s=f"{idx}",
                     fontsize=1,
                     zorder=7,
                     )
        EV, FE, EF = igl.edge_topology(v=self.vertices, f=self.faces)
        for e in EV:
            plt.plot([self.vertices[e[1], 0], self.vertices[e[0], 0]],
                     [self.vertices[e[1], 1], self.vertices[e[0], 1]],
                     color='k', linewidth=0.5)
        plt.axis("equal")
        plt.savefig(saveto / f"{name}.svg")
        plt.close()

    def build_vertex_to_faces_map(self):
        VF, NI = igl.vertex_triangle_adjacency(f=self.faces, n=len(self.vertices))
        vertex_degree = {i: NI[i+1] - NI[i] for i in range(len(self.vertices))}
        assert sum(vertex_degree.values()) == 3 * len(self.faces)
        vertex_to_faces = {
            i: VF[NI[i]:NI[i+1]]
            for i in range(len(self.vertices))
        }
        return vertex_to_faces

    def get_fillmap_face_pixels(
            self,
            this_face_id: int,
            fillmap: np.array,
    ):
        triang_points = self.vertices[self.faces[this_face_id], :]
        limits = (
            (np.floor(np.min(triang_points[:, 0])), np.ceil(np.max(triang_points[:, 0]))),
            (np.floor(np.min(triang_points[:, 1])), np.ceil(np.max(triang_points[:, 1]))),
        )
        p = list()
        for a in range(int(limits[0][0]), int(limits[0][1])):
            for b in range(int(limits[1][0]), int(limits[1][1])):
                p.append([a, b, 0])
        p = np.array(p).reshape(-1, 3).astype(float)
        a = np.array([triang_points[0, 0], triang_points[0, 1], 0]).reshape(1, 3)
        a = np.repeat(a, p.shape[0], axis=0)
        b = np.array([triang_points[1, 0], triang_points[1, 1], 0]).reshape(1, 3)
        b = np.repeat(b, p.shape[0], axis=0)
        c = np.array([triang_points[2, 0], triang_points[2, 1], 0]).reshape(1, 3)
        c = np.repeat(c, p.shape[0], axis=0)

        baryc_coords = igl.barycentric_coordinates_tri(p, a, b, c)
        internal_points = list()
        fillmap_values = set()
        for i_p in range(p.shape[0]):
            if (baryc_coords[i_p, 0] >= 0) and (baryc_coords[i_p, 1] >= 0) and (baryc_coords[i_p, 2] >= 0):
                internal_points.append([p[i_p, 0], p[i_p, 1]])
                fillmap_values.add(
                    fillmap[int(p[i_p, 1]), int(p[i_p, 0])]
                )
        return fillmap_values

    def build_face_to_fillmap_id(self, fillmap):
        """
        Build a map between face_id -> {0, 2} (set of region ID from fillmap inside this triangle)
        :param fillmap: np.array of id (int), comes from trapped ball
        :return:
        """
        face_to_fillmap = {x: {} for x in range(len(self.faces))}
        for x in range(len(self.faces)):
            face_to_fillmap[x] = self.get_fillmap_face_pixels(this_face_id=x, fillmap=fillmap)
        return face_to_fillmap

    def get_vertex_type_from_fillmap(
            self,
            fillmap,
    ):
        adjacency_list = igl.adjacency_list(self.faces)
        vertex_class = np.ones_like(self.vertices[:, 0])
        for i in range(len(vertex_class)):
            v_x, v_y = self.vertices[i][0], self.vertices[i][1]
            window_size = 3
            window = np.copy(fillmap[
                             int(v_y - window_size):int(v_y + window_size),
                             int(v_x - window_size):int(v_x + window_size),
                             ].flatten())
            window = window[window != 1]  # exclude background
            if i < self.n_svg_points:
                # boundary points are only those sampled from svg file
                vertex_class[i] = 0
            else:
                window = window[window != 0]
                if len(window) == 0:
                    print(f"Can't get vertex {i} class here ({v_x, v_y}), using window 5")
                    window_size = 5
                    window = np.copy(fillmap[
                                     int(v_y - window_size):int(v_y + window_size),
                                     int(v_x - window_size):int(v_x + window_size),
                                     ].flatten())
                    window = window[window != 0]
                    window = window[window != 1]
                    if len(window) == 0:
                        print(np.copy(fillmap[
                             int(v_y - window_size):int(v_y + window_size),
                             int(v_x - window_size):int(v_x + window_size),
                             ]))
                        warnings.warn(f"Can't get vertex  {i} class! ({v_x, v_y}), using largest window")
                        window_size = 10
                        window = np.copy(fillmap[
                                         int(v_y - window_size):int(v_y + window_size),
                                         int(v_x - window_size):int(v_x + window_size),
                                         ].flatten())
                        window = window[window != 0]
                        window = window[window != 1]
                        if len(window) == 0:
                            raise Exception(f"Can't get vertex {i} class!")
                vertex_class[i] = scipy.stats.mode(window, keepdims=False)[0]
                if vertex_class[i] == 0:
                    raise Exception("class 0")
            # print(scipy.stats.mode(window)[0][0])
        vertex_class = vertex_class.astype(int)

        for i_v in range(self.n_svg_points, len(vertex_class)):
            # for each vertex not on svg lines, check 1-ring neigh and warn if it is not uniform
            ring1_adj = adjacency_list[i_v]
            ring1_fillmap = [vertex_class[x] for x in ring1_adj]
            set_ring1 = set(ring1_fillmap)
            set_ring1.discard(0)
            if vertex_class[i_v] not in set_ring1:
                if len(set_ring1) == 1:
                    vertex_class[i_v] = set_ring1.pop()
                    print(f"Cured internal vertex {i_v} with 1-ring neighbors {set_ring1}")
                else:
                    warnings.warn(
                        f"Internal vertex {i_v}  CLASS {vertex_class[i_v]} 1-ring is not in ring-1 classes {set_ring1}")

        return vertex_class.astype(int)

    def get_vertices_to_camera_coords(
            self,
            imshape=(400, 400),
    ):
        pixels_to_camera_coordinates = get_pixel_coordinates(
            shape=imshape,
        )
        xgrid = pixels_to_camera_coordinates[..., 0]
        ygrid = pixels_to_camera_coordinates[..., 1]
        triang_x = self.interpolate_f_on_vertices(f_grid=xgrid)
        triang_y = self.interpolate_f_on_vertices(f_grid=ygrid)
        return np.vstack((triang_x, triang_y)).transpose()

    def vertices_to_camera_coords(
            self,
            imshape=(400, 400),
    ):
        self.vertices = self.get_vertices_to_camera_coords(imshape=imshape)

    @staticmethod
    def get_high_degree_svg_vertices(
            svg_edges: np.array,
    ):
        result = set()
        vertex_valence_counter = Counter(svg_edges.flatten().tolist())
        for k, v in vertex_valence_counter.items():
            if v == 2:
                continue
            if v == 1:
                print(f"svg vertex {k} has valence {v}")
            if v > 2:
                print(f"svg vertex {k} has high valence: {v}")
                result.add(k)
        return result

    @staticmethod
    def build_svg_adjacency_map(
            svg_edges: np.array,
    ):
        adjacency_map = {
            x: set()
            for x in range(np.max(svg_edges)+1)
        }
        for e in svg_edges:
            adjacency_map[e[0]].add(e[1])
            adjacency_map[e[1]].add(e[0])
        return adjacency_map

    def build_svg_vertex_to_region_dict(
            self,
            fillmap: np.array,
            svg_edges: np.array,
    ):
        print("\n\n ======================")
        adjacency_list = igl.adjacency_list(self.faces)
        map_vertex_to_faces = self.build_vertex_to_faces_map()
        vertex_fillmap_idx = self.get_vertex_type_from_fillmap(fillmap)
        n_regions = np.max(fillmap) + 1
        svg_adjacency_map = self.build_svg_adjacency_map(svg_edges=svg_edges)
        set_of_high_degree_svg_junctions = self.get_high_degree_svg_vertices(svg_edges=svg_edges)
        set_of_problematic_vertices = set()  # contains boundary verts with no corresponding regions or internal vertex with <2 corresponding region
        dict_svg_vertex_to_region = {x: set() for x in range(self.n_svg_points)}
        for i_svg_vertex in dict_svg_vertex_to_region.keys():
            if i_svg_vertex in set_of_high_degree_svg_junctions:
                continue
            adjacent_vertices = adjacency_list[i_svg_vertex]
            adjacent_internal_vertices = [x for x in adjacent_vertices if x >= self.n_svg_points]
            if len(adjacent_internal_vertices) == 0:
                print(f"warn: svg vertex {i_svg_vertex} has 0 internal neighbours")
            if (len(adjacent_internal_vertices) < 2) and (self.vertex_markers[i_svg_vertex] != 1):
                print(f"warn: svg vertex {i_svg_vertex} has 1 internal neighbour {adjacent_internal_vertices}")
            for iv in adjacent_internal_vertices:
                dict_svg_vertex_to_region[i_svg_vertex].add(vertex_fillmap_idx[iv])
            if len(dict_svg_vertex_to_region[i_svg_vertex]) <= 1:
                if (self.vertex_markers[i_svg_vertex] == 1) and (len(dict_svg_vertex_to_region[i_svg_vertex]) == 0):
                    print(f"ERR: svg boundary vertex {i_svg_vertex} has 0 correspondences {dict_svg_vertex_to_region[i_svg_vertex]}")
                    set_of_problematic_vertices.add(i_svg_vertex)
                if self.vertex_markers[i_svg_vertex] != 1:
                    print(f"ERR: svg interior vertex {i_svg_vertex} has 0 or 1 correspondences {dict_svg_vertex_to_region[i_svg_vertex]}")
                    set_of_problematic_vertices.add(i_svg_vertex)
            if len(dict_svg_vertex_to_region[i_svg_vertex]) > 2:
                warnings.warn(f"svg vertex {i_svg_vertex} has {len(dict_svg_vertex_to_region[i_svg_vertex])} connected "
                              f"regions ({dict_svg_vertex_to_region[i_svg_vertex]}), but it is not a high-degree vertex"
                              )
        print("Set of problematic vertices: ", set_of_problematic_vertices)
        i = 0
        while (len(set_of_problematic_vertices) > 0) and (i < 10):
            i += 1
            set_of_fixed_vertices = set()
            for x in set_of_problematic_vertices:
                adj = svg_adjacency_map[x]
                adj = adj - set_of_high_degree_svg_junctions
                adj = list(adj)
                if len(adj) == 0:
                    continue
                adj_regions = dict_svg_vertex_to_region[x].union(dict_svg_vertex_to_region[adj[0]])
                if len(adj) == 2:
                    adj_regions = adj_regions.union(dict_svg_vertex_to_region[adj[1]])
                if (len(adj_regions) > 1) or ((len(adj_regions) == 1) and (self.vertex_markers[x] == 1)):
                    dict_svg_vertex_to_region[x] = adj_regions
                    set_of_fixed_vertices.add(x)
                    print(f"Cured vertex {x} with adjacent vertices {adj} regions - {dict_svg_vertex_to_region[x]}")
            set_of_problematic_vertices = set_of_problematic_vertices - set_of_fixed_vertices
        print("After svg neighbors cure, problematic vertices are: ", set_of_problematic_vertices)
        set_of_fixed_vertices = set()
        for i_v in set_of_problematic_vertices:
            ring1_face_fillmap = set()
            for i_f in map_vertex_to_faces[i_v]:
                ring1_face_fillmap = ring1_face_fillmap.union(self.get_fillmap_face_pixels(i_f, fillmap=fillmap))
            ring1_face_fillmap.discard(0)
            ring1_face_fillmap.discard(1)
            if len(ring1_face_fillmap) > 1:
                print(f"ring1 face fillmap for vertex {i_v} says: {ring1_face_fillmap}")
                set_of_fixed_vertices.add(i_v)
                dict_svg_vertex_to_region[i_v] = ring1_face_fillmap
        set_of_problematic_vertices = set_of_problematic_vertices - set_of_fixed_vertices
        if len(set_of_problematic_vertices) > 0:
            warnings.warn(f"Can't cure problematic vertices: {set_of_problematic_vertices}")
            for i_svg_vertex in set_of_problematic_vertices:
                ring1_adj = adjacency_list[i_svg_vertex]
                ring2_adj = set()
                for x in ring1_adj:
                    ring2_adj = ring2_adj.union(set(adjacency_list[x]))
                ring2_adj_regions = {vertex_fillmap_idx[x] for x in ring2_adj}
                ring2_adj_regions.discard(0)
                ring2_adj_regions.discard(1)
                print(f"2-ring of vertex {i_svg_vertex} has regions {ring2_adj_regions}")
                if len(ring2_adj_regions) == 0:
                    raise Exception(f"2-ring neighbours of vertex {i_svg_vertex} is only 1 and 0 regions!")
                if len(ring2_adj_regions) > 2:
                    continue
                if len(dict_svg_vertex_to_region[i_svg_vertex]) >= 1:
                    continue
                print(f"new region corr: {i_svg_vertex} - {ring2_adj_regions}")
                dict_svg_vertex_to_region[i_svg_vertex] = ring2_adj_regions
        for i_svg_vertex in set_of_high_degree_svg_junctions:
            adjacent_vertices = adjacency_list[i_svg_vertex]
            adjacent_internal_vertices = [x for x in adjacent_vertices if x >= self.n_svg_points]
            adjacent_svg_vertices = svg_adjacency_map[i_svg_vertex]
            for iv in adjacent_internal_vertices:
                dict_svg_vertex_to_region[i_svg_vertex].add(vertex_fillmap_idx[iv])
            for iv in adjacent_svg_vertices:
                dict_svg_vertex_to_region[i_svg_vertex] = dict_svg_vertex_to_region[i_svg_vertex].union(
                    dict_svg_vertex_to_region[iv]
                )
        print("======================\n\n")
        return dict_svg_vertex_to_region

    def build_region_to_vertex_dicts(
            self,
            fillmap,
            svg_edges: np.array,
            plot=False,
            name="test",
            saveto=pathlib.Path("reports/"),
    ):
        """
        builds trapped region idx to internal vertices idx and junction idx correspondence dicts
        :param svg_edges: if not None, use to construct high-valence svg vertices
        :param saveto:
        :param fillmap: int np.array, contains trapped ball idx for each pixel
        :return: tuple(dict_region_to_internal_vertices_idx, dict_region_to_junction_vertices_idx)
        """
        svg_vertex_to_region = self.build_svg_vertex_to_region_dict(fillmap=fillmap, svg_edges=svg_edges)
        adjacency_list = igl.adjacency_list(self.faces)
        vertex_fillmap_idx = self.get_vertex_type_from_fillmap(fillmap)
        n_regions = np.max(fillmap) + 1
        dict_region_to_internal_vertices_idx = dict({
            i: np.array([], dtype=int)
            for i in range(n_regions)
        })
        dict_region_to_junction_vertices_idx = dict({
            i: np.array([], dtype=int)
            for i in range(n_regions)
        })
        dict_vertex_to_region = dict({
            i: list()
            for i in range(self.vertices.shape[0])
        })
        pure_junction_vertices = list()
        free_junction_vertices = list()
        all_junction_vertices = list()
        set_of_high_degree_svg_junctions = set()
        if svg_edges is not None:
            set_of_high_degree_svg_junctions = self.get_high_degree_svg_vertices(svg_edges=svg_edges)
        print("high valence vertices: ", set_of_high_degree_svg_junctions)
        for i_vertex in range(len(self.vertices)):
            if vertex_fillmap_idx[i_vertex] != 0:
                if i_vertex < self.n_svg_points:
                    raise Exception(f"SVG vertex {i_vertex} has fillmap {vertex_fillmap_idx[i_vertex]} but not 0")
                # if our vertex does not lie on pixels of type 0 (stroke)
                # then it is an internal point of a patch
                dict_region_to_internal_vertices_idx[vertex_fillmap_idx[i_vertex]] = np.append(
                    dict_region_to_internal_vertices_idx[vertex_fillmap_idx[i_vertex]], i_vertex
                )
                dict_vertex_to_region[i_vertex] = [vertex_fillmap_idx[i_vertex]]
            else:
                if i_vertex > self.n_svg_points:
                    raise Exception(f"Vertex {i_vertex} has fillmap 0 but it is not from svg")
                # our vertex is on pixels of class 0 (stroke)
                set_adjacent_regions = svg_vertex_to_region[i_vertex]
                if len(set_adjacent_regions) > 2:
                    # sorted_adjacent = sorted(list(set_adjacent_regions))
                    warnings.warn(
                        f"Vertex {i_vertex} has more then 2 adjacent regions ({set_adjacent_regions})"
                    )
                    # set_adjacent_regions = {sorted_adjacent[0], sorted_adjacent[1]}
                    # print(f"Vertex {i_vertex} has more then 2 adjacent regions ({sorted_adjacent}), "
                    #       f"I will drop all except two! {set_adjacent_regions}")
                dict_vertex_to_region[i_vertex] = sorted(list(set_adjacent_regions))
                if i_vertex < self.n_svg_points:
                    all_junction_vertices.append(i_vertex)
                    for un in set_adjacent_regions:
                        dict_region_to_junction_vertices_idx[un] = np.append(
                            dict_region_to_junction_vertices_idx[un], i_vertex
                        )
                    # we can store pure junctions and store boundaries separately
                    # if len(set_adjacent_regions) > 1:
                    #     for un in set_adjacent_regions:
                    #         dict_region_to_junction_vertices_idx[un] = np.append(
                    #             dict_region_to_junction_vertices_idx[un], i_vertex
                    #         )
                    #     pure_junction_vertices.append(i_vertex)
                    # else:
                    #     un = set_adjacent_regions.pop()
                    #     dict_region_to_internal_vertices_idx[un] = np.append(
                    #         dict_region_to_internal_vertices_idx[un], i_vertex
                    #     )
                    #     free_junction_vertices.append(i_vertex)
                else:
                    raise Exception(f"svg triang vertex {i_vertex} ({self.vertices[i_vertex, 0]}, {self.vertices[i_vertex, 1]}) seems like boundary")
        print(f"Out of {self.n_svg_points} svg vertices, {len(free_junction_vertices)} are free!")

        if plot:
            for ridx in range(2, n_regions):
                internal_idx = dict_region_to_internal_vertices_idx[ridx]
                junc_idx = dict_region_to_junction_vertices_idx[ridx]
                plt.figure()
                plt.imshow(fillmap, cmap="tab20b", interpolation="nearest", vmax=19.5, vmin=-0.5)
                plt.scatter(self.vertices[[internal_idx], 0], self.vertices[[internal_idx], 1], label="internal", marker=".", s=3, edgecolors='white', linewidths=0.5)
                plt.scatter(self.vertices[[junc_idx], 0], self.vertices[[junc_idx], 1], label="junction", color='red',
                            marker="*", s=5, edgecolors='k', linewidths=0.5)
                for idx in junc_idx:
                    v1 = self.vertices[idx]
                    plt.text(v1[0], v1[1], s=f"{idx}", fontsize=1)
                plt.title(f"region {ridx}")
                # plt.axis("equal")
                plt.legend()
                plt.savefig(saveto / f"{name}_svg_triang_r{ridx}.svg")
                plt.close()
                # plt.show()
                # break

        return dict_region_to_internal_vertices_idx, dict_region_to_junction_vertices_idx, \
               np.array(all_junction_vertices), dict_vertex_to_region

    def get_vertex_class_from_segmentation(
            self,
            segm,
    ):
        vertex_class = np.ones_like(self.vertices[:, 0], dtype=int)
        for i in range(len(vertex_class)):
            v_x, v_y = self.vertices[i][0], self.vertices[i][1]
            window_size = 2
            window = np.copy(segm[
                             int(v_y - window_size):int(v_y + window_size),
                             int(v_x - window_size):int(v_x + window_size),
                             ].flatten())
            window = window[window != 0]
            if len(window) == 0:
                v_x, v_y = self.vertices[i][0], self.vertices[i][1]
                window_size = 5
                window = np.copy(segm[
                                 int(v_y - window_size):int(v_y + window_size),
                                 int(v_x - window_size):int(v_x + window_size),
                                 ].flatten())
                window = window[window != 0]
                if len(window) == 0:
                    plt.figure()
                    plt.imshow(segm, cmap="tab10", vmin=-0.5, vmax=9.5)
                    plt.scatter([v_x], [v_y], marker="*", color='r', zorder=10)
                    plt.scatter(self.vertices[:,0], self.vertices[:,1], marker=".", color='k')
                    bugname = f"bug_{np.random.randint(10000)}"
                    plt.savefig(f"logs/{bugname}.png")
                    plt.close()
                    # plt.show()
                    raise Exception(f"Vertex {i} ({v_x}, {v_y}) appears to be a background -- see logs/{bugname}")
            vertex_class[i] = scipy.stats.mode(window.flatten(), keepdims=False)[0]
        return vertex_class.astype(int)

    def interpolate_f_on_vertices(
            self,
            f_grid: np.array,
    ):
        """
        given values of any function F on a regular grid, approximate values of F on triangulation vertices
        :param f_grid: np array
        :return:
        """
        xx, yy = np.arange(f_grid.shape[0]), np.arange(f_grid.shape[1])
        f = scipy.interpolate.interp2d(yy, xx, f_grid, kind='linear')
        vertex_values = np.zeros((self.vertices.shape[0]))
        for i in range(self.vertices.shape[0]):
            vertex_values[i] = f(x=self.vertices[i, 0], y=self.vertices[i, 1])
        return vertex_values

    def export_obj(self, file_path):
        with open(file_path, "w") as f:
            for vertex in self.vertices:
                f.write(
                    f"v {vertex[0]} {vertex[1]} {vertex[2] if len(vertex)>2 else ' 1'}\n"
                )
            for face in self.faces:
                f.write(
                    f"f {face[0]+1} {face[1]+1} {face[2]+1}\n"
                )


if __name__ == "__main__":
    # tr = myTraingulation.default()
    # tr = myTraingulation.default()
    myname = "p6_shampoo_bottle_view1"

    parser = argparse.ArgumentParser(description="Eval model")
    parser.add_argument("--input", default=myname, type=str, help="pngname")

    args = parser.parse_args()
    pngname = args.input

    # itemidx = 407
    # itemangle = 288
    # triang, svg_points, svg_edges = myTraingulation.from_svg(path_to_svg="clean_svgs/372_freestyle_288_01_vector.svg")
    triang, svg_points, svg_edges, svg_paths_edges = myTraingulation.from_svg(
        # path_to_svg=f"clean_svgs/{itemidx}_freestyle_{itemangle:03d}_01_vector.svg",
        path_to_svg=f"results/{pngname}/{pngname}_clean.svg",
        # triang_flags="qYY",
        svg_sampling_distance=10,
        triang_flags='YYqpa50',
    )
    # triang.vertices /= 2
    triang.plot(show=False, faces=True, saveto=Path(f"results/{pngname}/"))

    print(triang.vertices.shape)

    data = np.load(f"results/{pngname}/npz/{pngname}_depth.npz")
    # gtimage = np.pad(gt_data["sketch"], ((8, 8), (8, 8)), constant_values=gt_data["sketch"][0, 0])
    gtimage = data["sketch"]
    image_for_trapped_ball = ((1 - gtimage) * 255).astype(np.uint8)
    maindepthimage = data["depth"]
    # depthimage = np.pad(gt_data["depth"], ((8, 8), (8, 8)), constant_values=gt_data["depth"][0, 0])
    mainfillmap = my_trapped_ball(image_for_trapped_ball, first_ball=4)
    plt.imshow(mainfillmap)
    plt.show()
    nreg = np.max(mainfillmap) + 1

    # triang.vertices += 8
    #
    # vertex_colors = triang.get_vertex_type_from_fillmap(fillmap=mainfillmap)
    # vertex_depth = triang.interpolate_f_on_vertices(f_grid=maindepthimage)
    #
    # vertices3d = np.hstack(
    #     (
    #         triang.get_vertices_to_camera_coords(imshape=(512, 512)),
    #         4 - vertex_depth.reshape(-1, 1)
    #     )
    # )
    # m3d = MyMesh3D(
    #     vertices=vertices3d,
    #     vertex_markers=triang.vertex_markers,
    #     faces=triang.faces,
    #     holes=triang.holes,
    # )
    # m3d.vertices_to_camera_coords(imsize=416)
    # m3d.make_back_surface()
    # m3d.export_obj(file_path=f"results/{pngname}/{pngname}_predicted.obj", front_surface_count=len(triang.faces))

    # plt.figure()
    # plt.imshow(fillmap, cmap="tab20", interpolation="nearest", vmax=19.5, vmin=-0.5)
    # # plt.scatter(triang.vertices[triang.n_svg_points:, 0], triang.vertices[triang.n_svg_points:, 1], c=vertex_colors[triang.n_svg_points:], cmap="tab20", zorder=5, vmax=19.5, vmin=-0.5, edgecolors='k')
    # plt.scatter(triang.vertices[:, 0], triang.vertices[:, 1], c=vertex_colors[:], cmap="tab20", zorder=5, vmax=19.5, vmin=-0.5, edgecolors='k')
    # plt.savefig("reports/triang_fillmap.svg")
    # plt.show()

    # plt.figure()
    # plt.imshow(depthimage)
    # # triang.plot(faces=True, show=False)
    # plt.scatter(triang.vertices[:, 0], triang.vertices[:, 1], c=vertex_depth, zorder=5, edgecolors='k')

    # plt.show()

    region_to_internal, region_to_junction, pure_junctions, dict_vertex_to_regions = triang.build_region_to_vertex_dicts(fillmap=mainfillmap, plot=True, svg_edges=svg_edges)
