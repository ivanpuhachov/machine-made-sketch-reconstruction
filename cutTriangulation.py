import numpy as np
import matplotlib.pyplot as plt
import igl
from util_trappedball_fill import my_trapped_ball
import pathlib
from myTriangulation import myTraingulation
from copy import deepcopy
from collections import defaultdict
import warnings
import cv2
import matplotlib


class Chain:
    def __init__(
            self,
            vertex_tuple,
            edges_tuple,
    ):
        self.vertex_tuple = vertex_tuple
        self.edges_tuple = edges_tuple
        assert len(edges_tuple) + 1 == len(vertex_tuple)

    def flip_(self):
        """
        inplace flipping of chain sequence
        :return:
        """
        self.vertex_tuple = self.vertex_tuple[::-1]
        self.edges_tuple = self.edges_tuple[::-1]

    def update_by_edges_(self, new_edge_to_vertex_map):
        print(f"should we update chain {self.vertex_tuple} - {self.edges_tuple}?")
        new_vertex_list = list()
        edge_to_vertex_set = lambda x: set(new_edge_to_vertex_map[self.edges_tuple[x]])
        # check that first vertex is ok
        if self.vertex_tuple[0] not in edge_to_vertex_set(0):
            if self.vertex_tuple[1] not in edge_to_vertex_set(0):
                raise Exception(f"vertices {self.vertex_tuple[:2]} have nothing to do with new edge btwn {edge_to_vertex_set(0)}")
            edge0vertices = edge_to_vertex_set(0)
            edge0vertices.remove(self.vertex_tuple[1])
            new_vertex0_id = edge0vertices.pop()
            print(f"chain updated vertex {self.vertex_tuple[0]} to be {new_vertex0_id}")
            new_vertex_list.append(new_vertex0_id)
        else:
            new_vertex_list.append(self.vertex_tuple[0])
        for i_edge in range(len(self.edges_tuple) - 1):
            this_edge_vertices = edge_to_vertex_set(i_edge)
            next_edge_vertices = edge_to_vertex_set(i_edge + 1)
            set_edge_intersect = this_edge_vertices.intersection(next_edge_vertices)
            if len(set_edge_intersect) == 0:
                raise Exception(f"Edges {self.edges_tuple[i_edge]} ({this_edge_vertices}) and {self.edges_tuple[i_edge + 1]} ({next_edge_vertices}) have nothing in common")
            new_vertex_list.append(set_edge_intersect.pop())
        last_edge = edge_to_vertex_set(-1)
        last_edge.remove(new_vertex_list[-1])
        new_vertex_list.append(last_edge.pop())
        if tuple(new_vertex_list) != self.vertex_tuple:
            print(f"Updating chain \n{self.vertex_tuple}\n to \n{new_vertex_list}")
        self.vertex_tuple = tuple(new_vertex_list)

    def get_common_region(self, vertex_to_region: dict):
        # traverse the whole chain and take common region idx
        set_of_common_regions = set(vertex_to_region[self.vertex_tuple[0]])
        for chain_vertex in self.vertex_tuple:
            set_of_common_regions = set_of_common_regions.intersection(
                set(vertex_to_region[chain_vertex])
            )
        return set_of_common_regions

    def merge(self, other):
        if self.vertex_tuple[-1] == other.vertex_tuple[0]:
            return Chain(
                vertex_tuple=self.vertex_tuple + other.vertex_tuple[1:],
                edges_tuple=self.edges_tuple + other.edges_tuple,
            )
        if self.vertex_tuple[-1] == other.vertex_tuple[-1]:
            other.flip_()
            return self.merge(other)

        if self.vertex_tuple[0] == other.vertex_tuple[-1]:
            self.flip_()
            other.flip_()
            return self.merge(other)

        if self.vertex_tuple[0] == other.vertex_tuple[0]:
            self.flip_()
            return self.merge(other)

        raise Exception(f"Can't merge these chains! {self.vertex_tuple} -- {other.vertex_tuple}")


def get_third_vertex(
        face_id: int,
        faces: np.array,
        v1: int,
        v2: int,
):
    """
    return index of the third vertex from a face
    :param face_id:
    :param faces:
    :param v1:
    :param v2:
    :return:
    """
    this_face = faces[face_id]
    set_vertices = set(this_face)
    if (v1 not in set_vertices) or (v2 not in set_vertices):
        raise Exception(f"get_third_vertex: face {face_id} has vertices {this_face}, but not {v1} {v2}")
    set_vertices.remove(v1)
    set_vertices.remove(v2)
    return set_vertices.pop()


def get_next_edge_vertex(
        edge_to_vertices: np.array,
        edge_id: int,
        vertex_id: int,
):
    this_edge = set(edge_to_vertices[edge_id])
    if vertex_id not in this_edge:
        raise Exception(f"get_next_edge_vertex: edge {edge_id} has vertices {this_edge} but not {vertex_id}")
    this_edge.remove(vertex_id)
    return this_edge.pop()

def get_next_edge_id(
        edge_id: int,
        vertex_id: int,
        face_id: int,
        faces: np.array,
        mat_edges_to_vertices: np.array,
        map_vertices_to_edges: dict,
):
    """
    for a triangle ABC vertex B and edge AB return edge id for BC
    :param edge_id: edge we are considering
    :param vertex_id: vertex around which we rotate
    :param face_id: face we are in right now
    :param faces:
    :param mat_edges_to_vertices:
    :param map_vertices_to_edges:
    :return: edge id for
    """
    this_face = faces[face_id]
    set_vertices = set(this_face)
    this_edge_vertices = mat_edges_to_vertices[edge_id]
    set_vertices.remove(this_edge_vertices[0])
    set_vertices.remove(this_edge_vertices[1])
    third_vertex = set_vertices.pop()
    next_edge_id = set(map_vertices_to_edges[vertex_id]).intersection(set(map_vertices_to_edges[third_vertex])).pop()
    return next_edge_id


def get_face_across(
        this_face_id: int,
        edge_id: int,
        mat_edges_to_faces: np.array,
):
    faces_near_edge = set(mat_edges_to_faces[edge_id])
    faces_near_edge.remove(this_face_id)
    return faces_near_edge.pop()


def get_edges_from_face(
        this_face_id: int,
        faces: np.array,
        map_vertices_to_edges: dict,
):
    verA, verB, verC = faces[this_face_id, 0], faces[this_face_id, 1], faces[this_face_id, 2],
    setA = set(map_vertices_to_edges[verA])
    setB = set(map_vertices_to_edges[verB])
    setC = set(map_vertices_to_edges[verC])
    a, b, c = setB.intersection(setC).pop(), setA.intersection(setC).pop(), setA.intersection(setB).pop()
    return a, b, c


def cutMeshAlongChain(
        cutchain: Chain,
        triangulation: myTraingulation,
        mat_edges_to_vertices: np.array,
        mat_edges_to_faces: np.array,
        map_vertices_to_edges: dict,
        vertex_to_region: dict,
):
    if len(cutchain.edges_tuple) == 1:
        print("- Can't cut along chain with 2 vertices!")
        # TODO: we can cut this when one vertex is on the boundary!
        return 0, triangulation.vertices, triangulation.faces, triangulation.vertex_markers, vertex_to_region

    if (triangulation.vertex_markers[cutchain.vertex_tuple[0]] != 1) and (triangulation.vertex_markers[cutchain.vertex_tuple[-1]] == 1):
        print("Flip the chain!")
        cutchain.flip_()
        print("newchain: ", cutchain.vertex_tuple)

    # here we look for a region that needs to be separated from others
    first_edge = cutchain.edges_tuple[0]
    left_side_face = min(mat_edges_to_faces[first_edge])
    right_side_face = max(mat_edges_to_faces[first_edge])
    vertex_on_left_side_face = get_third_vertex(
            face_id=left_side_face,
            faces=triangulation.faces,
            v1=cutchain.vertex_tuple[0],
            v2=cutchain.vertex_tuple[1],
        )
    vertex_on_right_side_face = get_third_vertex(
        face_id=right_side_face,
        faces=triangulation.faces,
        v1=cutchain.vertex_tuple[0],
        v2=cutchain.vertex_tuple[1],
    )
    if left_side_face == -1:
        raise Exception("Can't cut along boundary edge!")

    if vertex_on_left_side_face < triangulation.n_svg_points:
        # we prefer having left side vertex free (internal), not from svg sampled points (junction)
        left_side_face, right_side_face = right_side_face, left_side_face
        vertex_on_left_side_face, vertex_on_right_side_face = vertex_on_right_side_face, vertex_on_left_side_face

    set_of_common_regions = cutchain.get_common_region(vertex_to_region=vertex_to_region)
    print("Common regions along chain vertices: ", set_of_common_regions)

    left_side_region = vertex_to_region[vertex_on_left_side_face][0]  # contains the region we separate with the cut
    right_side_region = vertex_to_region[vertex_on_right_side_face][0]  # auxilary region, region on the opposite side of the cut

    if triangulation.vertex_markers[cutchain.vertex_tuple[0]] == 1:
        # check that we are separating region that lies to the boundary
        set_boundary_regions_here = set()
        for e in map_vertices_to_edges[cutchain.vertex_tuple[0]]:
            if (mat_edges_to_faces[e][0] == -1) or (mat_edges_to_faces[e][1] == -1):
                other_vertex = get_next_edge_vertex(
                    edge_to_vertices=mat_edges_to_vertices,
                    edge_id=e,
                    vertex_id=cutchain.vertex_tuple[0]
                )
                set_boundary_regions_here = set_boundary_regions_here.union(set(vertex_to_region[other_vertex]))
        print(f"Boundary regions here are: {set_boundary_regions_here}")
        better_common_regions = set_boundary_regions_here.intersection(set_of_common_regions)
        if len(better_common_regions) > 0:
            set_of_common_regions = better_common_regions
        else:
            warnings.warn(f"Can't pick a boundary region from common regions {set_of_common_regions}, cut may fail")

    if not left_side_region in set_of_common_regions:
        ls_regions = set(vertex_to_region[vertex_on_left_side_face])
        candidates = ls_regions.intersection(set_of_common_regions)
        if len(candidates) > 0:
            left_side_region = candidates.pop()
        else:
            rs_regions = set(vertex_to_region[vertex_on_right_side_face])
            candidates = rs_regions.intersection(set_of_common_regions)
            if len(candidates) > 0:
                left_side_region = candidates.pop()
                left_side_face = right_side_face
            else:
                raise Exception(f"I can't assign a region to cut along chain {cutchain.vertex_tuple} with common regions ({set_of_common_regions})")

    print(f"I'm cutting chain with {len(cutchain.edges_tuple)} edges: ", cutchain.edges_tuple)
    print(f"and {len(cutchain.vertex_tuple)} vertices: ", cutchain.vertex_tuple)
    print(f"to separate region {left_side_region}")
    faces_to_cut_left = [left_side_face]
    vertices_to_cut_on = []

    boundary_regions_on_the_left = set()  # traverse around chain ends to keep regions we see
    if triangulation.vertex_markers[cutchain.vertex_tuple[0]] == 1:
        vertices_to_cut_on.append(cutchain.vertex_tuple[0])
        print(f"First chain vertex {cutchain.vertex_tuple[0]} lies on the boundary!")
        # then we need to traverse faces attached to first chain vertex until we reach the boundary
        current_edge = cutchain.edges_tuple[0]
        traversing_around_vertex_id = cutchain.vertex_tuple[0]
        root_left_side_face = left_side_face
        left_side_edge = get_next_edge_id(
            edge_id=current_edge,
            vertex_id=traversing_around_vertex_id,
            face_id=root_left_side_face,
            faces=triangulation.faces,
            map_vertices_to_edges=map_vertices_to_edges,
            mat_edges_to_vertices=mat_edges_to_vertices,
        )
        left_side_temp_vertex = get_next_edge_vertex(
            edge_to_vertices=mat_edges_to_vertices,
            edge_id=left_side_edge,
            vertex_id=traversing_around_vertex_id,
        )
        boundary_regions_on_the_left = boundary_regions_on_the_left.union(set(vertex_to_region[left_side_temp_vertex]))
        while (mat_edges_to_faces[left_side_edge][0] != -1) and (mat_edges_to_faces[left_side_edge][1] != -1):
            root_left_side_face = get_face_across(
                this_face_id=root_left_side_face,
                edge_id=left_side_edge,
                mat_edges_to_faces=mat_edges_to_faces,
            )
            left_side_edge = get_next_edge_id(
                edge_id=left_side_edge,
                vertex_id=traversing_around_vertex_id,
                face_id=root_left_side_face,
                faces=triangulation.faces,
                map_vertices_to_edges=map_vertices_to_edges,
                mat_edges_to_vertices=mat_edges_to_vertices,
            )
            faces_to_cut_left.append(root_left_side_face)
            print("Found root face to cut: ", root_left_side_face)
            left_side_temp_vertex = get_next_edge_vertex(
                edge_to_vertices=mat_edges_to_vertices,
                edge_id=left_side_edge,
                vertex_id=traversing_around_vertex_id,
            )
            boundary_regions_on_the_left = boundary_regions_on_the_left.union(set(vertex_to_region[left_side_temp_vertex]))
        print(f"Boundary regions found (chain start): {boundary_regions_on_the_left}")

    for i in range(len(cutchain.edges_tuple) - 1):
        current_edge = cutchain.edges_tuple[i]
        next_edge = cutchain.edges_tuple[i+1]
        vertex_to_duplicate = cutchain.vertex_tuple[i+1]
        vertices_to_cut_on.append(vertex_to_duplicate)
        next_chain_vertex = cutchain.vertex_tuple[i+2]
        assert current_edge in map_vertices_to_edges[vertex_to_duplicate]
        assert next_edge in map_vertices_to_edges[vertex_to_duplicate]
        # traversing the face
        left_side_edge = get_next_edge_id(
            edge_id=current_edge,
            vertex_id=vertex_to_duplicate,
            face_id=left_side_face,
            faces=triangulation.faces,
            map_vertices_to_edges=map_vertices_to_edges,
            mat_edges_to_vertices=mat_edges_to_vertices,
        )
        while left_side_edge != next_edge:
            # traverse faces until we reach the next edge
            left_side_face = get_face_across(
                this_face_id=left_side_face,
                edge_id=left_side_edge,
                mat_edges_to_faces=mat_edges_to_faces,
            )
            left_side_edge = get_next_edge_id(
                edge_id=left_side_edge,
                vertex_id=vertex_to_duplicate,
                face_id=left_side_face,
                faces=triangulation.faces,
                map_vertices_to_edges=map_vertices_to_edges,
                mat_edges_to_vertices=mat_edges_to_vertices,
            )
            faces_to_cut_left.append(left_side_face)
            # print(f"Append face {left_side_face}: {faces[left_side_face]}")

    if triangulation.vertex_markers[cutchain.vertex_tuple[-1]] == 1:
        vertices_to_cut_on.append(cutchain.vertex_tuple[-1])
        print("Last chain vertex lies on the boundary!")
        # then we need to traverse faces attached to first chain vertex until we reach the boundary
        current_edge = cutchain.edges_tuple[-1]
        traversing_around_vertex_id = cutchain.vertex_tuple[-1]
        root_left_side_face = faces_to_cut_left[-1]
        left_side_edge = get_next_edge_id(
            edge_id=current_edge,
            vertex_id=traversing_around_vertex_id,
            face_id=root_left_side_face,
            faces=triangulation.faces,
            map_vertices_to_edges=map_vertices_to_edges,
            mat_edges_to_vertices=mat_edges_to_vertices,
        )
        left_side_temp_vertex = get_next_edge_vertex(
            edge_to_vertices=mat_edges_to_vertices,
            edge_id=left_side_edge,
            vertex_id=traversing_around_vertex_id,
        )
        boundary_regions_on_the_left = boundary_regions_on_the_left.union(set(vertex_to_region[left_side_temp_vertex]))
        while (mat_edges_to_faces[left_side_edge][0] != -1) and (mat_edges_to_faces[left_side_edge][1] != -1):
            root_left_side_face = get_face_across(
                this_face_id=root_left_side_face,
                edge_id=left_side_edge,
                mat_edges_to_faces=mat_edges_to_faces,
            )
            left_side_edge = get_next_edge_id(
                edge_id=left_side_edge,
                vertex_id=traversing_around_vertex_id,
                face_id=root_left_side_face,
                faces=triangulation.faces,
                map_vertices_to_edges=map_vertices_to_edges,
                mat_edges_to_vertices=mat_edges_to_vertices,
            )
            faces_to_cut_left.append(root_left_side_face)
            print("Found root face to cut: ", root_left_side_face)
            left_side_temp_vertex = get_next_edge_vertex(
                edge_to_vertices=mat_edges_to_vertices,
                edge_id=left_side_edge,
                vertex_id=traversing_around_vertex_id,
            )
            boundary_regions_on_the_left = boundary_regions_on_the_left.union(set(vertex_to_region[left_side_temp_vertex]))
        print(f"Boundary regions found (chain end): {boundary_regions_on_the_left}")

    print("Faces to cut (left): ", faces_to_cut_left)
    print("Vertices to duplicate: ", vertices_to_cut_on)
    print("Edges to cut along: ", cutchain.edges_tuple)
    # now copy vertices (we so not duplicate start and end vertices)
    n_new_vertices = len(vertices_to_cut_on)
    vertices_to_copy = triangulation.vertices[np.array(vertices_to_cut_on), :]
    newvertices = np.vstack((triangulation.vertices, vertices_to_copy))

    # build mapping from old idx to duplicated vertices idx
    newvertex_map = {
        i: i for i in range(len(triangulation.vertices))
    }  # map vertices to update left faces
    for i in range(0, len(vertices_to_cut_on)):
        cv = vertices_to_cut_on[i]
        newvertex_map[cv] = len(triangulation.vertices) + i

    # build set of edges (on the left side) that might need to be updated with new vertex id
    set_edges_on_the_left = set()
    set_faces_on_the_left = set(faces_to_cut_left)
    for x in cutchain.vertex_tuple:
        for e in map_vertices_to_edges[x]:
            if (mat_edges_to_faces[e][0] in set_faces_on_the_left) or (mat_edges_to_faces[e][1] in set_faces_on_the_left):
                set_edges_on_the_left.add(e)

    set_edges_on_the_left = set_edges_on_the_left - set(cutchain.edges_tuple)
    print("These edges need to be updated with new vertex id: ", set_edges_on_the_left)

    # create new edges and update edges_to_vertices
    newvertices_idx = np.arange(len(vertices_to_cut_on)) + len(triangulation.vertices)
    duplicated_edges_idx = np.arange(len(cutchain.edges_tuple)) + len(mat_edges_to_vertices)
    duplicated_edges_to_vertices = list()
    for e in cutchain.edges_tuple:
        old_edge = mat_edges_to_vertices[e]
        new_edge = [newvertex_map[x] for x in old_edge]
        duplicated_edges_to_vertices.append(new_edge)
    duplicated_edges_to_vertices = np.array(duplicated_edges_to_vertices)
    print(f"Current mat_edges_to_vertices shape: {mat_edges_to_vertices.shape}")
    print(f"Updating edges with these new {duplicated_edges_to_vertices.shape}")
    print(duplicated_edges_to_vertices)
    for e in set_edges_on_the_left:
        old_edge = mat_edges_to_vertices[e]
        new_edge = [newvertex_map[x] for x in old_edge]
        mat_edges_to_vertices[e] = new_edge
    mat_edges_to_vertices = np.vstack(
        (
            mat_edges_to_vertices,
            duplicated_edges_to_vertices,
        )
    )

    # update and build new edges_to_faces
    duplicated_edges_to_faces = list()  # will contain EF for newly created edges
    set_faces_to_cut_left = set(faces_to_cut_left)
    for e in cutchain.edges_tuple:
        old_edge_f = mat_edges_to_faces[e]
        # new edge (left) has (left_face, -1)
        left_f = [x if x in set_faces_to_cut_left else -1 for x in old_edge_f]
        duplicated_edges_to_faces.append(left_f)
        # old edge (right) has now (-1, right_face)
        right_f = [x if x not in set_faces_to_cut_left else -1 for x in old_edge_f]
        mat_edges_to_faces[e] = right_f
    duplicated_edges_to_faces = np.array(duplicated_edges_to_faces)
    print(f"Updating mat_edges_to_faces ({mat_edges_to_faces.shape}) with {duplicated_edges_to_faces.shape}")
    mat_edges_to_faces = np.vstack(
        (
            mat_edges_to_faces,
            duplicated_edges_to_faces,
        )
    )
    # build correctly assigned faces with newvertex_map
    # we do not create new face, we simply need to update existing faces vertices idx
    newfaces = np.copy(triangulation.faces)
    for f in faces_to_cut_left:
        oldface = triangulation.faces[f]
        newface = np.array([newvertex_map[x] for x in oldface])
        newfaces[f] = newface
        # print(oldface, " -> ", newface)

    # build boundary markers
    newboundarymarkers = np.hstack(
        (
            triangulation.vertex_markers,
            np.ones(n_new_vertices),
        )
    )
    newboundarymarkers[np.array(cutchain.vertex_tuple)] = 1

    # update vertex_to_region
    vertices_that_went_left = list()
    vertices_that_went_right = list()
    for chain_vertex in vertices_to_cut_on:
        old_regions = set(vertex_to_region[chain_vertex])
        regions_right = old_regions - {left_side_region}
        regions_left = {left_side_region}
        if triangulation.vertex_markers[chain_vertex] == 1:
            # for a boundary vertex we need to carefully assign regions
            regions_right = old_regions - boundary_regions_on_the_left
            regions_left = old_regions.intersection(boundary_regions_on_the_left)
        vertex_to_region[chain_vertex] = list(regions_right)
        vertices_that_went_right.append(chain_vertex)
        if len(vertex_to_region[chain_vertex]) == 0:
            raise Exception(f"Empty correspondace for vertex {chain_vertex}")
        this_vertex_duplicate_id = newvertex_map[chain_vertex]
        print(f"vertex {chain_vertex} (r: {regions_right}) copied to {this_vertex_duplicate_id} (r: {regions_left})")
        vertex_to_region[this_vertex_duplicate_id] = list(regions_left)
        vertices_that_went_left.append(this_vertex_duplicate_id)

    # update and build new vertices_to_edges
    set_edges_to_cut_left = set()
    for f in faces_to_cut_left:
        edges_of_this_face = get_edges_from_face(this_face_id=f, faces=triangulation.faces,
                                                 map_vertices_to_edges=map_vertices_to_edges)
        set_edges_to_cut_left.update(edges_of_this_face)
    set_edges_to_cut_left = set_edges_to_cut_left - set(cutchain.edges_tuple)

    print(f"Updating vertex to edge correspondances")
    for i_vertex in range(len(cutchain.vertex_tuple)):
        i_edge = i_vertex
        if i_vertex == len(cutchain.vertex_tuple) - 1:
            i_edge = i_vertex - 1
        cut_edge = cutchain.edges_tuple[i_edge]
        dup_edge = duplicated_edges_idx[i_edge]
        vertex1 = cutchain.vertex_tuple[i_vertex]
        vertex1_dup = newvertex_map[vertex1]
        vertex1_edges = map_vertices_to_edges[vertex1]
        print(f"vertex {vertex1} -> edges {map_vertices_to_edges[vertex1]}")
        if vertex1 == vertex1_dup:
            # this happens at cutchain ends
            if (i_vertex != 0) and (i_vertex != len(cutchain.vertex_tuple) - 1):
                raise Exception(f"Vertex {vertex1} is not at the chain end, yet it was not duplicated?")
            map_vertices_to_edges[vertex1].append(dup_edge)
            print(f"vertex {vertex1} now -> edges {map_vertices_to_edges[vertex1]}")
        else:
            inner_edges_left = set(vertex1_edges).intersection(set_edges_to_cut_left)
            inner_edges_right = set(vertex1_edges) - set_edges_to_cut_left
            map_vertices_to_edges[vertex1] = list(inner_edges_right)
            map_vertices_to_edges[vertex1_dup] = list(inner_edges_left)
            map_vertices_to_edges[vertex1_dup].append(dup_edge)
            if (i_edge > 0) and (i_edge < len(cutchain.edges_tuple) - 1):
                map_vertices_to_edges[vertex1_dup].append(duplicated_edges_idx[i_edge-1])
            print(f"vertex {vertex1} -> edges {map_vertices_to_edges[vertex1]}")
            print(f"vertex {vertex1_dup} -> edges {map_vertices_to_edges[vertex1_dup]}")

    return faces_to_cut_left, newvertices, newfaces, newboundarymarkers, vertex_to_region, \
           (vertices_that_went_left, vertices_that_went_right), \
           mat_edges_to_vertices, mat_edges_to_faces, map_vertices_to_edges


def extract_chain(
        vertex_map_cut_edge: dict,
        edges_to_vertices: np.array,
        vertex_to_regions: dict,
        high_valence_vertices: set,
        vertex_boundary_markers: np.array,
):
    start_vertex = -1
    start_edge = -1
    for iv, ies in vertex_map_cut_edge.items():
        if len(ies) == 1:
            start_vertex = iv
            start_edge = ies[0]
            print(f"\n- found start {start_vertex} ({start_edge}), {ies}")
            break
    if (start_vertex == -1) or (start_edge == -1):
        raise Exception("Can't find a place to start")  # this typically means we hit a loop in edge graphs, #TODO: fix this
    chain_vertex_list = [start_vertex]
    chain_edges_list = [start_edge]
    # remove this edge from map as we don't want to use it in the future
    continue_is_possible = True
    common_regions_along_chain = set(vertex_to_regions[start_vertex])
    print(f"Initial regions for starting vertex {start_vertex}: {common_regions_along_chain}")
    while continue_is_possible:
        # for chain A - B - C, current_edge is 1 and current_vertex is A
        # we want to add vertex B and, potentially, edge 2
        current_edge = chain_edges_list[-1]
        current_vertex = chain_vertex_list[-1]
        vertex_map_cut_edge[current_vertex].remove(current_edge)
        # find vertex on the other end of the current edge (B)
        vertex_to_append = get_next_edge_vertex(edges_to_vertices, current_edge, current_vertex)
        # add this vertex to list
        chain_vertex_list.append(vertex_to_append)
        common_regions_along_chain = common_regions_along_chain.intersection(set(vertex_to_regions[vertex_to_append]))
        # print(f"edge {current_edge} -> vertex {vertex_to_append}")
        # remove the edge from map
        vertex_map_cut_edge[vertex_to_append].remove(current_edge)
        # check if we can continue the chain
        continue_is_possible = False
        if vertex_boundary_markers[vertex_to_append] == 1:
            print(f"We reached boundary vertex {vertex_to_append}, stop tracing")
            break
        if vertex_to_append in high_valence_vertices:
            print(f"We reached high-valence vertex {vertex_to_append}, searching for continuation")
        else:
            for candidate_edge_to_append in vertex_map_cut_edge[vertex_to_append]:
                # if there is exactly one edge coming out of vertex B
                # candidate_edge_to_append = vertex_map_cut_edge[vertex_to_append][0]
                candidate_vertex_to_append = get_next_edge_vertex(edges_to_vertices, candidate_edge_to_append, vertex_to_append)
                # check that correspondance holds
                if len(common_regions_along_chain.intersection(set(vertex_to_regions[candidate_vertex_to_append]))) > 0:
                    chain_edges_list.append(candidate_edge_to_append)
                    continue_is_possible = True
                    break
        #     else:
        #         print(f"Lost common region {common_regions_along_chain}, can't continue to {candidate_vertex_to_append} with {vertex_to_regions[candidate_vertex_to_append]}")
        #         continue_is_possible = False
        # else:
        #     print(f"High-valence vertex {vertex_to_append}, can't continue ({vertex_map_cut_edge[vertex_to_append]})")
        #     continue_is_possible = False

        if vertex_to_append in high_valence_vertices:
            if continue_is_possible:
                print(f"Found continuation edge {chain_edges_list[-1]}")
            else:
                print("No continuation found")
                # print(f"Lost common region {common_regions_along_chain}, can't continue to {candidate_vertex_to_append} with {vertex_to_regions[candidate_vertex_to_append]}")
                continue_is_possible = False
        # else:
        #     print(f"High-valence vertex {vertex_to_append}, can't continue ({vertex_map_cut_edge[vertex_to_append]})")
        #     continue_is_possible = False
    return Chain(
        vertex_tuple=tuple(chain_vertex_list),
        edges_tuple=tuple(chain_edges_list),
    )


def depthGradient(
    depth_values,
):
    # d0, d1 = np.gradient(depth_values, 1 / depth_values.shape[0], edge_order=2)
    grad_x = cv2.Sobel(depth_values, cv2.CV_16S, 1, 0, ksize=5, scale=depth_values.shape[0] / 100, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(depth_values, cv2.CV_16S, 0, 1, ksize=5, scale=depth_values.shape[0] / 100, delta=0, borderType=cv2.BORDER_DEFAULT)
    # return d0, d1
    return grad_y, grad_x


def cut_condition(
        triang: myTraingulation,
        depthimage: np.array,
        vertex_id: int,
        depth_threshold=0.3,
):
    """
    return True if we should cut on this vertex
    :return:
    """
    v_x, v_y = triang.vertices[vertex_id][0], triang.vertices[vertex_id][1]
    window_size = 6
    window = np.copy(depthimage[
                     int(v_y - window_size):int(v_y + window_size),
                     int(v_x - window_size):int(v_x + window_size),
                     ].flatten())

    if np.max(window) - np.min(window) > depth_threshold:
        return True
    return False


def findEdgesToCut(
        triang: myTraingulation,
        svg_edges,
        svg_points,
        depthimage,
        # fillmap,
        EV,
        EF,
        dict_vertex_to_region: dict,
        report=True,
        saveto=pathlib.Path("reports/"),
):
    # vertex_colors = triang.get_vertex_type_from_fillmap(fillmap=fillmap)
    vertex_depth = triang.interpolate_f_on_vertices(f_grid=depthimage)
    ax_y_depth_gradient, ax_x_depth_gradient = depthGradient(depth_values=depthimage)
    depth_gradient = np.stack(
        (ax_x_depth_gradient, ax_y_depth_gradient),
    )
    gradnorm = np.linalg.norm(depth_gradient, axis=0)
    vertex_depth_grad_x = triang.interpolate_f_on_vertices(f_grid=ax_x_depth_gradient)
    vertex_depth_grad_y = triang.interpolate_f_on_vertices(f_grid=-ax_y_depth_gradient)
    vertex_grad = np.stack(
        (vertex_depth_grad_x, vertex_depth_grad_y),
    )
    vertex_gradnorm = np.linalg.norm(vertex_grad, axis=0)
    grad_threshold = 25
    grad_mask = vertex_gradnorm[:triang.n_svg_points] > grad_threshold

    svg_cut_edge_idx = list()  # contains a list of all edges to be cut
    dict_vertex_to_cut_edge = {
        i: list()
        for i in range(triang.n_svg_points)
    }  # stores list of edges to cut around this vertex
    svg_vertex_to_edges = {i: list() for i in range(triang.n_svg_points)}
    for i in range(len(EV)):
        e = EV[i]
        if (e[0] < triang.n_svg_points) and (e[1] < triang.n_svg_points):
            svg_vertex_to_edges[e[0]].append(i)
            svg_vertex_to_edges[e[1]].append(i)

    def cut_condition_edge(
            vertex_id1: int,
            vertex_id2: int,
            grad_norm_thr=10,
    ):
        return (vertex_gradnorm[vertex_id1] > grad_norm_thr) or (vertex_gradnorm[vertex_id2] > grad_norm_thr)

    # iterate through svg edges
    for i in range(len(svg_edges)):
        orig_edge = svg_edges[i]
        v1 = orig_edge[0]
        v2 = orig_edge[1]
        if (cut_condition(triang=triang, depthimage=depthimage, vertex_id=v1) or (cut_condition(triang=triang, depthimage=depthimage, vertex_id=v2))) or (cut_condition_edge(v1, v2)):
            # add this edge to cut edge set
            igl_edge_set = set(svg_vertex_to_edges[v1]).intersection(
                set(svg_vertex_to_edges[v2])
            )
            if len(igl_edge_set) == 0:
                print(svg_edges)
                print(f"svg edges : {svg_edges.shape}")
                print(f"svg points : {svg_points.shape}")
                print(EV)
                raise Exception(f"No common triangulation edge between {v1} {v2}, but specified in SVG sampling!")
            if len(igl_edge_set) > 1:
                raise Exception(f"More than one common edge between {v1} {v2}")
            igl_edge = igl_edge_set.pop()
            if (EF[igl_edge][0] != -1) and (EF[igl_edge][1] != -1):
                # otherwise, it is a boundary edge
                if len(dict_vertex_to_region[v1]) == 1:
                    warnings.warn(f"cut is impossible: vertex {v1} is not boundary but is only connected to one region {dict_vertex_to_region[v1]}")
                    # raise Exception("STOP")
                    continue
                if len(dict_vertex_to_region[v2]) == 1:
                    warnings.warn(f"cut is impossible: vertex {v2} is not boundary but is only connected to one region {dict_vertex_to_region[v2]}")
                    # raise Exception("STOP")
                    continue
                svg_cut_edge_idx.append(igl_edge)
                dict_vertex_to_cut_edge[v1].append(igl_edge)
                dict_vertex_to_cut_edge[v2].append(igl_edge)
                # print(f"Cut edge {igl_edge} ({v1} - {v2})")
    if report:
        plt.figure()
        plt.imshow(depthimage)
        plt.colorbar()
        plt.title("Vertices where jump was detected locally")
        for i in range(triang.n_svg_points):
            if cut_condition(triang, depthimage, vertex_id=i):
                plt.scatter([triang.vertices[i, 0]], triang.vertices[i, 1], color='red', edgecolors='k', s=20,
                            linewidths=0.4)
            else:
                plt.scatter([triang.vertices[i, 0]], triang.vertices[i, 1], color='white', edgecolors='gray', s=10,
                            linewidths=0.2)
        plt.savefig(saveto / "verts.svg")
        plt.close()

        plt.figure()
        for i in range(len(svg_edges)):
            orig_edge = svg_edges[i]
            v1 = orig_edge[0]
            v2 = orig_edge[1]
            plt.plot([triang.vertices[v2, 0], triang.vertices[v1, 0]],
                     [triang.vertices[v2, 1], triang.vertices[v1, 1]],
                     color='gray', linewidth=1.5)
        plt.imshow(depthimage)
        plt.colorbar()
        # plt.imshow(gradnorm, cmap="plasma", zorder=1)
        # plt.colorbar()
        # triang.plot(faces=True, show=False)
        plt.quiver(
            triang.vertices[:triang.n_svg_points, 0][grad_mask],
            triang.vertices[:triang.n_svg_points, 1][grad_mask],
            vertex_depth_grad_x[:triang.n_svg_points][grad_mask],
            vertex_depth_grad_y[:triang.n_svg_points][grad_mask],
            width=0.01, scale=400, color='red', zorder=2)
        plt.title("Triang with gradients")
        plt.savefig(saveto / "grads.svg")
        plt.close()

        plt.figure()
        plt.imshow(depthimage)
        triang.plot(faces=True, show=False)
        plt.scatter(triang.vertices[:triang.n_svg_points, 0], triang.vertices[:triang.n_svg_points, 1],
                    color="red", zorder=5, edgecolors='k', s=1, linewidths=0.1)
        # plt.scatter(triang.vertices[triang.n_svg_points:, 0], triang.vertices[triang.n_svg_points:, 1],
        #             c=vertex_colors[triang.n_svg_points:], cmap="tab20", zorder=5, vmax=19.5, vmin=-0.5, edgecolors='k')
        # plt.scatter(triang.vertices[[pure_junctions], 0], triang.vertices[[pure_junctions], 1], color="red",
        #             marker="*", s=0.5, zorder=5)
        # plt.scatter(triang.vertices[[free_svg_vertices], 0], triang.vertices[[free_svg_vertices], 1], color="orange",
        #             marker=".", s=0.5)
        for i in range(triang.n_svg_points):
            v1 = triang.vertices[i]
            plt.text(v1[0], v1[1], s=f"{i}",
                     fontsize=1,
                     zorder=6,
                     )
        for ie in svg_cut_edge_idx:
            i1 = EV[ie][0]
            v1 = triang.vertices[i1]
            i2 = EV[ie][1]
            v2 = triang.vertices[i2]
            plt.plot([v1[0], v2[0]],
                     [v1[1], v2[1]],
                     color='orange',
                     zorder=4,
                     )
        plt.title("all cut edges")
        plt.savefig(saveto / "cuts.svg")
        plt.close()
    print("----------------------")
    for e in svg_cut_edge_idx:
        print(f"[{EV[e][0]}, {EV[e][1]}]")
    return svg_cut_edge_idx, dict_vertex_to_cut_edge


def cutTriangulation(
        triang: myTraingulation,
        svg_edges,
        svg_points,
        fillmap: np.array,
        depthimage: np.array,
        saveto: pathlib.Path,
        edges_to_cut=None,
):
    """

    :param triang:
    :param svg_edges:
    :param svg_points:
    :param fillmap:
    :param depthimage:
    :param saveto:
    :param edges_to_cut:
    :return:
    """
    region_to_internal, region_to_junction, pure_junctions, dict_vertex_to_region = triang.build_region_to_vertex_dicts(
        fillmap=fillmap,
        plot=True,
        saveto=saveto,
        svg_edges=svg_edges,
    )
    return cutExistingTriangulation(
        triang=triang,
        svg_edges=svg_edges,
        svg_points=svg_points,
        depthimage=depthimage,
        pure_junctions=pure_junctions,
        dict_vertex_to_region=dict_vertex_to_region,
        dict_region_to_junctions=region_to_junction,
        saveto=saveto,
        edges_to_cut=edges_to_cut,
    )


def cutExistingTriangulation(
        triang: myTraingulation,
        svg_edges,
        svg_points,
        depthimage: np.array,
        pure_junctions: np.array,
        dict_vertex_to_region: dict,
        dict_region_to_junctions: dict,
        saveto: pathlib.Path,
        edges_to_cut=None,
):
    n_patches = max(dict_region_to_junctions.keys()) + 1
    n_init_vertices = triang.vertices.shape[0]
    print("Vertices before cut: ", n_init_vertices)
    print("Init pure_junctions cutTriangulation: ", pure_junctions)
    print("len pure_junctions: ", len(pure_junctions))
    all_junction_idx = np.arange(triang.n_svg_points)
    temp = np.arange(triang.n_svg_points)
    temp[pure_junctions] = -1
    free_svg_vertices = np.copy(temp[temp >= 0])
    print("free_svg_vertices: ", free_svg_vertices)
    print("len free: ", len(free_svg_vertices))

    EV, FE, EF = igl.edge_topology(v=triang.vertices, f=triang.faces)

    vertices_to_edges_map = {i: list() for i in range(triang.vertices.shape[0])}
    for i in range(len(EV)):
        e = EV[i]
        vertices_to_edges_map[e[0]].append(i)
        vertices_to_edges_map[e[1]].append(i)

    if edges_to_cut is None:
        svg_cut_edge_idx, dict_vertex_to_cut_edge = findEdgesToCut(
            triang=triang,
            svg_edges=svg_edges,
            svg_points=svg_points,
            depthimage=depthimage,
            # fillmap=fillmap,
            EV=EV,
            EF=EF,
            dict_vertex_to_region=dict_vertex_to_region,
            report=True,
            saveto=saveto,
        )
    else:
        # edges to cut is a list like [[16, 17], [17,4] ..]
        svg_cut_edge_idx = list()  # contains a list of all edges to be cut
        dict_vertex_to_cut_edge = {
            i: list()
            for i in range(triang.n_svg_points)
        }  # stores list of edges to cut around this vertex
        for e in edges_to_cut:
            igl_edge_id_set = set(vertices_to_edges_map[e[0]]).intersection(
                set(vertices_to_edges_map[e[1]])
            )
            if len(igl_edge_id_set) != 1:
                raise Exception(f"Cant find edge between vertices {e[0]} {e[1]}, {igl_edge_id_set}")
            igl_edge_id = igl_edge_id_set.pop()
            svg_cut_edge_idx.append(igl_edge_id)
            dict_vertex_to_cut_edge[e[0]].append(igl_edge_id)
            dict_vertex_to_cut_edge[e[1]].append(igl_edge_id)

    set_of_cut_edges = set(svg_cut_edge_idx)

    high_valence_cut_vertices = {x for x in dict_vertex_to_cut_edge.keys() if len(dict_vertex_to_cut_edge[x])>2}
    print("High-velence vertices: ", high_valence_cut_vertices)

    for x in high_valence_cut_vertices:
        print(f"High-valence vertex {x}: {dict_vertex_to_cut_edge[x]}")

    list_of_cut_chains = list()

    while len(set_of_cut_edges) > 0:
        cc = extract_chain(
            vertex_map_cut_edge=dict_vertex_to_cut_edge,
            edges_to_vertices=EV,
            vertex_to_regions=dict_vertex_to_region,
            high_valence_vertices=high_valence_cut_vertices,
            vertex_boundary_markers=triang.vertex_markers,
        )
        print("Chain extracted: ", cc.vertex_tuple)
        list_of_cut_chains.append(cc)
        for e in cc.edges_tuple:
            set_of_cut_edges.remove(e)

    # list_of_cut_chains.sort(key=lambda x: len(x.vertex_tuple), reverse=True)

    pairs_of_duplicates = list()
    executed_cutchains = list()

    updated_vertex_to_region = deepcopy(dict_vertex_to_region)

    print("\nAll cutchains detected: ")
    map_vertex_to_chain = defaultdict(list)
    list_chain_common_region = list()
    for i in range(len(list_of_cut_chains)):
        cc = list_of_cut_chains[i]
        r = cc.get_common_region(vertex_to_region=dict_vertex_to_region)
        list_chain_common_region.append(r)
        print(f"Chain {i}, common region {r}")
        print(f"- chain vertices {cc.vertex_tuple}, regions {r}")
        v1, v2 = cc.vertex_tuple[0], cc.vertex_tuple[-1]
        map_vertex_to_chain[v1].append(i)
        map_vertex_to_chain[v2].append(i)

    set_high_valence = {v for v in map_vertex_to_chain.keys() if len(map_vertex_to_chain[v]) > 1}
    print("High valence chain ends: ", set_high_valence)

    plt.figure()
    plt.imshow(depthimage, zorder=2, alpha=0.1)
    plt.scatter(triang.vertices[:triang.n_svg_points, 0], triang.vertices[:triang.n_svg_points, 1],
                color="red", zorder=6, edgecolors='k', s=1, linewidths=0.1)
    for idx in range(triang.n_svg_points):
        v1 = triang.vertices[idx]
        plt.text(v1[0], v1[1], s=f"{idx}",
                 fontsize=1.5,
                 zorder=7,
                 )
    for e in EV:
        plt.plot([triang.vertices[e[1], 0], triang.vertices[e[0], 0]],
                 [triang.vertices[e[1], 1], triang.vertices[e[0], 1]],
                 color='gray', linewidth=0.2, zorder=3)
    cmap = matplotlib.cm.get_cmap('tab20')
    for i_cc in range(len(list_of_cut_chains)):
        cc = list_of_cut_chains[i_cc]
        if len(cc.edges_tuple) > 1:
            plt.plot(triang.vertices[cc.vertex_tuple, 0], triang.vertices[cc.vertex_tuple, 1],
                     color=cmap((i_cc % 20) / 20), zorder=4)
    plt.title("Extracted cut chains")
    plt.savefig(saveto / "chains.svg")
    plt.close()

    potential_cut_vertices = set()
    for i_cc in range(len(list_of_cut_chains)):
        # TODO: this is a temporary tweak to not cut anything and solve with pyomo instead
        cc = list_of_cut_chains[i_cc]
        potential_cut_vertices = potential_cut_vertices.union(set(cc.vertex_tuple))
        if edges_to_cut is None:
            continue
        print("\n--- cut ---")
        print("CutChain vertex tuple: ", cc.vertex_tuple)
        print(triang.vertices.shape)
        if len(cc.edges_tuple) > 1:
            faces_left, newvertices, newfaces, newmarkers, newvertex_to_region, pair_of_duplicates, \
            newedges_v, newedges_f, newvertex_e = cutMeshAlongChain(
                cutchain=cc,
                triangulation=triang,
                mat_edges_to_faces=EF,
                mat_edges_to_vertices=EV,
                map_vertices_to_edges=vertices_to_edges_map,
                vertex_to_region=dict_vertex_to_region,
            )
            triang.vertices = newvertices
            triang.faces = newfaces
            triang.vertex_markers = newmarkers
            EF = newedges_f
            EV = newedges_v
            vertices_to_edges_map = newvertex_e
            for k in newvertex_to_region.keys():
                updated_vertex_to_region[k] = newvertex_to_region[k]
            pairs_of_duplicates.append(pair_of_duplicates[0])
            pairs_of_duplicates.append(pair_of_duplicates[1])
            executed_cutchains.append(cc.vertex_tuple)
            # update next chains in list with new vertex id
            for j_cc in range(i_cc+1, len(list_of_cut_chains)):
                jcc = list_of_cut_chains[j_cc]
                jcc.update_by_edges_(new_edge_to_vertex_map=newedges_v)
        # break

    # extract_chain(
    #     vertex_map_cut_edge=dict_vertex_to_cut_edge,
    #     edges_to_vertices=EV,
    # )

    print("Potential cut vertices: ", potential_cut_vertices)

    n_final_vertices = triang.vertices.shape[0]
    print("Vertices after cut: ", n_final_vertices)
    print("Pairs of duplicates: ", pairs_of_duplicates)

    newregion_to_internal = {
        i: list()
        for i in range(n_patches)
    }
    newregion_to_junction = {
        i: list()
        for i in range(n_patches)
    }

    # here we separate boundary vertices from junction vertices
    newpure_junctions = list()
    newall_junctions = list()
    vertex_svg_flag = np.zeros_like(triang.vertex_markers, dtype=int)  # contains 1 if the vertex is on svg
    vertex_svg_flag[:triang.n_svg_points] = 1
    vertex_svg_flag[n_init_vertices:] = 1  # as we only cut along svg edges, we could have dublicated some

    for i in range(len(triang.vertices)):
        corr_regions = updated_vertex_to_region[i]
        if len(corr_regions) > 1:
            newpure_junctions.append(i)
            if vertex_svg_flag[i] != 1:
                raise Exception(f"Vertex {i} not on SVG is a junction?")
        if vertex_svg_flag[i] == 1:
            newall_junctions.append(i)
            for r in corr_regions:
                newregion_to_junction[r].append(i)
        else:
            r = corr_regions[0]
            newregion_to_internal[r].append(i)

    # reshuffle vertices such that pure junctions are in the beginning
    purejunction_marker = np.zeros(shape=(triang.vertices.shape[0]), dtype=int)
    purejunction_marker[newpure_junctions] = 1
    print("newpure_junctions: ", newpure_junctions)
    print("newall_junctions: ", newall_junctions)
    n_pure_junction = purejunction_marker.sum()
    old_to_new_vertex_map = triang.reshuffle_triangulation_vertices(
        use_this_marker=vertex_svg_flag,
        return_mapping=True,
    )

    triang.plot(
        show=False,
        faces=True,
        saveto=saveto,
        name=f"triang_after_cuts",
    )

    for i in range(n_patches):
        newregion_to_internal[i] = np.array([old_to_new_vertex_map[x] for x in newregion_to_internal[i]], dtype=int)
        newregion_to_junction[i] = np.array([old_to_new_vertex_map[x] for x in newregion_to_junction[i]], dtype=int)

    # newpure_junctions = np.arange(n_pure_junction, dtype=int)
    # print("n_pure_junction: ", n_pure_junction)

    reshuffled_all_junctions = np.array([
        old_to_new_vertex_map[x] for x in newall_junctions
    ], dtype=int)
    print(f"All junctions ({reshuffled_all_junctions.shape}) after cut: ", reshuffled_all_junctions)

    print("pairs_of_duplicates: ", pairs_of_duplicates)

    pairs_of_duplicates = [
        [
            old_to_new_vertex_map[v]
            for v in xs
        ]
        for xs in pairs_of_duplicates
    ]

    print("pairs_of_duplicates (renumerated) : ", pairs_of_duplicates)

    return triang, newregion_to_internal, newregion_to_junction, reshuffled_all_junctions, pairs_of_duplicates, executed_cutchains


if __name__ == "__main__":
    print("------ Triangulation!")
    # pngname = "6_freestyle_288_01"
    # pngname = "assorted_Posts_008_1"
    # pngname = "Cylindrical_Parts_011_1"
    # pngname = "Prismatic_Stock_017_1"
    # pngname = "6_freestyle_288_01"
    # pngname = "npr_1015_67.6_132.72_1.4"
    pngname = "npr_1020_52.42_39.33_1.4"
    # pngname = "npr_1041_45.27_-134.8_1.4"`
    start_triang, start_svg_points, start_svg_edges, start_path_edges = myTraingulation.from_svg(
        # path_to_svg=f"clean_svgs/{itemidx}_freestyle_{itemangle:03d}_01_vector.svg",
        path_to_svg=f"results/{pngname}/{pngname}_clean.svg",
        # triang_flags="qYY",
        svg_sampling_distance=10,
        triang_flags='YYqpa50',
        # svg_sampling_distance=20,
        # triang_flags='YYqpa100',
    )
    print(start_path_edges)
    # start_triang.vertices /= 2
    start_triang.plot(show=False, faces=True, saveto=pathlib.Path(f"reports/"), name=f"{pngname}_triang_before_cuts")

    print("--------- DATA")
    depthdata = np.load(f"results/{pngname}/npz/{pngname}_depth.npz")
    # gtimage = np.pad(gt_data["sketch"], ((8, 8), (8, 8)), constant_values=gt_data["sketch"][0, 0])
    input_image = depthdata["sketch"]
    image_for_trapped_ball = ((1 - input_image) * 255).astype(np.uint8)
    input_depthimage = depthdata["depth"]
    # depthimage = np.pad(gt_data["depth"], ((8, 8), (8, 8)), constant_values=gt_data["depth"][0, 0])
    input_fillmap = my_trapped_ball(image_for_trapped_ball)

    trng, rg_to_internal, rg_to_junction, reshuff_all_junctions, prs_of_duplicates, list_of_chains = cutTriangulation(
        triang=start_triang,
        svg_points=start_svg_points,
        svg_edges=start_svg_edges,
        depthimage=input_depthimage,
        fillmap=input_fillmap,
        saveto=pathlib.Path("reports/"),
    )

    trng.vertices = np.hstack((
        trng.vertices,
        np.ones((trng.vertices.shape[0], 1))
    ))
    for i in range(len(prs_of_duplicates)):
        if i % 2 == 0:
            trng.vertices[prs_of_duplicates[i], 2] = 10
    trng.export_obj("reports/test.obj")
    print("Exported")
