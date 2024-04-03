import numpy as np
import argparse
from svgpathtools import svg2paths2, wsvg, Path, Line
from util_svgs import save_paths_to_svg
from collections import Counter
from simplification.cutil import (
    simplify_coords,
    simplify_coords_idx,
    simplify_coords_vw,
    simplify_coords_vw_idx,
    simplify_coords_vwp,
)
import warnings


def svg_simplifier(
    path_to_svg: str,
    thr_pathlength=1,
):
    paths, attributes, svg_attributes = svg2paths2(path_to_svg)
    # print(paths)
    newpaths_list = []
    for p in paths:
        if p.length() < thr_pathlength:
            warnings.warn(f"(del) Path has length < {thr_pathlength}: \n{p}")
            continue
        if len(p) < 5:
            warnings.warn(f"(skip) Path has only {len(p)} nodes: \n{p}")
            newpaths_list.append(p)
            continue
        # print(p)
        # print(p[0])
        thispath_start = p.start
        thispath_end = p.end
        # print(thispath_start)
        thispathcoords = [[thispath_start.real, thispath_start.imag]]
        # print(thispathcoords)
        for s in p:
            segend = s.end
            thispathcoords.append([segend.real, segend.imag])
        thispathcoords = np.array(thispathcoords)
        simplifiedcoords = simplify_coords(thispathcoords, epsilon=0.1)
        # simplifiedcoords = simplify_coords_vwp(thispathcoords, epsilon=30)
        # simplify_coords SOMETIMES starts from origin
        startindex = 1
        if np.sum(simplifiedcoords[0, :]**2) < 0.1:
            startindex = 2
        newthispath_list = list()
        for i in range(startindex, len(simplifiedcoords)):
            a = simplifiedcoords[i-1]
            b = simplifiedcoords[i]
            newthispath_list.append(
                Line(start=a[0]+a[1]*1j, end=b[0] + b[1]*1j)
            )
        if len(newthispath_list) == 0:
            print(thispathcoords)
            print(simplifiedcoords)
            raise Exception("empty")
        newthispath = Path(*newthispath_list)
        newpaths_list.append(newthispath)
        # print(f"after simplification: {thispath_start} {thispath_end} -> {newthispath.start} {newthispath.end}")
        newthispath.start = thispath_start
        newthispath.end = thispath_end
    return newpaths_list


def cut_loose_ends(
        paths_list: list,
        endpoints_array: np.array,
        path_to_endpoints: list,
):
    """
    Remove paths that has unique ends. Remove paths that are short loops
    :param paths_list:
    :param endpoints_array:
    :param path_to_endpoints:
    :return:
    """
    path_lengths = [x.length() for x in paths_list]
    endpoints_idx_flat = []
    for x in path_to_endpoints:
        endpoints_idx_flat.extend(x)
    fc = Counter(endpoints_idx_flat)
    loose_ends = {x for x in fc.keys() if fc[x] == 1}
    print("Loose ends: ", loose_ends)
    paths_to_delete = list()
    for i_p in range(len(paths_list)):
        if path_lengths[i_p] < 20:
            print(f"Path {i_p} has ends ({path_to_endpoints[i_p]}) and length {path_lengths[i_p]}")
            if path_to_endpoints[i_p][0] == path_to_endpoints[i_p][1]:
                paths_to_delete.append(i_p)
            # if (path_to_endpoints[i_p][0] in loose_ends) or (path_to_endpoints[i_p][1] in loose_ends):
            #     print(f" - Path {i_p} has loose end ({path_to_endpoints[i_p]}) and length {path_lengths[i_p]}")
            #     paths_to_delete.append(i_p)
    print("Paths to delete: ", paths_to_delete)
    for x in paths_to_delete[::-1]:
        del(paths_list[x])

    return paths_list, endpoints_array, path_to_endpoints


def svg_cleaner(
        path_to_svg: str,
):
    """
    split paths if intersection is detected. in the end paths do not intersect with each other, except for the endpoints
    :param path_to_svg:
    :return:
    """
    paths, _, _ = svg2paths2(path_to_svg)
    paths.sort(key=lambda x: x.length())
    print("Init paths len: ", len(paths))
    save_paths_to_svg(save_to=path_to_svg.replace(".svg", "_raw.svg"), paths_to_save=paths)
    paths = svg_simplifier(path_to_svg)
    print("After simplification: ", len(paths))
    save_paths_to_svg(save_to=path_to_svg.replace(".svg", "_simplified.svg"), paths_to_save=paths)
    new_paths_list = list()
    new_endpoints_to_array = np.array([])   # contains a list of all endpoints, we use it to force paths to start at end at the same points
    path_to_endpoints_idx = list()
    i_path = 0
    # discard intersections if they are small (T 0-1 parametrizes the entire path)
    eps = 0.5
    while i_path < len(paths):
        print(f"--- {i_path} / {len(paths)}")
        # save_paths_to_svg(save_to="temp.svg", paths_to_save=paths)
        thispath = paths[i_path]
        found_intersection = False
        for j_path in range(i_path + 1, len(paths)):
            # searching if thispath intersects any other path
            # print(f"{j_path} / {len(paths)}")
            otherpath = paths[j_path]
            try:
                intersections = thispath.intersect(otherpath)
                if len(intersections) > 0:
                    (T1, seg1, t1), (T2, seg2, t2) = intersections[0]
                    print(f"{i_path} x {j_path}", T1, T2)
                    # get the smallest part of paths before intersection
                    cut1 = min(T1 * thispath.length(), (1 - T1) * thispath.length())
                    cut2 = min(T2 * otherpath.length(), (1 - T2) * otherpath.length())
                    if (cut1 > eps) or (cut2 > eps):
                        print(f"-> gotcha, {cut1} {cut2}")
                        found_intersection = True
                        break
                    else:
                        print(f"intersection cuts {cut1} and {cut2}")
            except Exception as e:
                print(thispath)
                print(otherpath)
                print(len(otherpath))
                raise Exception("Error when intersecting")
        if found_intersection:
            # if we have intersection
            otherpath = paths[j_path]
            # remove current path and the one it intersects
            paths.pop(j_path)
            paths.pop(i_path)
            if (T1 > 0) and (T1 < 1):
                intersection_point = thispath.point(T1)
                # if path 1 is intersected at midpoint, split and add both parts to the end of our list
                segment_index, segment_t = thispath.T2t(T1)
                middle_segment_first_half, middle_segment_second_half = thispath[segment_index].split(segment_t)
                # first_half = Path(*(thispath[:segment_index - 1] + [middle_segment_first_half]))
                # second_half = Path(*([middle_segment_second_half] + thispath[segment_index + 1:]))
                if segment_index == 0:
                    thispath.start = intersection_point
                    paths.append(thispath)
                else:
                    if segment_index == len(thispath) - 1:
                        thispath.end = intersection_point
                        paths.append(thispath)
                    else:
                        first_half = Path(*(thispath[:segment_index - 1]))
                        if len(first_half) > 0:
                            first_half[-1].end = intersection_point
                            paths.append(first_half)
                        second_half = Path(*(thispath[segment_index + 1:]))
                        if len(second_half) > 0:
                            second_half[0].start = intersection_point
                            paths.append(second_half)
                print(f"Path {i_path} got split at {T1}, segment index {segment_index}")
            else:
                # else if it is intersected at the endpoint, put it back
                paths.append(thispath)
                print(f"Path {i_path} is not split and now at {len(paths) - 1}")
            if (T2 > 0) and (T2 < 1):
                intersection_point = otherpath.point(T2)
                # if path 2 is intersected at midpoint, split and add both parts to the end of our list
                segment_index, segment_t = otherpath.T2t(T2)
                middle_segment_first_half, middle_segment_second_half = otherpath[segment_index].split(segment_t)
                # first_half = Path(*(otherpath[:segment_index - 1] + [middle_segment_first_half]))
                # second_half = Path(*([middle_segment_second_half] + otherpath[segment_index + 1:]))
                if segment_index == 0:
                    otherpath.start = intersection_point
                    paths.append(otherpath)
                else:
                    if segment_index == len(otherpath) - 1:
                        otherpath.end = intersection_point
                        paths.append(otherpath)
                    else:
                        first_half = Path(*(otherpath[:segment_index - 1]))
                        if len(first_half) > 0:
                            first_half[-1].end = intersection_point
                            paths.append(first_half)
                        second_half = Path(*(otherpath[segment_index + 1:]))
                        if len(second_half) > 0:
                            second_half[0].start = intersection_point
                            paths.append(second_half)
                print(f"Path {j_path} got split at {T2}, segment index {segment_index}")
            else:
                # else if it intersected at the endpoint, put it back
                paths.append(otherpath)
                print(f"Path {j_path} is not split and now at {len(paths) - 1}")
        else:
            # if we don't have any intersections with this path, add it to our new list and check the endpoints
            a = thispath.point(0)
            if len(new_endpoints_to_array) == 0:
                new_endpoints_to_array = np.append(new_endpoints_to_array, a)
                idxa = 0
            else:
                distances = np.absolute(new_endpoints_to_array - a)
                print(f"point {a}, min distance: ", np.min(distances))
                if np.min(distances) > 20 * eps:
                    new_endpoints_to_array = np.append(new_endpoints_to_array, a)
                    idxa = len(new_endpoints_to_array) - 1
                else:
                    idxa = np.argmin(distances)
            a = new_endpoints_to_array[idxa]
            thispath[0].start = a

            b = thispath.point(1)
            distances = np.absolute(new_endpoints_to_array - b)
            print(f"point {b}, min distance: ", np.min(distances))
            if np.min(distances) > 20 * eps:
                new_endpoints_to_array = np.append(new_endpoints_to_array, b)
                idxb = len(new_endpoints_to_array) - 1
            else:
                idxb = np.argmin(distances)
            b = new_endpoints_to_array[idxb]
            thispath[-1].end = b

            new_paths_list.append(thispath)
            path_to_endpoints_idx.append([idxa, idxb])
            i_path += 1

    return new_paths_list, new_endpoints_to_array, path_to_endpoints_idx


if __name__ == "__main__":
    testname = "visible_free2cad_cut"
    parser = argparse.ArgumentParser(description="Eval model")
    parser.add_argument("--input", default=f"results/{testname}/{testname}.svg", type=str, help="path to input svg")
    parser.add_argument("--output", default=f"results/{testname}/{testname}_clean.svg", type=str, help="path to output svg")

    args = parser.parse_args()

    # simplifiepaths = svg_simplifier(path_to_svg=f'{args.input}')
    # save_paths_to_svg(save_to="simplified.svg", paths_to_save=simplifiepaths)
    newpaths, endpoints_array, path_to_endpoints = svg_cleaner(path_to_svg=f'{args.input}')
    print("Len paths: ", len(newpaths))
    print("Len path_to_endpoints: ", len(path_to_endpoints))
    newpaths, endpoints_array, path_to_endpoints = cut_loose_ends(newpaths, endpoints_array, path_to_endpoints )
    print("Len paths after cleaning: ", len(newpaths))
    save_paths_to_svg(save_to=args.output, paths_to_save=newpaths)
