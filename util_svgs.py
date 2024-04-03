import os
import traceback
import warnings

from svgpathtools import svg2paths2, wsvg, Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pathlib
# from svg_cleaner import svg_cleaner


def save_paths_to_svg(
        save_to: str,
        paths_to_save: list[Path],
):
    sorted_paths = sorted(paths_to_save, key=lambda x: x.length(), reverse=True)
    attributes_to_save = [
        {
            'fill': 'none',
            'stroke-width': 3,
            'stroke': "rgb({},{},{})".format(*get_tab20_rgb(i)),
        }
        for i in range(len(paths_to_save))
    ]
    wsvg(sorted_paths, attributes=attributes_to_save, filename=save_to)


def naive_sampler(
        path_to_svg: str,
):
    paths, attributes, svg_attributes = svg2paths2(path_to_svg)
    vertices = np.zeros(shape=(0, 2), dtype=int)
    edges = list()
    for p in paths[:-1]:
        for i in np.linspace(0, 1, 20, endpoint=True):
            x = p.point(i).real
            y = 400 - p.point(i).imag
            vertices = np.vstack((vertices, np.array([x, y])))
            if i > 0:
                edges.append([len(vertices)-1, len(vertices)-2])
    return vertices, edges


def build_stroke_to_endpoints(
        paths: list[Path],
        eps=0.5,
):
    paths_ends_to_vertices_idx = list()
    vertices_complex = np.array([paths[0].start])
    for p in paths:
        a = p[0].start
        distances = np.absolute(vertices_complex - a)
        if np.min(distances) > 2 * eps:
            vertices_complex = np.append(vertices_complex, a)
            idxa = len(vertices_complex) - 1
        else:
            idxa = np.argmin(distances)
        b = p[-1].end
        distances = np.absolute(vertices_complex - b)
        if np.min(distances) > 2 * eps:
            vertices_complex = np.append(vertices_complex, b)
            idxb = len(vertices_complex) - 1
        else:
            idxb = np.argmin(distances)
        paths_ends_to_vertices_idx.append([idxa, idxb])
    return paths_ends_to_vertices_idx, vertices_complex


def clean_svg_sampler(
        path_to_svg: str,
        sampling_distance=5,
        plot=True,
        show=False,
):
    """
    samples svg paths given the distance
    :param path_to_svg:
    :param sampling_distance:
    :param plot:
    :param show:
    :return:
    """
    paths, _, _ = svg2paths2(path_to_svg)
    # save_paths_to_svg(paths_to_save=paths, save_to="reports/read.svg")
    print(f"Read {path_to_svg}, has {len(paths)} paths")
    # TODO: svg_remove_loose_ends seems similar to cut_loose_ends in svg_cleaner.py
    svg_remove_loose_ends(paths, length_threshold=sampling_distance/2)
    print(f"After cleaning, {len(paths)} remains")
    # save_paths_to_svg(paths_to_save=paths, save_to="reports/cleared.svg")
    edges = list()
    paths_ends_to_vertices_idx, vertices_complex = build_stroke_to_endpoints(
        paths=paths,
        eps=0.5,
    )
    paths_edges = list()
    # print(paths_ends_to_vertices_idx)
    # print(vertices_complex)
    for i_path in range(len(paths)):
        p = paths[i_path]
        thispath_edges = list()
        if p.length() < 1:
            warnings.warn(
                f"Ignoring the path {i_path} with length {p.length()}"
            )
            continue
        if p.length() <= sampling_distance:
            print(f"path {i_path} is too short! {paths_ends_to_vertices_idx[i_path]}")
            if paths_ends_to_vertices_idx[i_path][0] != paths_ends_to_vertices_idx[i_path][1]:
                thispath_edges.append(paths_ends_to_vertices_idx[i_path])
        else:
            idxa = paths_ends_to_vertices_idx[i_path][0]
            idxb = paths_ends_to_vertices_idx[i_path][1]
            n = int(p.length() // sampling_distance)
            print(f"path {i_path} ({idxa} -> {idxb}) len {p.length()} - {n} samples")
            newpoints = np.array([])
            if n == 1:
                newpoints = np.array([p.point(0.5)])
            else:
                for j in np.linspace(start=0, stop=1, num=n+1, endpoint=False)[1:]:
                    newpoints = np.append(newpoints, p.point(j))
            internal_points_idxs = np.arange(start=0, stop=n) + len(vertices_complex)
            vertices_complex = np.append(vertices_complex, newpoints)
            thispath_edges.append([idxa, internal_points_idxs[0]])
            for j in range(n-1):
                thispath_edges.append([internal_points_idxs[j], internal_points_idxs[j+1]])
            thispath_edges.append([internal_points_idxs[-1], idxb])
        edges.extend(thispath_edges)
        paths_edges.append(thispath_edges)

    vertices = np.zeros(shape=(len(vertices_complex), 2), dtype=float)
    vertices[:, 0] = vertices_complex.real
    # vertices[:, 1] = y_flip_value - vertices_complex.imag
    vertices[:, 1] = vertices_complex.imag

    # print(edges)
    # print(vertices_complex)

    if plot:
        plt.figure()
        plt.title("Sampled edges")
        plt.scatter(vertices[:, 0], vertices[:, 1], color='white', edgecolors='k', s=4, linewidths=0.2, zorder=5)
        for i in range(len(edges)):
            e = edges[i]
            plt.plot(
                [vertices[e[0], 0], vertices[e[1], 0]],
                [vertices[e[0], 1], vertices[e[1], 1]],
            )
        # plt.scatter([100], [100])
        plt.axis("equal")
        origpath = pathlib.Path(path_to_svg)
        plt.savefig(origpath.parents[0] / "reports" / "edges.svg")
        if show:
            plt.show()
        plt.close()

    return vertices, edges, paths_edges


def svg_remove_loose_ends(
        paths: list[Path],
        length_threshold=2.0,
):
    paths_ends_to_vertices_idx, vertices_complex = build_stroke_to_endpoints(paths=paths, eps=0.5)
    all_endpoints_idx = [item for sublist in paths_ends_to_vertices_idx for item in sublist]
    endpoint_counter = Counter(all_endpoints_idx)
    loose_endpoints = {key for key, value in endpoint_counter.items() if value==1}
    paths_to_remove = list()
    for i_path in range(len(paths)):
        idxa, idxb = paths_ends_to_vertices_idx[i_path][0], paths_ends_to_vertices_idx[i_path][1]
        p = paths[i_path]
        if p.length() < length_threshold:
            print(f"Path {i_path} is short ({p.length()}). Its endpoints: "
                  f"{idxa} ({endpoint_counter[idxa]} times) "
                  f"{idxb} ({endpoint_counter[idxb]} times)"
                  )
            if (idxa in loose_endpoints) or (idxb in loose_endpoints) or (idxa == idxb):
                print(f"- gotcha, remove path {i_path}")
                paths_to_remove.append(i_path)
    for i_path in sorted(paths_to_remove, reverse=True):
        print(f"remove path {i_path}")
        paths.pop(i_path)
    return paths


def get_tab20_rgb(ind: int):
    ii = ind % 20
    r, g, b, a = plt.cm.tab20(ii)
    return int(255*r), int(255*g), int(255*b)


def experiments():
    # paths, attributes, svg_attributes = svg2paths2('svgs/372_freestyle_252_01_vector.svg')
    paths, attributes, svg_attributes = svg2paths2('svgs/399_freestyle_000_01_vector.svg')
    # print(attributes)
    for i in range(len(attributes)):
        r, g, b = get_tab20_rgb(i)
        attributes[i]['stroke'] = f'rgb({r},{g},{b})'
        attributes[i]['stroke-width'] = 3

    wsvg(paths, attributes=attributes, svg_attributes=svg_attributes, filename='output22.svg')
    # print(paths[0][:3])
    # paths[0][0].start =  200+70j
    # print(paths[0][:2])
    # print(attributes[0])
    #
    # x = list()
    # y = list()
    # for p in paths:
    #     for i in np.linspace(0, 1, 10):
    #         x.append(p.point(i).real)
    #         y.append(400 - p.point(i).imag)
    #
    # list_paths = np.array([])
    # list_endpoints = np.array([])
    # dict_path_to_endpoints = dict()

    # for i_path in range(len(paths)):
    #     p = paths[i_path]
    #     if len(li)
    # print(paths[0])
    # print(paths[0].point(0.1).real)

    # newpaths, endpoints_array, path_to_endpoints = svg_cleaner(path_to_svg='svgs/372_freestyle_252_01_vector.svg')
    newpaths, endpoints_array, path_to_endpoints = svg_cleaner(path_to_svg='svgs/399_freestyle_000_01_vector.svg')

    new_attributes = [
        {
            # 'points': f"{paths[i][0].start.real},{paths[i][0].start.imag} " + " ".join(
            #     [
            #         f"{paths[i][j].end.real},{paths[i][j].end.imag}"
            #         for j in range(len(paths[i]))
            #     ]
            #     ),
            'fill': 'none',
            'stroke-width': 3,
            'stroke': "rgb({},{},{})".format(*get_tab20_rgb(i)),
        }
        for i in range(len(newpaths))
    ]

    wsvg(newpaths, attributes=new_attributes, filename='output2.svg')

    # print(endpoints_array)
    # print(endpoints_array.shape)
    # print(path_to_endpoints)

    # vert, ed = clean_svg_sampler(path_to_svg="clean_svgs/372_freestyle_252_01_vector.svg")
    vert, ed, p_ed = clean_svg_sampler(path_to_svg="output2.svg")

    plt.figure()
    # plt.scatter(vert[:, 0], vert[:, 1])
    for e in ed:
        plt.plot([vert[e[0], 0], vert[e[1], 0]], [vert[e[0], 1], vert[e[1], 1]])
    # plt.scatter([100], [100])
    plt.axis("equal")
    plt.savefig("reports/edges.svg")
    plt.close()


def clean_all():
    for fname in os.listdir("svgs/"):
        print(f"\n\n ========== {fname} ============")
        if not fname.endswith("svg"):
            continue
        if fname in os.listdir("clean_svgs"):
            print(f"{fname} is done already")
            continue
        try:
            newpaths, endpoints_array, path_to_endpoints = svg_cleaner(path_to_svg=f'svgs/{fname}')
            new_attributes = [
                {
                    'fill': 'none',
                    'stroke-width': 3,
                    'stroke': "rgb({},{},{})".format(*get_tab20_rgb(i)),
                }
                for i in range(len(newpaths))
            ]
            wsvg(newpaths, attributes=new_attributes, filename=f'clean_svgs/{fname}')
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            pass


if __name__ == "__main__":
    # experiments()
    clean_all()
