import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from test_cylinder_fit import get_plane_wireframe
import scipy.optimize as spo


def plane_evalZ(x, y, a, b, c, d):
    """
    evaluate z at x,y coords, plane Ax+By+Cz+D = 0
    :param x: np list
    :param y: np list
    :param a: float
    :param b: float
    :param c: float
    :param d: float
    :return: new z values
    """
    if c != 0:
        return (a * x + b * y + d) / (-c)
    else:
        return d


def plane_eval_dist(x, y, z, a, b, c, d):
    """
    evaluate total euclidean distance from points (x,y,z) to plane Ax+By+Cz+D = 0
    @param x:
    @param y:
    @param z:
    @param a:
    @param b:
    @param c:
    @param d:
    """
    distance = np.abs(a * x + b * y + c * z + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
    return np.sum(distance)


def plane_eval_dist2(x, y, z, a, b, c, d):
    """
    compute L2^2 distance from (x,y,z) to plane (a,b,c,d)
    :param x:
    :param y:
    :param z:
    :param a:
    :param b:
    :param c:
    :param d:
    :return:
    """
    distance = (a * x + b * y + c * z + d) ** 2 / (a ** 2 + b ** 2 + c ** 2)
    return np.sum(distance)


def plane_eval_distZ2(x, y, z, a, b, c, d, w_norm=2):
    znew = (a * x + b * y + d) / (-c)
    dist2 = np.sum((z - znew)**2)
    n = np.array([a, b, c])
    norm_penalty = (1 - n.dot(n))**2
    value = dist2 + w_norm * norm_penalty
    return value


def grad_params_plane_distZ2(x, y, z, a, b, c, d, w_norm=2):
    zplane = (a * x + b * y + d) / (-c)
    z_zplane = z - zplane
    n = np.array([a, b, c])
    dEda = 2 * (1 / c) * z_zplane.dot(x)
    dEdb = 2 * (1 / c) * z_zplane.dot(y)
    dEdc = 2 * (1 / c) * z_zplane.dot(zplane)
    dEdd = 2 * (1 / c) * np.sum(z_zplane)
    grad_dist = np.array([dEda, dEdb, dEdc, dEdd])
    grad_norm_penalty = 4 * (n.dot(n) - 1) * n
    grad_norm_penalty = np.append(grad_norm_penalty, values=[0])
    grad_total = grad_dist + w_norm * grad_norm_penalty
    return grad_total


def grad_points_plane_distZ2(x, y, z, a, b, c, d):
    zplane = (a * x + b * y + d) / (-c)
    z_zplane = z - zplane
    dEdx = 2 * (a / c) * z_zplane
    dEdy = 2 * (b / c) * z_zplane
    dEdz = 2 * z_zplane
    return dEdx, dEdy, dEdz


def fit_plane_weighted(x, y, z, weights):
    """
    Fit plane wrt to weights. Error: Z only, from pdf "LeastSquaresFitting.pdf"
    :param x: np list of X coords
    :param y: np list of Y coords
    :param z: np list of Z coords
    :param weights: np list of weights
    :return: a,b,c,d from plane eq Ax+By+Cz+D = 0
    """
    meanx, meany, meanz = np.mean(x * weights), np.mean(y * weights), np.mean(z * weights)
    devx, devy, devz = x - meanx, y - meany, z - meanz
    l00 = np.sum(weights * (devx ** 2))
    l01 = np.sum(weights * (devx * devy))
    l11 = np.sum(weights * (devy ** 2))
    r0 = np.sum(weights * devx * devz)
    r1 = np.sum(weights * devy * devz)
    assert (det := l00 * l11 - l01 ** 2) != 0
    bara0 = (l11 * r0 - l01 * r1) / det
    bara1 = (l00 * r1 - l01 * r0) / det
    b = meanz - bara0 * meanx - bara1 * meany
    # newz = bara0 * x + bara1 * y + b
    print(f"a: {bara0}; b: {bara1}")
    return bara0, bara1, -1, b


def fit_plane_singlestep(x, y, z):
    """
        Fit plane with Z-coord error. Error: Z only, from pdf "LeastSquaresFitting.pdf"
        :param x: np list of X coords
        :param y: np list of Y coords
        :param z: np list of Z coords
        :return: updated z coordinates, keeping x,y
        """
    return fit_plane_weighted(x, y, z, np.ones_like(x))


def fit_plane_euclidean(x, y, z):
    """
    Fit plane with Euclidean dist error. Uses eigen decomposition
    :param x: np list of X coords
    :param y: np list of Y coords
    :param z: np list of Z coords
    :return: updated z coordinates, keeping x,y
    """
    meanx, meany, meanz = np.mean(x), np.mean(y), np.mean(z)
    devx, devy, devz = x - meanx, y - meany, z - meanz
    A = np.vstack((devx, devy, devz))
    # print(A.shape)
    C = np.matmul(A, A.transpose())
    # print(C.shape)
    w, v = np.linalg.eig(C)
    minval = np.argmin(w)
    minvec = v[:, minval]
    # print(w)
    # print(minval)
    # print(minvec)
    # print(v)
    a, b, c = minvec[0], minvec[1], minvec[2]
    d = -(a * meanx + b * meany + c * meanz)
    # newz = (a * x + b * y + d) / (-c)
    return a, b, c, d


def fit_plane_outliers(x, y, z):
    """
    Iterative fit to suppress outliers. Uses Z-coord error
    :param x: np list of X coords
    :param y: np list of Y coords
    :param z: np list of Z coords
    :return: updated z coordinates, keeping x,y
    """
    sigma = 0.5
    z_prev = z
    weights = np.ones_like(x, dtype=float)
    for iteration in range(5):
        newparams = fit_plane_weighted(x, y, z, weights)
        newz = plane_evalZ(x, y, *newparams)
        dist = (z - newz) ** 2
        weights = np.exp(- dist / sigma)
        update = np.sum((newz - z_prev) ** 2)
        print(f"iter {iteration}: update {update}")
        z_prev = newz
    return newz


if __name__ == "__main__":
    np.random.seed(0)
    # random.seed(0)

    truea = 1
    trueb = 2
    truec = 3
    trued = 4

    n_points = 5

    points_x = np.random.rand(n_points) * 2 + 3
    points_y = np.random.rand(n_points) * 4 - 2
    points_z = (truea * points_x + trueb * points_y + trued) / (-truec)

    estimatea, estimateb, estimatec, estimated = -1.2, 1.5, 3.5, 4
    estimate_params = np.array([estimatea, estimateb, estimatec, estimated])

    z2_energy = plane_eval_distZ2(
        x=points_x,
        y=points_y,
        z=points_z,
        a=estimatea,
        b=estimateb,
        c=estimatec,
        d=estimated,
    )

    print("z2_energy: ", z2_energy)

    num_grad_z2_energy = spo.approx_fprime(
        xk=estimate_params,
        f=lambda x: plane_eval_distZ2(
            x=points_x,
            y=points_y,
            z=points_z,
            a=x[0],
            b=x[1],
            c=x[2],
            d=x[3],
        ),
    )

    print("num_grad_z2_energy params: ", num_grad_z2_energy)

    formula_grad = grad_params_plane_distZ2(
        x=points_x,
        y=points_y,
        z=points_z,
        a=estimatea,
        b=estimateb,
        c=estimatec,
        d=estimated,
    )
    print("formula_grad params: ", formula_grad)

    diff = formula_grad - num_grad_z2_energy
    print("Diff grad params: ", np.linalg.norm(diff))

    print("=======================")

    flat_points = np.hstack(
        (
            points_x,
            points_y,
            points_z,
        )
    )

    z2_energy = plane_eval_distZ2(
        x=flat_points[:n_points],
        y=flat_points[n_points:2 * n_points],
        z=flat_points[2 * n_points:],
        a=estimatea,
        b=estimateb,
        c=estimatec,
        d=estimated,
    )

    print("z2_energy: ", z2_energy)

    num_grad_z2_energy = spo.approx_fprime(
        xk=flat_points,
        f=lambda x: plane_eval_distZ2(
            x=x[:n_points],
            y=x[n_points:2*n_points],
            z=x[2*n_points:],
            a=estimate_params[0],
            b=estimate_params[1],
            c=estimate_params[2],
            d=estimate_params[3],
        ),
    )

    print("num_grad_z2_energy: (points): ", num_grad_z2_energy)

    formula_grad_points = grad_points_plane_distZ2(
        x=flat_points[:n_points],
        y=flat_points[n_points:2 * n_points],
        z=flat_points[2 * n_points:],
        a=estimate_params[0],
        b=estimate_params[1],
        c=estimate_params[2],
        d=estimate_params[3],
    )

    print("formula_grad params (points): ", formula_grad_points)


    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # plt.title("Gauss image")
    #
    # ax.scatter3D(points_x, points_y, points_z, c="blue")
    #
    # X, Y, Z = get_plane_wireframe(
    #     point=np.array([points_x[0], points_y[0], points_z[0]]),
    #     normal=np.array([truea, trueb, truec]),
    # )
    # ax.plot_wireframe(X, Y, Z,
    #                   label="estimated",
    #                   color="red",
    #                   # rstride=2, cstride=2,
    #                   )
    #
    # plt.legend()
    # plt.show()
    # plt.close()
