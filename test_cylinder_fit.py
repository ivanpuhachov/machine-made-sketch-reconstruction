import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import skg
import random
from test_fit_circle import get_xy_circle
from geomfitty.fit3d import cylinder_fit, cylinder_fit_residuals, distances_to_line
from pyomo_distances import cylinder_distance
import pyomo.environ as pyo


def get_triplet(w: np.array):
    """
    Return
    :param w: np array of dim 3
    :return: e0, e1, e2 - orthonormal basis, e0 || w
    """
    e0 = w / np.linalg.norm(w)
    e1 = np.array([0, w[2], -w[1]])
    if w[2] == 0:
        e1 = np.array([w[1], -w[0], 0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(e0, e1)
    return e0, e1, e2


def skew_lines_dist(
        unit_a,
        unit_b,
        point_a,
        point_b,
):
    d = point_b - point_a
    uadotub = np.dot(unit_a, unit_b)
    uadotd = np.dot(unit_a, d)
    ubdotd = np.dot(unit_b, d)
    A = np.array([
        [1, -uadotub],
        [uadotub, -1]
    ])
    b = np.array([uadotd, ubdotd])
    if np.linalg.det(A) == 0:
        print("zero determinant")
        return
    s = np.matmul(np.linalg.inv(A), b)
    k = - unit_a * s[0] + d + unit_b * s[1]
    return np.array([s[0], s[1], np.linalg.norm(k)])


def solve_quadratic_equation(
        a: float,
        b: float,
        c: float,
):
    discriminant = b ** 2 - 4 * a * c
    sqd = np.sqrt(discriminant)
    return (-b + sqd) / (2*a), (-b - sqd) / (2*a),


def generate_cylinder_points(
        trueC: np.array,
        trueW: np.array,
        truer2: float,
        height=3.0,
        n=10,
):
    random_height = height * (np.random.rand(1, n) - 0.5)
    random_phi = 0.5 * np.pi * np.random.rand(1, n)
    truex1 = np.array([0, trueW[2], -trueW[1]]).reshape((-1, 1))
    truex0 = np.cross(truex1.squeeze(1), trueW).reshape((-1, 1))
    truex0 /= np.linalg.norm(truex0)
    truex1 /= np.linalg.norm(truex1)
    truex2 = trueW.reshape(-1, 1) / np.linalg.norm(trueW)
    result = np.sqrt(truer2) * (
            truex0 * np.sin(random_phi) + truex1 * np.cos(random_phi)
    ) + truex2 * random_height + trueC.reshape(3, 1)
    normals = - truex0 * np.sin(random_phi) - truex1 * np.cos(random_phi)
    # result = truex0 * np.sin(random_phi)
    return result, normals


def plot_cylinder(ax, pointC, directionW, r, height=2):
    nlines = 20
    x1 = np.array([0, directionW[2], -directionW[1]]).reshape((-1, 1))
    x0 = np.cross(x1.squeeze(1), directionW).reshape((-1, 1))
    x0 /= np.linalg.norm(x0)
    x1 /= np.linalg.norm(x1)
    x2 = directionW.reshape(-1, 1) / np.linalg.norm(directionW)
    h = np.linspace(-height, height, nlines).reshape(1, nlines)
    theta = np.linspace(-1 * np.pi, 1 * np.pi, nlines)
    for i in range(nlines):
        points = pointC.reshape(3, 1) + h[0, i] * x2 + r * (x0 * np.sin(theta) + x1 * np.cos(theta))
        ax.plot3D(points[0, :], points[1, :], points[2, :], 'gray')


def plot_gauss_image(ax, pts, show=False):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", linewidths=0.5)
    ax.scatter3D([0], [0], [0], color='k')
    ax.scatter3D(pts[0, :], pts[1, :], pts[2, :], c="blue", label="normals")

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_aspect('equal')
    if show:
        plt.legend()
        plt.show()
        plt.close()


def cylinder_evalZ(
        x,
        y,
        z,
        c,
        w,
        r2,
        debug=True,
):
    newz = np.copy(z)
    for i_cyl_point in range(len(x)):
        # todo make it faster
        cx, cy, cz = x[i_cyl_point], y[i_cyl_point], z[i_cyl_point]
        cp = np.array([cx, cy, cz])
        try:
            res = CylinderFit.project_point_on_cyl(
                x=cx,
                y=cy,
                z=cz,
                t=np.array([0, 0, 1]),
                c=c,
                w=w,
                r2=r2,
                debug=debug,
            )
            if res is None:
                print("None!: ", cp)
                continue
        except Exception as e:
            continue
        p1, p2 = res
        d1, d2 = np.linalg.norm(cp - p1), np.linalg.norm(cp - p2)
        if d1 < d2:
            newz[i_cyl_point] = p1[2]
        else:
            newz[i_cyl_point] = p2[2]
    return newz


def cylinder_find_closest(
        x,
        y,
        z,
        c,
        w,
        r2,
        debug=True,
):
    newx = np.copy(x)
    newy = np.copy(y)
    newz = np.copy(z)
    for i_cyl_point in range(len(x)):
        cx, cy, cz = x[i_cyl_point], y[i_cyl_point], z[i_cyl_point]
        newp = CylinderFit.find_closest_point(
            x=cx,
            y=cy,
            z=cz,
            c=c,
            w=w,
            r2=r2,
            debug=debug,
        )
        newx[i_cyl_point] = newp[0]
        newy[i_cyl_point] = newp[1]
        newz[i_cyl_point] = newp[2]
    return newx, newy, newz


def cylinder_projectZ_or_closest(
        x,
        y,
        z,
        c,
        w,
        r2,
        debug=True,
):
    newx = np.copy(x)
    newy = np.copy(y)
    newz = np.copy(z)
    for i_cyl_point in range(len(x)):
        cx, cy, cz = x[i_cyl_point], y[i_cyl_point], z[i_cyl_point]
        cp = np.array([cx, cy, cz])
        try:
            res = CylinderFit.project_point_on_cyl(
                x=cx,
                y=cy,
                z=cz,
                t=np.array([0, 0, 1]),
                c=c,
                w=w,
                r2=r2,
                debug=debug,
            )
            if res is None:
                print("None!: ", cp)
                continue
            p1, p2 = res
            d1, d2 = np.linalg.norm(cp - p1), np.linalg.norm(cp - p2)
            if d1 < d2:
                newz[i_cyl_point] = p1[2]
            else:
                newz[i_cyl_point] = p2[2]
        except Exception as e:
            if str(e) == "NoIntersection":
                newp = CylinderFit.find_closest_point(
                    x=cx,
                    y=cy,
                    z=cz,
                    c=c,
                    w=w,
                    r2=r2,
                    debug=debug,
                )
                newx[i_cyl_point] = newp[0]
                newy[i_cyl_point] = newp[1]
                newz[i_cyl_point] = newp[2]
            else:
                print(e)
                continue
    return newx, newy, newz


def cylinder_find_closest_alongZ(
        x,
        y,
        z,
        c,
        w,
        r2,
        debug=True,
):
    newz = np.copy(z)
    for i_cyl_point in range(len(x)):
        cx, cy, cz = x[i_cyl_point], y[i_cyl_point], z[i_cyl_point]
        p1, p2, status = CylinderFit.find_points_along_ray(
            x=cx,
            y=cy,
            z=cz,
            t=np.array([0, 0, 1]),
            c=c,
            w=w,
            r2=r2,
        )
        if status:
            # then ray (0,0,1) intersects cylinder, select closest point to prediction
            cp = np.array([cx, cy, cz])
            d1, d2 = np.linalg.norm(cp - p1), np.linalg.norm(cp - p2)
            if d1 < d2:
                newz[i_cyl_point] = p1[2]
            else:
                newz[i_cyl_point] = p2[2]
        else:
            # then there is no intersection, find_points_along_ray returned closest point on the ray
            newz[i_cyl_point] = p1[2]
    return newz




def fit_plane(x, y, z):
    print(f"Fit plane on {len(x)} points")
    meanx, meany, meanz = np.mean(x), np.mean(y), np.mean(z)
    devx, devy, devz = x - meanx, y - meany, z - meanz
    A = np.vstack((devx, devy, devz))
    C = np.matmul(A, A.transpose())
    w, v = np.linalg.eig(C)
    minval = np.argmin(w)
    minvec = v[:, minval]
    a, b, c = minvec[0], minvec[1], minvec[2]
    d = -(a * meanx + b * meany + c * meanz)
    print("Fit plane result: ", a, b, c, d)
    return a, b, c, d


def fit_plane_with_origin(x, y, z):
    return fit_plane(
        x=np.append(x, 0),
        y=np.append(y, 0),
        z=np.append(z, 0),
    )


def get_plane_wireframe(
        point: np.array,
        normal: np.array,
):
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    a, b, c = normal[0], normal[1], normal[2]
    d = -point.dot(normal)

    print(f"Plot plane {a:.2f},{b:.2f},{c:.2f}")

    if np.abs(c) >= 1e-3:
        X, Y = np.meshgrid(x, y)
        Z = (a * X + b * Y + d) / (-c)
        return X, Y, Z

    if np.abs(b) >= 1e-3:
        X, Z = np.meshgrid(x, y)
        Y = (a * X + c * Z + d) / (-b)
        return X, Y, Z

    Y, Z = np.meshgrid(x, y)
    X = (b * Y + c * Z + d) / (-a)
    return X, Y, Z


def cylinder_alignment_energy2_xyz(
        c,
        w,
        r2: float,
        x,
        y,
        z,
):
    points = np.vstack((x, y, z))
    # print(points.shape)
    return cylinder_alignment_energy2(c, w, r2, points)


def cylinder_alignment_geomfitty_xyz(
        c,
        w,
        x,
        y,
        z,
):
    points = np.vstack((x, y, z))
    return cylinder_alignment_geomfitty(c,w,points)


def cylinder_alignment_geomfitty(
        c,
        w,
        points,
):
    """

    :param c:
    :param w:
    :param points: 3xN array
    :return:
    """
    residuals = cylinder_fit_residuals(
        anchor_direction=np.concatenate((c, w)),
        points=points.transpose(),
        weights=None,
    )
    # print(points.shape)
    return 0.5 * np.sum(residuals ** 2)


def cylinder_estimate_r(
        c,
        w,
        x,
        y,
        z,
):
    points = np.vstack((x, y, z)).transpose()
    distances = distances_to_line(
        anchor=c,
        direction=w,
        points=points,
    )
    radius = np.average(distances)
    return radius



def cylinder_alignment_energy2_unoptimized(
        c,
        w,
        r2: float,
        points: np.array,
):
    """
    Eval (L2)^2 alignment energy for points. UNOPTIMIZED
    :param c:
    :param w:
    :param r2:
    :param points: array (3, n_points)
    :return:
    """
    c = np.array(c).reshape(3, 1)
    w = np.array(w).reshape(3, 1)

    value = np.zeros((1, 1))
    n_data_points = points.shape[1]
    # print(points.shape)
    for i in range(n_data_points):
        xi = points[:, i].reshape(3, 1)
        wwt = np.matmul(w, w.transpose())
        iwwt = np.identity(3) - wwt
        value += (np.matmul((xi - c).transpose(), np.matmul(iwwt, xi - c)) - r2) ** 2
    return value[0, 0]


def cylinder_alignment_energy2(
        c,
        w,
        r2: float,
        points,
        weight_norm_w=0.1,
):
    """
    Eval L2 alignment energy for points. Optimized with einsum
    :param c:
    :param w:
    :param r2:
    :param points: array (3, n_points)
    :param weight_norm_w:
    :return:
    """
    c = np.array(c).reshape(3, 1)
    w = np.array(w).reshape(3, 1)
    wwt = np.matmul(w, w.transpose())
    iwwt = np.identity(3) - wwt
    n_data_points = points.shape[1]
    average_point = np.mean(points, axis=1)
    xc = points - c
    # iwwtxc = np.einsum("ki,il->kl", iwwt, xc)
    # print(iwwtxc.shape)
    # xciwwtxc = np.einsum("ij,ij->j", xc, iwwtxc)
    # print(xciwwtxc.shape)
    # value = np.sum((xciwwtxc - r2)**2)

    # iwwtxc = np.einsum("ki,il->kl", iwwt, xc)
    temp = np.einsum("kl,ki,il->l", xc, iwwt, xc)
    value = np.sum((temp - r2)**2)
    value += weight_norm_w * n_data_points * ((np.linalg.norm(w)**2 - 1)**2)
    # value += weight_norm_w * n_data_points * (np.linalg.norm(c - average_point)**2)
    return value


def cylinder_alignment_energy_dist(
        c,
        w,
        points,
):
    """
    Eval L2 alignment energy for points. Optimized with einsum
    :param c:
    :param w:
    :param points: array (3, n_points)
    :return:
    """
    c = np.array(c).reshape(3, 1)
    what = np.array(w).reshape(3, 1) / np.linalg.norm(w)
    wwt = np.matmul(what, what.transpose())
    iwwt = np.identity(3) - wwt
    n_data_points = points.shape[1]
    xc = points - c
    Di2 = np.einsum("kl,ki,il->l", xc, iwwt, xc)  # contains squared distances
    if np.min(Di2) < 0:
        print(c)
        print(w)
        print(np.linalg.norm(w))
        print(wwt)
        print(iwwt)
        print(np.min(Di2))
        print(np.argmin(Di2))
        raise Exception("SQRTnegative")

    Di = np.sqrt(Di2)
    averageDi = np.mean(Di)
    value = 0.5 * np.sum((Di - averageDi)**2) + 100 * \
            (
                np.sum(1 - w.transpose().dot(w)) ** 2
            )
    # return value
    return value


def jacobian_params_cylinder_alignment_energy_dist(
        c: np.array,
        w: np.array,
        points: np.array,
):
    c = np.array(c).reshape(3, 1)
    w_hat = np.array(w).reshape(3, 1) / np.linalg.norm(w)
    wwt = np.matmul(w_hat, w_hat.transpose())
    iwwt = np.identity(3) - wwt
    n_data_points = points.shape[1]
    xc = points - c
    Di2 = np.einsum("kl,ki,il->l", xc, iwwt, xc)  # contains squared distances
    Di = np.sqrt(Di2)
    meanD = np.mean(Di)
    Di_meanD = (Di - meanD).reshape(1,-1)

    dDi2dC = -2 * np.matmul(iwwt, xc)
    dDidC = 0.5 * (1 / Di) * dDi2dC
    dmeanDdC = (1 / n_data_points) * np.sum(dDidC, axis=1)
    dDi_meanD_dC = dDidC - dmeanDdC.reshape(3, 1)
    dEdC = 0.5 * np.sum(2 * Di_meanD * dDi_meanD_dC, axis=1)

    temp = np.matmul(xc.transpose(), w_hat)
    dDi2dW = -2 * xc * temp.transpose()
    dDidW = 0.5 * (1 / Di) * dDi2dW
    dmeanDdW = (1 / n_data_points) * np.sum(dDidW, axis=1)
    dDi_meanD_dWhat = dDidW - dmeanDdW.reshape(3, 1)
    dDi_meanD_dW = np.matmul(iwwt, dDi_meanD_dWhat) / np.linalg.norm(w)
    dEdW = 0.5 * np.sum(2 * Di_meanD * dDi_meanD_dW, axis=1) + 100 * 4 * (np.sum(w.transpose().dot(w)) - 1) * w.reshape(3)
    return dEdC, dEdW


def jacobian_points_cylinder_alignment_energy_dist(
        c: np.array,
        w: np.array,
        points: np.array,
):
    c = np.array(c).reshape(3, 1)
    w_hat = np.array(w).reshape(3, 1) / np.linalg.norm(w)
    wwt = np.matmul(w_hat, w_hat.transpose())
    iwwt = np.identity(3) - wwt
    n_data_points = points.shape[1]
    xc = points - c
    Di2 = np.einsum("kl,ki,il->l", xc, iwwt, xc)  # contains squared distances
    Di = np.sqrt(Di2)
    meanD = np.mean(Di)
    Di_meanD = (Di - meanD).reshape(1, -1)

    dDi2dXi = 2 * np.matmul(iwwt, xc)
    dDidXi = 0.5 * (1 / Di) * dDi2dXi
    dmeanDdXi = (1 / n_data_points) * dDidXi
    dEdXi = Di_meanD * dDidXi
    return dEdXi


def cylinder_alignment_energy2_jacobian(
        c,
        w,
        r2: float,
        points,
        weight_norm_w=0.1,
):
    component_c = np.zeros((3, 1), dtype=float)
    component_w = np.zeros((3, 1), dtype=float)
    component_r2 = np.zeros((1, 1), dtype=float)
    average_point = np.mean(points, axis=1)
    n_data_points = points.shape[1]
    wwt = np.matmul(w, w.transpose())
    iwwt = np.identity(3) - wwt
    # TODO: rewrite with eigsum
    for i in range(n_data_points):
        xi = points[:, i].reshape(3, 1)
        err = (np.matmul((xi - c).transpose(), np.matmul(iwwt, xi - c)) - r2)
        component_c += -4 * err * np.matmul(iwwt, xi - c)
        component_w += -4 * err * np.matmul(w.transpose(), xi - c) * (xi - c)
        component_r2 += -2 * err
    component_w += n_data_points * weight_norm_w * 4 * (np.linalg.norm(w) - 1) * w
    # component_c += n_data_points * weight_norm_w * 2 * (c - average_point)
    result = np.vstack((component_c, component_w, component_r2))
    return result[:, 0]


class CylinderFit:
    def __init__(
            self,
            points: np.array,
            normals: np.array = None,
    ):
        assert points.shape[0] == 3
        self.points = points
        self.normals = normals
        if normals is not None:
            assert normals.shape[0] == 3
        self.n_data_points = points.shape[1]

    def estimateW_from_normals(self, plot=False):
        assert self.normals is not None
        plane_a, plane_b, plane_c, plane_d = fit_plane_with_origin(
            x=self.normals[0, :],
            y=self.normals[1, :],
            z=self.normals[2, :],
        )
        estimateW = np.array([plane_a, plane_b, plane_c])

        if plot:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            plt.title("Gauss image")

            plot_gauss_image(ax, pts=self.normals, show=False)

            X, Y, Z = get_plane_wireframe(
                point=np.zeros(3),
                normal=estimateW,
            )
            ax.plot_wireframe(X, Y, Z,
                              label="estimated",
                              color="red",
                              # rstride=2, cstride=2,
                              )

            plt.legend()
            plt.show()
            plt.close()

        return estimateW

    def estimate_c_r_from_projection(
            self,
            estimateW,
            plot=False,
    ):
        points_baricenter = np.mean(self.points, axis=1)
        vp0, vp1, vp2 = get_triplet(w=estimateW)
        projection_1 = vp1.reshape(1, 3).dot(self.points - points_baricenter.reshape(3, 1))
        projection_2 = vp2.reshape(1, 3).dot(self.points - points_baricenter.reshape(3, 1))

        projection_data = np.vstack(
            [
                projection_1,
                projection_2,
            ]
        )
        estimated_projection_r, estimated_projection_2d_c = skg.nsphere_fit(projection_data, axis=0, scaling=True)
        estimated_projection_3d_c = points_baricenter + estimated_projection_2d_c[0] * vp1 + estimated_projection_2d_c[1] * vp2

        if plot:
            print("projection_data.shape: ", projection_data.shape)
            print("projected_r found: ", estimated_projection_r)
            print("projectrion 2d center found: ", estimated_projection_2d_c)

            plt.figure()
            plt.scatter(projection_1, projection_2)
            x, y = get_xy_circle(center=estimated_projection_2d_c, r=estimated_projection_r)
            plt.plot(x, y, color="gold")
            plt.axis("equal")
            plt.show()

        return estimated_projection_3d_c, estimated_projection_r

    def objectiveE(self, params, weight_norm_w=0):
        c = np.array([params[0], params[1], params[2]]).reshape(3, 1)
        w = np.array([params[3], params[4], params[5]]).reshape(3, 1)
        r2 = params[6]
        return cylinder_alignment_energy2(
            c=c,
            w=w,
            r2=r2,
            points=self.points,
            weight_norm_w=weight_norm_w,
        )

    def jacobian(self, params, weight_norm_w=0):
        c = np.array([params[0], params[1], params[2]]).reshape(3, 1)
        w = np.array([params[3], params[4], params[5]]).reshape(3, 1)
        r2 = params[6]
        return cylinder_alignment_energy2_jacobian(
            c=c,
            w=w,
            r2=r2,
            points=self.points,
            weight_norm_w=weight_norm_w,
        )

    def minimize(self, x0=None, debug=False, weight_norm_w=0.1):
        average = np.mean(self.points, axis=1)
        print("average ", average.shape)
        if x0 is None:
            x0 = np.array([average[0], average[1], average[2], 2, 0.5, 1, 1])
        # TODO: use lambdas to pass default weight_norm_w
        result = spo.minimize(
            self.objectiveE,
            x0=x0,
            # jac=self.jacobian,
            method="L-BFGS-B",
            options={
                # "maxiter": 20,
                "maxfun": 1000000,
                "disp": True,
            }
        )
        if debug:
            print(result)
            print("solution: ", result.x)
            print("best value: ", result.fun)
        sol = result.x
        predC = sol[:3]
        predW = sol[3:6]
        predr2 = sol[6]
        return predC, predW, predr2

    def pyomo_fit(
            self,
            estimateW=None,
    ):
        if estimateW is None:
            print("\n==== estimate W ====")
            estimateW = self.estimateW_from_normals()
        else:
            estimateW = np.array([estimateW[0], estimateW[1], estimateW[2],])

        print("\n==== estimate c and r ====")
        estimateC, estimateR = self.estimate_c_r_from_projection(
            estimateW=estimateW,
        )

        model = pyo.ConcreteModel()
        model.c = pyo.Var([0,1,2], initialize=estimateC.tolist())
        model.w = pyo.Var([0,1,2], initialize=estimateW.tolist())
        model.r2 = pyo.Var(initialize=estimateR**2, domain=pyo.NonNegativeReals)

        model.xx = pyo.Var(initialize=1.5)
        def rosenbrock(m):
            return (1.6 - m.xx) ** 2

        model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)
        for i in range(self.points.shape[1] // 100):
            # TODO: something smarter than this (to decrease number of terms)
            x, y, z = self.points[0, i*100], self.points[1, i*100], self.points[2, i*100]
            model.obj += cylinder_distance(
                x=x, y=y, z=z,
                cx=model.c[0], cy=model.c[1], cz=model.c[2],
                w1=model.w[0], w2=model.w[1], w3=model.w[2],
                r2=model.r2,
            )

        def wRule(m):
            return m.w[0] * m.w[0] + m.w[1] * m.w[1] + m.w[2] * m.w[2] == 1

        model.Boundx = pyo.Constraint(rule=wRule)

        solver = pyo.SolverFactory('ipopt')
        solver.options['max_iter'] = 4000
        status = solver.solve(model, tee=True, report_timing=True)

        predC = np.array([pyo.value(model.c[0]), pyo.value(model.c[1]), pyo.value(model.c[2]),])
        predW = np.array([pyo.value(model.w[0]), pyo.value(model.w[1]), pyo.value(model.w[2]),])
        predr2 = pyo.value(model.r2)

        return predC, predW, predr2

    def fit(
            self,
            plot=False,
            estimateW=None,
    ):
        if estimateW is None:
            print("\n==== estimate W ====")
            estimateW = self.estimateW_from_normals(plot=plot)
        else:
            estimateW = np.array([estimateW[0], estimateW[1], estimateW[2],])
        # estimateW = np.random.rand(3) * 2 - 1
        # estimateW = self.estimateW_from_normals(plot=plot)
        # estimateW = np.array([0.8, 0, 0.6])
        # estimateW = estimateW / np.linalg.norm(estimateW)

        print("\n==== estimate c and r ====")
        estimateC, estimateR = self.estimate_c_r_from_projection(
            estimateW=estimateW,
            plot=plot,
        )

        print("\n==== x0 ====")
        x0 = np.array([
            estimateC[0],
            estimateC[1],
            estimateC[2],
            estimateW[0],
            estimateW[1],
            estimateW[2],
            estimateR ** 2
        ])
        print("x0: ", x0)
        print("Objective at x0:", cylinder_alignment_geomfitty_xyz(
            c=estimateC,
            w=estimateW,
            x=self.points[0, :],
            y=self.points[1, :],
            z=self.points[2, :],
        ))
        # print("Objective at x0 (ww=1):", self.objectiveE(params=x0, weight_norm_w=1))
        # print("Objective at x0 (ww=0):", self.objectiveE(params=x0, weight_norm_w=0))
        # print("Objective at x0 (ww=0.1):", self.objectiveE(params=x0, weight_norm_w=0.1))

        print("\n==== SOLVE ====")

        cyl = cylinder_fit(
            points=self.points.transpose(),
            initial_guess=x0[:6],
        )
        # optC, optW, optr2 = self.minimize(x0=x0, debug=False)
        optC, optW, optr2 = cyl.anchor_point, cyl.direction, cyl.radius ** 2
        print("Objective at x_fin:", cylinder_alignment_geomfitty_xyz(
            c=optC,
            w=optW,
            x=self.points[0, :],
            y=self.points[1, :],
            z=self.points[2, :],
        ))
        betterC = optC + optW * (self.points.mean(axis=1) - optC).dot(optW)

        if plot:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            plt.title("Cylinder fit")
            ax.scatter3D(self.points[0, :], self.points[1, :], self.points[2, :], c="blue")
            # ax.scatter3D(trueC[0], trueC[1], trueC[2], marker="^", c='blue', label="GT")
            ax.scatter3D(betterC[0], betterC[1], betterC[2], marker="^", c='red', label="optC")
            ax.plot3D(
                [(betterC - optW)[0], (betterC + optW)[0]],
                [(betterC - optW)[1], (betterC + optW)[1]],
                [(betterC - optW)[2], (betterC + optW)[2]],
                c="red", label="optW"
            )
            ax.scatter3D(estimateC[0], estimateC[1], estimateC[2], marker="^", c='orange', label="estimC")
            ax.plot3D(
                [(estimateC - estimateW)[0], (estimateC + estimateW)[0]],
                [(estimateC - estimateW)[1], (estimateC + estimateW)[1]],
                [(estimateC - estimateW)[2], (estimateC + estimateW)[2]],
                c="orange", label="estimW"
            )
            plot_cylinder(ax, pointC=betterC, directionW=optW, r=np.sqrt(np.abs(optr2)))

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # ax.set_xlim(-2, 2)
            # ax.set_ylim(-2, 2)
            # ax.set_zlim(-7, -3)

            ax.set_aspect('equal')

            plt.legend()
            plt.show()
            plt.close()

        return betterC, optW, optr2

    def plot_scatter(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.points[0, :], self.points[1, :], self.points[2, :], c=self.points[2, :], cmap='Greens')
        plt.show()

    def plot_normals(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.title("Gauss image")
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:20j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="gray", linewidths=0.5)
        ax.scatter3D([0], [0], [0], color='k')
        ax.scatter3D(self.normals[0, :], self.normals[1, :], self.normals[2, :], c="blue", label="normals")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-1, 0, 1])

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.legend()
        plt.show()
        plt.close()

    def plot_data(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.points[0, :], self.points[1, :], self.points[2, :], color='green', label="points")
        # ax.quiver(
        #     self.points[0, :], self.points[1, :], self.points[2, :],
        #     np.zeros(50), np.ones(50), np.zeros(50),
        #     # length=0.1,
        #     # normalize=True,
        #     length=2, normalize=True
        # )
        for i in range(self.points.shape[1]):
            scale = 2
            ax.plot(
                [self.points[0, i], self.points[0, i] + scale * self.normals[0, i]],
                [self.points[1, i], self.points[1, i] + scale * self.normals[1, i]],
                [self.points[2, i], self.points[2, i] + scale * self.normals[2, i]],
                color='grey',
            )
        ax.set_xlim3d(0, 4)
        ax.set_ylim3d(0, 4)
        # ax.set_zlim3d(0, 0.8)
        plt.show()

    @staticmethod
    def find_closest_point(
            x: float,
            y: float,
            z: float,
            c: np.array,
            w: np.array,
            r2: float,
            debug=False,
    ):
        uw = w / np.linalg.norm(w)  # unit w
        xi = np.array([x, y, z])
        xic = xi - c
        projection = np.dot(xic, uw) * uw
        axis_closest_point = c + projection
        dist = np.linalg.norm(xi - axis_closest_point)
        direction_to_axis = (axis_closest_point - xi) / dist
        dd = dist - np.sqrt(r2)
        newpoint = xi + dd * direction_to_axis
        return newpoint

    @staticmethod
    def project_point_on_cyl(
            x: float,
            y: float,
            z: float,
            t: np.array,
            c: np.array,
            w: np.array,
            r2: float,
            debug=False,
    ):
        """
        project point(x,y,z) to the closest point on a cylinder along ray t. cylinder params c, w, r2
        :param x:
        :param y:
        :param z:
        :param t: projection direction, array (3)
        :param c: cone vertex, array (3)
        :param w: cone axis, array (3)
        :param r2:
        :param debug:
        :return:
        """
        point1, point2, status = CylinderFit.find_points_along_ray(x, y, z, t, c, w, r2)
        if not status:
            if debug:
                print("projection does not intersect the cylinder")
            raise Exception("NoIntersection")
        else:
            return point1, point2

    @staticmethod
    def find_points_along_ray(
            x: float,
            y: float,
            z: float,
            t: np.array,
            c: np.array,
            w: np.array,
            r2: float,
    ):
        """
        Given a point (x,y,z) cast a ray along (t) and find 2 intersection points.
        If intersection, return (point1, point2, True)
        If no intersection, return (point3, point3, False) where point3 is the closest point on a cylinder along ray 
        :param x:
        :param y:
        :param z:
        :param t:
        :param c:
        :param w:
        :param r2:
        :return:
        """
        uw = w / np.linalg.norm(w)  # unit w
        ut = t / np.linalg.norm(t)  # unit t
        sol = skew_lines_dist(
            unit_a=ut,
            unit_b=uw,
            point_a=np.array([x, y, z]),
            point_b=c,
        )
        if sol is None:
            raise Exception(f"These skew lines are parallel! {ut} {uw}")
        dist = sol[2]
        if dist ** 2 > r2:
            point3 = np.array([x, y, z]) + sol[0] * ut  # point on the projection line, closest to the axis
            return point3, point3, False
        else:
            cos2angle = np.dot(uw, ut) ** 2
            sin2angle = 1 - cos2angle
            step2 = (r2 - dist ** 2) / sin2angle
            step = np.sqrt(step2)
            point1 = np.array([x, y, z]) + (sol[0] - step) * ut
            point2 = np.array([x, y, z]) + (sol[0] + step) * ut
            point3 = np.array([x, y, z]) + sol[0] * ut  # point on the projection line, closest to the axis
            point4 = c + sol[1] * uw  # point on the axis, closest to the projection line
            return point1, point2, True

    def plot_solution(
            self,
            C: np.array,
            W: np.array,
            r2: float,
    ):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.points[0, :], self.points[1, :], self.points[2, :], c=self.points[2, :], cmap='Greens')
        plot_cylinder(ax, pointC=C, directionW=W, r=np.sqrt(np.abs(r2)))
        ax.plot3D(
            [(C - W)[0], (C + W)[0]],
            [(C - W)[1], (C + W)[1]],
            [(C - W)[2], (C + W)[2]],
            c="orange", label="estimW"
        )
        plt.show()


def experiment():
    np.set_printoptions(precision=3)

    trueC = np.array((12, 3, 8), dtype=float)
    trueW = np.array((2, 1, 1), dtype=float)
    trueW = trueW / np.linalg.norm(trueW)
    truer2 = 36

    n_points = 5

    # np.random.seed(0)
    # random.seed(0)

    data, normals = generate_cylinder_points(
        trueC, trueW, truer2,
        n=n_points,
    )

    data += (np.random.rand(data.shape[0], data.shape[1]) - 0.5) * 0.01 * np.sqrt(truer2)
    normals += (np.random.rand(data.shape[0], data.shape[1]) - 0.5) * 0.2
    normals = normals / np.linalg.norm(normals, axis=0)
    print("normalize shape", np.linalg.norm(normals, axis=0).shape)
    print("data shape: ", data.shape)

    gauss_image_normals_planefit = fit_plane_with_origin(
        x=normals[0, :],
        y=normals[1, :],
        z=normals[2, :],
    )
    a, b, c, d = gauss_image_normals_planefit
    estimateW = np.array([a, b, c])

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # plt.title("Gauss image")
    # u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:20j]
    # x = np.cos(u) * np.sin(v)
    # y = np.sin(u) * np.sin(v)
    # z = np.cos(v)
    # ax.plot_wireframe(x, y, z, color="gray", linewidths=0.5)
    # ax.scatter3D([0], [0], [0], color='k')
    # ax.scatter3D(normals[0, :], normals[1, :], normals[2, :], cmap="Blues")
    # X, Y, Z = get_plane_wireframe(
    #     point=np.zeros(3),
    #     normal=trueW,
    # )
    # ax.plot_wireframe(X, Y, Z,
    #                   label="GT plane",
    #                   color="green",
    #                   # rstride=2, cstride=2,
    #                   )
    # X, Y, Z = get_plane_wireframe(
    #     point=np.zeros(3),
    #     normal=estimateW,
    # )
    # ax.plot_wireframe(X, Y, Z,
    #                   label="estimated",
    #                   color="red",
    #                   # rstride=2, cstride=2,
    #                   )
    # ax.set_xticks([-1, 0, 1])
    # ax.set_yticks([-1, 0, 1])
    # ax.set_zticks([-1, 0, 1])
    #
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)
    # plt.legend()
    # plt.show()
    # plt.close()

    print("\n==== SETUP ====")

    cylfit = CylinderFit(points=data)

    testpoint = np.array([12, 3, 8, 1, 1, 1, 4])
    print("objective function shape: ", cylfit.objectiveE(params=np.array([0, 0, 0, 1, 0.5, 1, 4])).shape)

    print("Geomfitty energy:", cylinder_alignment_geomfitty_xyz(
        c=trueC,
        w=estimateW,
        x=data[0, :],
        y=data[1, :],
        z=data[2, :],
    ))

    print("My computed energy: ", cylinder_alignment_energy_dist(
        c=trueC,
        w=estimateW,
        points=data,
    ))

    my_num_grad = spo.approx_fprime(
        xk=testpoint,
        f=lambda x: cylinder_alignment_energy_dist(
            c=x[:3],
            w=x[3:6],
            points=data,
        ),
    )
    print("my num grad parameters: ", my_num_grad)

    geomfitty_num_grad = spo.approx_fprime(
        xk=testpoint,
        f=lambda p: cylinder_alignment_geomfitty_xyz(
            c=p[:3],
            w=p[3:6],
            x=data[0, :],
            y=data[1, :],
            z=data[2, :],
        ),
    )
    print("geomfitty num grad parameters: ", my_num_grad)

    mygrad = jacobian_params_cylinder_alignment_energy_dist(
        points=data,
        c=testpoint[:3],
        w=testpoint[3:6],
    )
    print(mygrad)

    print("======================\n")

    cyl_points_flat = np.hstack(
        (
            data[0, :],
            data[1, :],
            data[2, :],
        )
    )
    print(cyl_points_flat.shape)

    my_num_grad = spo.approx_fprime(
        xk=cyl_points_flat,
        f=lambda x: cylinder_alignment_energy_dist(
            c=trueC,
            w=estimateW,
            points=np.vstack(
                (
                    x[:n_points],
                    x[n_points:2*n_points],
                    x[2*n_points:3*n_points]
                )
            ),
        ),
    )
    print("my num grad points: ", my_num_grad)

    my_grad = jacobian_points_cylinder_alignment_energy_dist(
        c=trueC,
        w=trueW,
        points=np.vstack(
            (
                cyl_points_flat[:n_points],
                cyl_points_flat[n_points:2 * n_points],
                cyl_points_flat[2 * n_points:3 * n_points]
            )
        ),
    )

    print("my formula grad points shape: ", my_grad.shape)
    print("my formula grad points: ", my_grad)

    print("======================\n")

    print("jacobian shape: ", cylfit.jacobian(params=np.array([0, 0, 0, 0, 0.5, 1, 4])).shape)
    print("objective function value: ", cylfit.objectiveE(params=testpoint))
    mygrad = cylfit.jacobian(params=testpoint)
    print("jacobian shape: ", mygrad.shape)

    grad_error = spo.check_grad(
        func=cylfit.objectiveE,
        grad=cylfit.jacobian,
        x0=testpoint,
    )
    mygrad_numerical = spo.approx_fprime(
        xk=testpoint,
        f=cylfit.objectiveE,
    )

    print("grad computed: ", mygrad)
    print("grad numerical: ", mygrad_numerical)
    print("diff: ", mygrad - mygrad_numerical)

    print("grad_error: ", grad_error)
    assert grad_error < 0.1
    return

    gt_objective_value = cylfit.objectiveE(
        params=np.array(
            [
                trueC[0], trueC[1], trueC[2],
                trueW[0], trueW[1], trueW[2],
                truer2,
            ]
        )
    )
    print("Objective at gt: ", gt_objective_value)

    print("\n==== PROJECTING ON 2d PLANE ====")
    points_baricenter = np.mean(data, axis=1)
    vp0, vp1, vp2 = get_triplet(w=estimateW)
    projection_1 = vp1.reshape(1, 3).dot(data - points_baricenter.reshape(3, 1))
    projection_2 = vp2.reshape(1, 3).dot(data - points_baricenter.reshape(3, 1))

    projection_data = np.vstack(
        [
            projection_1,
            projection_2,
        ]
    )
    print("projection_data.shape: ", projection_data.shape)
    estimated_projection_r, estimated_projection_2d_c = skg.nsphere_fit(projection_data, axis=0, scaling=True)
    print("projected_r found: ", estimated_projection_r)
    print("projectrion 2d center found: ", estimated_projection_2d_c)

    plt.figure()
    plt.scatter(projection_1, projection_2)
    x, y = get_xy_circle(center=estimated_projection_2d_c, r=estimated_projection_r)
    plt.plot(x, y, color="gold")
    plt.axis("equal")
    plt.show()

    # x, y = get_xy_circle(center=, r=r)

    estimateC = points_baricenter + estimated_projection_2d_c[0] * vp1 + estimated_projection_2d_c[1] * vp2

    print("\n==== x0 ====")

    x0 = np.array([
        estimateC[0],
        estimateC[1],
        estimateC[2],
        estimateW[0],
        estimateW[1],
        estimateW[2],
        estimated_projection_r**2
    ])
    print("x0: ", x0)
    print("Objective at x0:", cylfit.objectiveE(params=x0))
    print("\n==== OPTIMIZE ====")
    # predC, predW, predr2 = cylfit.fit_cylinder(x0=None, debug=True)
    predC, predW, predr2 = cylfit.minimize(x0=x0, debug=False)
    print("objective optimal found: ",
          cylfit.objectiveE(params=np.array([predC[0], predC[1], predC[2], predW[0], predW[1], predW[2], predr2])))
    betterC = predC + predW * (data[:, 0] - predC).dot(predW)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title("Cylinder fit")
    ax.scatter3D(data[0, :], data[1, :], data[2, :], c=data[2, :], cmap='Blues')
    ax.scatter3D(trueC[0], trueC[1], trueC[2], marker="^", c='blue', label="GT")
    ax.scatter3D(betterC[0], betterC[1], betterC[2], marker="^", c='red')
    ax.scatter3D(estimateC[0], estimateC[1], estimateC[2], marker="^", c='orange', label="estimate")
    ax.plot3D(
        [(trueC - trueW)[0], (trueC + trueW)[0]],
        [(trueC - trueW)[1], (trueC + trueW)[1]],
        [(trueC - trueW)[2], (trueC + trueW)[2]],
        c="blue",
    )
    ax.plot3D(
        [(estimateC - estimateW)[0], (estimateC + estimateW)[0]],
        [(estimateC - estimateW)[1], (estimateC + estimateW)[1]],
        [(estimateC - estimateW)[2], (estimateC + estimateW)[2]],
        c="orange",
    )
    plot_cylinder(ax, pointC=betterC, directionW=predW, r=np.sqrt(np.abs(predr2)))
    plt.legend()
    plt.show()
    plt.close()


def point_to_line_distance(
        point,
        lineA,
        lineT,
):
    uT = lineT / np.linalg.norm(lineT)
    dotprod = np.dot(point - lineA, uT)
    projpoint = lineA + dotprod * lineT
    return np.linalg.norm(point - projpoint)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    experiment()

    # trueC = np.array((12, 3, 8), dtype=float)
    # trueW = np.array((2, 1, 1), dtype=float)
    # trueW = trueW / np.linalg.norm(trueW)
    # truer2 = 36
    #
    # n_points = 50
    #
    # np.random.seed(0)
    # random.seed(0)
    #
    # data, normals = generate_cylinder_points(
    #     trueC, trueW, truer2,
    #     n=n_points,
    # )
    #
    # data += (np.random.rand(data.shape[0], data.shape[1]) - 0.5) * 0.1 * np.sqrt(truer2)
    # normals += (np.random.rand(data.shape[0], data.shape[1]) - 0.5) * 0.2
    # normals = normals / np.linalg.norm(normals, axis=0)
    # print("normalize shape", np.linalg.norm(normals, axis=0).shape)
    # print("data shape: ", data.shape)
    #
    # cylfit = CylinderFit(points=data, normals=normals)
    # solC, solW, solR2 = cylfit.fit(plot=False)
    #
    # print("trueC: ", trueC)
    # print("trueW: ", trueW)
    # print("truer2: ", truer2)
    # print("solC: ", solC)
    # print("solW: ", solW)
    # print("||solW||: ", np.linalg.norm(solW))
    # print("solR2: ", solR2)
    #
    # mypoint = data[:, 0] + np.array([5, 2, 0])
    # mydirection = np.array([0.3, 0.2, 1])
    # closestpoint = CylinderFit.find_closest_point(
    #     mypoint[0], mypoint[1], mypoint[2],
    #     c=solC, w=solW, r2=solR2,
    # )
    #
    # p1, p2 = cylfit.project_point_on_cyl(mypoint[0], mypoint[1], mypoint[2], t=mydirection, c=solC, w=solW, r2=solR2)
    # print("p1: ", p1)
    # cc = cylinder_alignment_energy2(c=solC, w=solW, r2=solR2, points=p1.reshape(3, 1))
    # print("p1 energy: ", cc)
    # print("p1 dist to C,W: ", point_to_line_distance(p1, lineA=solC, lineT=solW)**2)
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # plt.title("Cylinder fit")
    # ax.scatter3D(data[0, :], data[1, :], data[2, :], c="green")
    # ax.scatter3D(mypoint[0], mypoint[1], mypoint[2], c='k', marker="^")
    # ax.scatter3D(closestpoint[0], closestpoint[1], closestpoint[2], c='purple', marker="^")
    # ax.scatter3D(p1[0], p1[1], p1[2], c='blue', marker="^")
    # ax.scatter3D(p2[0], p2[1], p2[2], c='blue', marker="^")
    # # ax.scatter3D(trueC[0], trueC[1], trueC[2], marker="^", c='blue', label="GT")
    # ax.scatter3D(solC[0], solC[1], solC[2], marker="^", c='red', label="C")
    # ax.plot3D(
    #     [(solC - solW)[0], (solC + solW)[0]],
    #     [(solC - solW)[1], (solC + solW)[1]],
    #     [(solC - solW)[2], (solC + solW)[2]],
    #     c="red", label="optW"
    # )
    #
    # ax.plot3D(
    #     [(mypoint - mydirection)[0], (mypoint + 20 * mydirection)[0]],
    #     [(mypoint - mydirection)[1], (mypoint + 20 * mydirection)[1]],
    #     [(mypoint - mydirection)[2], (mypoint + 20 * mydirection)[2]],
    #     c="k", label="line"
    # )
    #
    # plot_cylinder(ax, pointC=solC, directionW=solW, r=np.sqrt(np.abs(solR2)), height=10)
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # ax.set_aspect('equal')
    #
    # plt.legend()
    # plt.show()
    # plt.close()
