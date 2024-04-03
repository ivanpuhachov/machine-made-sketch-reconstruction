import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
from pyomo_distances import sphere_distance
import pyomo.environ as pyo

from test_cylinder_fit import get_triplet, solve_quadratic_equation


def generate_sphere_points(
        c: np.array,
        r2: float,
        e0: np.array,
        theta0: float,
        theta1: float,
        n=20,
):
    e0, e1, e2 = get_triplet(w=e0)
    phi0 = np.pi / 6
    phi1 = 2 * np.pi / 3
    random_theta = theta0 + (theta1 - theta0) * np.random.rand(1, n)
    random_phi = phi0 + (phi1 - phi0) * np.random.rand(1, n)
    point_directions = e0.reshape(3, 1) * np.cos(random_theta) + \
                       e1.reshape(3, 1) * (np.cos(random_phi) * np.sin(random_theta)) + \
                       e2.reshape(3, 1) * (np.sin(random_phi) * np.sin(random_theta))
    points = c.reshape(3, 1) + np.sqrt(r2) * point_directions
    normals = - point_directions
    return points, normals


def sphere_evalZ(
        x,
        y,
        z,
        c,
        r2,
):
    newz = np.copy(z)
    for i_sphere_point in range(len(x)):
        cx, cy, cz = x[i_sphere_point], y[i_sphere_point], z[i_sphere_point]
        cp = np.array([cx, cy, cz])
        try:
            res = SphereFit.project_point_on_sphere(
                x=cx,
                y=cy,
                z=cz,
                t=np.array([0, 0, 1]),
                c=c,
                r2=r2,
            )
            point1, point2 = res
            d1, d2 = np.linalg.norm(cp - point1), np.linalg.norm(cp - point2)
            if d1 < d2:
                newz[i_sphere_point] = point1[2]
            else:
                newz[i_sphere_point] = point2[2]
        except Exception as e:
            print(e)
            continue
    return newz


def sphere_find_closest(
        x,
        y,
        z,
        c,
        r2,
):
    newx = np.copy(x)
    newy = np.copy(y)
    newz = np.copy(z)
    for i_sphere_point in range(len(x)):
        cx, cy, cz = x[i_sphere_point], y[i_sphere_point], z[i_sphere_point]
        newp = SphereFit.find_closest_point(
            x=cx,
            y=cy,
            z=cz,
            c=c,
            r2=r2,
        )
        newx[i_sphere_point] = newp[0]
        newy[i_sphere_point] = newp[1]
        newz[i_sphere_point] = newp[2]
    return newx, newy, newz


def sphere_projectZ_or_closest(
        x,
        y,
        z,
        c,
        r2,
):
    newx = np.copy(x)
    newy = np.copy(y)
    newz = np.copy(z)
    for i_sphere_point in range(len(x)):
        cx, cy, cz = x[i_sphere_point], y[i_sphere_point], z[i_sphere_point]
        cp = np.array([cx, cy, cz])
        try:
            res = SphereFit.project_point_on_sphere(
                x=cx,
                y=cy,
                z=cz,
                t=np.array([0, 0, 1]),
                c=c,
                r2=r2,
            )
            point1, point2 = res
            d1, d2 = np.linalg.norm(cp - point1), np.linalg.norm(cp - point2)
            if d1 < d2:
                newz[i_sphere_point] = point1[2]
            else:
                newz[i_sphere_point] = point2[2]
        except Exception as e:
            if str(e) == "NoIntersection":
                pp = SphereFit.find_closest_point(
                    x=cx,
                    y=cy,
                    z=cz,
                    c=c,
                    r2=r2,
                )
                newx[i_sphere_point] = pp[0]
                newy[i_sphere_point] = pp[1]
                newz[i_sphere_point] = pp[2]
            else:
                raise e
    return newx, newy, newz


def sphere_alignment_residuals(
        points,
        c,
        r2,
):
    A = np.hstack(
        (
            2 * points.transpose(),
            np.ones(shape=(points.shape[1], 1))
        )
    )
    x = np.array([c[0], c[1], c[2], r2 - c[0]**2 - c[1]**2 - c[2]**2]).reshape(4, 1)
    b = (np.linalg.norm(points, axis=0)**2).reshape(points.shape[1], 1)
    residuals = np.matmul(A, x) - b
    return residuals.reshape(points.shape[1])


def sphere_alignment_energy2(
        points,
        c,
        r2,
):
    res = sphere_alignment_residuals(points=points, c=c, r2=r2)
    return np.sum(res**2)


def grad_params_sphere_alignment_energy2(
        points: np.array,
        c: np.array,
        r2: float,
):
    """

    :param points: shape (3, N)
    :param c:
    :param r2:
    :return:
    """
    c = np.array(c).reshape(3, 1)
    xc = points - c
    distXiC2 = np.sum(xc**2, axis=0).reshape(1, -1)
    distXiC2_r2 = distXiC2 - r2
    dEdC = -4 * np.sum(distXiC2_r2 * xc, axis=1)
    dEdR2 = -2 * np.sum(distXiC2_r2)
    return dEdC, dEdR2


def grad_points_sphere_alignment_energy2(
        points: np.array,
        c: np.array,
        r2: float,
):
    """

    :param points: shape (3, N)
    :param c:
    :param r2:
    :return:
    """
    c = np.array(c).reshape(3, 1)
    xc = points - c
    distXiC2 = np.sum(xc**2, axis=0).reshape(1, -1)
    distXiC2_r2 = distXiC2 - r2
    dEdXi = 4 * distXiC2_r2 * xc
    return dEdXi


class SphereFit:
    def __init__(
            self,
            points: np.array,
            normals: np.array = None,
    ):
        assert points.shape[0] == 3
        if normals is not None:
            assert normals.shape == points.shape
        self.points = points
        self.normals = normals
        self.n_points = points.shape[1]

    @staticmethod
    def plot_sphere(
            c: np.array,
            r2: float,
            ax,
    ):
        r = np.sqrt(r2)
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:20j]
        x = r * np.cos(u) * np.sin(v) + c[0]
        y = r * np.sin(u) * np.sin(v) + c[1]
        z = r * np.cos(v) + c[2]
        ax.plot_wireframe(x, y, z, color="gray", linewidths=0.5)
        ax.scatter3D([c[0]], [c[1]], [c[2]], color='k')


    def minimize_least_squares(
            self,
            x0: np.array,
            debug=True,
    ):
        assert len(x0) == 4
        result = spo.least_squares(
            fun=lambda x: sphere_alignment_residuals(
                points=self.points,
                c=x[:3],
                r2=x[3]
            ),
            x0=x0,
            jac='3-point',
            method='lm',
            # method='trf',
            verbose=2,
        )
        sol = result.x
        c = sol[:3]
        r2 = sol[3]
        if debug:
            # print(result)
            print("solution: ", result.x)
        return c, r2

    def pyomo_fit(
            self,
    ):
        estimateC = np.mean(self.points, axis=1)
        estimater2 = np.sum((estimateC - self.points[:, 0]) ** 2)

        model = pyo.ConcreteModel()
        model.c = pyo.Var([0, 1, 2], initialize=estimateC.tolist())
        model.r2 = pyo.Var(initialize=estimater2, domain=pyo.NonNegativeReals)

        model.xx = pyo.Var(initialize=1.5)

        def rosenbrock(m):
            return (1.6 - m.xx) ** 2

        model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)

        for i in range(self.points.shape[1] // 100):
            # TODO: something smarter than this (to decrease number of terms)
            x, y, z = self.points[0, i], self.points[1, i], self.points[2, i]
            model.obj += sphere_distance(
                x=x, y=y, z=z,
                cx=model.c[0], cy=model.c[1], cz=model.c[2],
                r2=model.r2,
            )

        solver = pyo.SolverFactory('ipopt')
        solver.options['max_iter'] = 4000
        status = solver.solve(model, tee=True, report_timing=True)

        predC = np.array([pyo.value(model.c[0]), pyo.value(model.c[1]), pyo.value(model.c[2]), ])
        predr2 = pyo.value(model.r2)

        return predC, predr2

    def fit(
            self,
            plot=False,
            show=False,
    ):
        estimateC = np.mean(self.points, axis=1)
        estimater2 = np.sum((estimateC - self.points[:, 0])**2)
        x0 = np.array([estimateC[0], estimateC[1], estimateC[2], estimater2])
        x0res = sphere_alignment_residuals(points=self.points, c=estimateC, r2=estimater2)
        x0val = sphere_alignment_energy2(points=self.points, c=estimateC, r2=estimater2)
        c, r2 = self.minimize_least_squares(x0=x0, debug=True,)

        print("x0: ", x0)
        print("Objective at x0: ", x0val)
        print("Objective at xfin: ", sphere_alignment_energy2(points=self.points, c=c, r2=r2))
        print("optC: ", c)
        print("optR2: ", r2)

        if plot:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            plt.title("Sphere fit")
            ax.scatter3D(self.points[0, :], self.points[1, :], self.points[2, :], c='green')
            # for i in range(data.shape[1]):
            #     pp = data[:, i]
            #     nn = gtnormals[:, i]
            #     ax.plot3D(
            #         [pp[0], (pp + nn)[0]],
            #         [pp[1], (pp + nn)[1]],
            #         [pp[2], (pp + nn)[2]],
            #         c="gray",
            #     )
            SphereFit.plot_sphere(c=c, r2=r2, ax=ax)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.set_aspect('equal')
            # plt.legend()
            if show:
                plt.show()

        return c, r2

    @staticmethod
    def project_point_on_sphere(
            x: float,
            y: float,
            z: float,
            t: np.array,
            c: np.array,
            r2: float,
    ):
        ut = t / np.linalg.norm(t)

        pX = np.array([x, y, z]) - c
        closest_p = pX + ut * np.dot(-pX, ut)
        dist2 = np.linalg.norm(closest_p) ** 2
        if dist2 >= r2:
            raise Exception("NoIntersection")
            # return closest_p + c, closest_p + c

        t1, t2 = solve_quadratic_equation(
            a=ut[0] ** 2 + ut[1]**2 + ut[2]**2,
            b=2*(ut[0] * pX[0] + ut[1] * pX[1] + ut[2] * pX[2]),
            c=pX[0]**2 + pX[1]**2 + pX[2]**2 - r2,
        )

        point1 = pX + ut * t1 + c
        point2 = pX + ut * t2 + c
        return point1, point2

    @staticmethod
    def find_closest_point(
            x: float,
            y: float,
            z: float,
            c: np.array,
            r2: float,
    ):
        pointX = np.array([x, y, z])
        xc = c - pointX
        unit_xc = xc / np.linalg.norm(xc)
        dd = np.linalg.norm(xc)
        direction = dd - np.sqrt(r2)
        return pointX + unit_xc * direction


if __name__ == "__main__":
    trueC = np.array((1, 2, 3))
    trueR2 = 16
    n_points = 5
    data, gtnormals = generate_sphere_points(
        c=trueC,
        r2=trueR2,
        e0=np.array((1, 1, 1)),
        theta0=np.pi / 6,
        theta1=np.pi / 3,
        n=n_points,
    )

    estimateC = np.array([0, 0, 0])
    estimateR2 = 10
    estimateparams = np.hstack((estimateC, estimateR2))
    flat_points = np.hstack(
        (
            data[0, :],
            data[1, :],
            data[2, :],
        )
    )

    energy_value = sphere_alignment_energy2(
        points=data,
        c=estimateC,
        r2=estimateR2,
    )

    print("energy_value: ", energy_value)

    num_grad_sphere_alignment_energy2 = spo.approx_fprime(
        xk=estimateparams,
        f=lambda x: sphere_alignment_energy2(
            points=data,
            c=x[:3],
            r2=x[3],
        ),
    )

    print("num_grad_z2_energy params: ", num_grad_sphere_alignment_energy2)

    my_grad = grad_params_sphere_alignment_energy2(
        points=data,
        c=estimateparams[:3],
        r2=estimateparams[3],
    )

    print("my grad params: ", my_grad)

    print("======================")
    num_grad_sphere_alignment_energy2 = spo.approx_fprime(
        xk=flat_points,
        f=lambda x: sphere_alignment_energy2(
            points=np.vstack(
                (
                    x[:n_points],
                    x[n_points:2*n_points],
                    x[2*n_points:3*n_points],
                )
            ),
            c=estimateparams[:3],
            r2=estimateparams[3],
        ),
    )

    print("num_grad_z2_energy points: ", num_grad_sphere_alignment_energy2)

    my_grad = grad_points_sphere_alignment_energy2(
        points=np.vstack(
                (
                    flat_points[:n_points],
                    flat_points[n_points:2*n_points],
                    flat_points[2*n_points:3*n_points],
                )
            ),
        c=estimateparams[:3],
        r2=estimateparams[3],
    )
    print("my grad points: ", my_grad)


    print("======================")

    print("data.shape: ", data.shape)
    print("normals.shape: ", gtnormals.shape)

    spherefit = SphereFit(points=data, normals=gtnormals)
    optC, optR2 = spherefit.pyomo_fit()

    print(trueC, " -- opt ", optC)
    print(trueR2, " -- opt ", optR2)

    # mypoint = np.array([6, 4, 4.5])
    # myproj = np.array([-1, -2, -1])
    # p1, p2 = spherefit.project_point_on_sphere(
    #     x=mypoint[0],
    #     y=mypoint[1],
    #     z=mypoint[2],
    #     t=myproj,
    #     c=optC,
    #     r2=optR2,
    # )
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # plt.title("Sphere fit")
    # ax.scatter3D(data[0, :], data[1, :], data[2, :], c='green')
    # for i in range(data.shape[1]):
    #     pp = data[:, i]
    #     nn = gtnormals[:, i]
    #     ax.plot3D(
    #         [pp[0], (pp + nn)[0]],
    #         [pp[1], (pp + nn)[1]],
    #         [pp[2], (pp + nn)[2]],
    #         c="gray",
    #     )
    # SphereFit.plot_sphere(c=optC, r2=optR2, ax=ax)
    #
    # ax.scatter3D(mypoint[0], mypoint[1], mypoint[2], c='black', marker="*", label="p")
    # ax.scatter3D(p1[0], p1[1], p1[2], c='red', marker="*", label="p1")
    # ax.scatter3D(p2[0], p2[1], p2[2], c='red', marker="*", label="p2")
    # ax.plot3D(
    #     [mypoint[0], (mypoint + 2 * myproj)[0]],
    #     [mypoint[1], (mypoint + 2 * myproj)[1]],
    #     [mypoint[2], (mypoint + 2 * myproj)[2]],
    #     c="orange", label="proj"
    # )
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # ax.set_aspect('equal')
    # # plt.legend()
    # plt.show()
