import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

from test_cylinder_fit import get_triplet, fit_plane, plot_gauss_image, \
    get_plane_wireframe, skew_lines_dist, solve_quadratic_equation
from geomfitty.another_cone_fit import cone_fit_residuals, cone_fit_residuals_split
from pyomo_distances import cone_distance
import pyomo.environ as pyo


def generate_cone_points(
        v: np.array,
        u: np.array,
        theta: float,
        h0: float,
        h1: float,
        n=100,
):
    random_height = h0 + (h1 - h0) * np.random.rand(1, n)
    random_phi = np.pi * np.random.rand(1, n)
    radii = random_height * np.tan(theta)
    e0, e1, e2 = get_triplet(w=u)
    # points = random_height * e0.reshape(3, 1) + radii * (
    #         np.sin(random_phi) * e1.reshape(3, 1) + np.cos(random_phi) * e2.reshape((3,1))
    # )
    points = v.reshape(3, 1) + random_height * e0.reshape(3, 1) + radii * (
            np.sin(random_phi) * e1.reshape(3, 1) + np.cos(random_phi) * e2.reshape((3, 1))
    )
    normals = np.tan(theta) * (
            - np.sin(random_phi) * e1.reshape(3, 1) - np.cos(random_phi) * e2.reshape((3,1))
    ) + e0.reshape(3, 1) * np.tan(theta)**2
    return points, normals


def plot_cone(
        ax,
        v: np.array,
        u: np.array,
        theta: float,
        h0: float,
        h1: float,
):
    n_lines = 10
    n_points_on_circle = 37
    height = np.linspace(start=h0, stop=h1, num=n_lines, endpoint=True)
    radii = height * np.tan(theta)
    e0, e1, e2 = get_triplet(w=u)
    phi = np.linspace(-1 * np.pi, 1 * np.pi, n_points_on_circle).reshape(1, n_points_on_circle)
    for i in range(n_lines):
        points = v.reshape(3, 1) + \
                 height[i] * e0.reshape(3, 1) + \
                 radii[i] * e1.reshape(3, 1) * np.sin(phi) + \
                 radii[i] * e2.reshape(3, 1) * np.cos(phi)
        ax.plot3D(points[0, :], points[1, :], points[2, :], 'gray')
    n_sections = 6
    for i in range(n_sections):
        pphi = i * 2 * np.pi / n_sections
        points = v.reshape(3, 1) + \
                 height[[0, -1]] * e0.reshape(3, 1) + \
                 radii[[0, -1]] * e1.reshape(3, 1) * np.sin(pphi) + \
                 radii[[0, -1]] * e2.reshape(3, 1) * np.cos(pphi)
        ax.plot3D(points[0, :], points[1, :], points[2, :], 'gray')


def cone_alignment_energy2_vw(
        v: np.array,
        w: np.array,
        points: np.array,
):
    """
    compute alignment energy. NOTE: cone_alignment_energy2_vw * np.cos(theta) ^ 4 = cone_alignment_energy2_vutheta
    :param v:
    :param w:
    :param points: array (3, n_points)
    :return:
    """
    v = np.array(v).reshape(3, 1)
    w = np.array(w).reshape(3, 1)
    wwt = np.matmul(w, w.transpose())
    iwwt = np.identity(3) - wwt
    xv = points - v
    temp = np.einsum("kl,ki,il->l", xv, iwwt, xv)
    value = np.sum(temp ** 2)
    return value


def cone_alignment_energy_vutheta(
        v: np.array,
        u: np.array,
        theta: float,
        points: np.array,
):
    """

    :param v: np array, len 3, cone center
    :param u: np array, len 3, cone axis
    :param theta: float, angle
    :param points: 3xn array
    :return:
    """
    v = np.array(v).reshape(3, 1)
    u = np.array(u).reshape(3, 1)
    uut = np.matmul(u, u.transpose())
    icosuut = np.identity(3) * (np.cos(theta) ** 2) - uut
    xv = points - v
    return np.einsum("kl,ki,il->l", xv, icosuut, xv)


def cone_alignment_energy2_vutheta(
        v: np.array,
        u: np.array,
        theta: float,
        points: np.array,
):
    """
    compute alignment energy^2
    :param v: np array, len 3, cone center
    :param u: np array, len 3, cone axis
    :param theta: float, angle
    :param points: 3xn array
    :return:
    """
    temp = cone_alignment_energy_vutheta(v, u, theta, points)
    value = np.sum(temp ** 2)
    return value


def cone_alignment_energy2(
        v: np.array,
        u: np.array,
        theta: float,
        points: np.array,
        weight_norm_u=100.0,
):
    cone_en = cone_alignment_energy2_vutheta(v, u, theta, points)
    n_data_points = points.shape[1]
    cone_en += n_data_points * weight_norm_u * ((np.linalg.norm(u)**2 - 1)**2)
    return cone_en


def rotate_axis_to_z(
        u,
):
    """
    return u aligned with z-axis and transformation matrix (newpoints = np.matmul(points.transpose(), matr).transpose())
    :param u: np array (3)
    :return:
    """
    x, y, z = u[0], u[1], u[2]
    a = np.sqrt(x**2 + y**2)
    if a == 0:
        return u, np.eye(3)
    b = np.sqrt(x**2 + y**2 + z**2)
    rot1 = np.array([
        [y / a, x / a, 0],
        [-x / a, y / a, 0],
        [0, 0, 1]
    ])
    rot2 = np.array([
        [1, 0, 0],
        [0, z / b, a / b],
        [0, -a / b, z / b]
    ])
    newu = np.matmul(np.matmul(u, rot1), rot2)
    matr = np.matmul(rot1, rot2)
    return newu, matr


def cone_evalZ(
        x,
        y,
        z,
        v,
        u,
        theta,
        debug=True,
):
    newz = np.copy(z)
    for i_cone_point in range(len(x)):
        cx, cy, cz = x[i_cone_point], y[i_cone_point], z[i_cone_point]
        cp = np.array([cx, cy, cz])
        res = ConeFit.project_point_on_cone(
            x=cx,
            y=cy,
            z=cz,
            t=np.array([0, 0, 1]),
            v=v,
            u=u,
            theta=theta,
            debug=debug,
        )
        if res is None:
            print("None!: ", cp)
            continue
        p1, p2 = res
        d1, d2 = np.linalg.norm(cp - p1), np.linalg.norm(cp - p2)
        if d1 < d2:
            newz[i_cone_point] = p1[2]
        else:
            newz[i_cone_point] = p2[2]
        # if np.isnan(newz).any():
        #     print("NaN")
    print("cone projection done")
    return newz


class ConeFit:
    def __init__(
            self,
            points: np.array,
            normals: np.array = None,
            method="lm",
    ):
        """
        class to perform cone fit
        :param points:
        :param normals:
        :param method:
        """
        assert points.shape[0] == 3
        if normals is not None:
            assert normals.shape == points.shape
        self.points = points
        self.normals = normals
        self.n_points = points.shape[1]
        self.method = method

    @staticmethod
    def splitx0_vw(x0: np.array):
        v = np.array([x0[0], x0[1], x0[2]])
        w = np.array([x0[3], x0[4], x0[5]])
        return v, w

    @staticmethod
    def splitx0_vutheta(x0: np.array):
        v = np.array([x0[0], x0[1], x0[2]])
        u = np.array([x0[3], x0[4], x0[5]])
        theta = x0[6]
        return v, u, theta

    def objectiveE_vw(self, params):
        v, w = self.splitx0_vw(x0=params)
        return cone_alignment_energy2_vw(
            v=v,
            w=w,
            points=self.points,
        )

    def objectiveE_vutheta(self, params):
        v, u, theta = self.splitx0_vutheta(x0=params)
        return cone_alignment_energy2_vutheta(
            v=v,
            u=u,
            theta=theta,
            points=self.points,
        )

    def objectiveE_vutheta_uweight(self, params, weight_norm_u=1.0):
        v, u, theta = self.splitx0_vutheta(x0=params)
        return cone_alignment_energy2(
            v=v,
            u=u,
            theta=theta,
            points=self.points,
            weight_norm_u=weight_norm_u,
        )

    def residuals_vutheta(self, params):
        v, u, theta = self.splitx0_vutheta(x0=params)
        return cone_alignment_energy_vutheta(
            v=v,
            u=u,
            theta=theta,
            points=self.points,
        ) + 10.0 * (np.linalg.norm(u) - 1)

    def residuals_vutheta_weight(self, params, weight_norm_u=10.0):
        v, u, theta = self.splitx0_vutheta(x0=params)
        return cone_fit_residuals_split(
            theta=theta,
            axis=u,
            vertex=v,
            points=self.points.transpose(),
        ) + weight_norm_u * (np.linalg.norm(u) - 1)
        # return cone_alignment_energy_vutheta(
        #     v=v,
        #     u=u,
        #     theta=theta,
        #     points=self.points,
        # ) + weight_norm_u * (np.linalg.norm(u) - 1)

    def estimateUTheta_from_normals(self, plot=False):
        assert self.normals is not None
        plane_a, plane_b, plane_c, plane_d = fit_plane(
            x=self.normals[0, :],
            y=self.normals[1, :],
            z=self.normals[2, :],
        )
        dist = np.abs(plane_d) / np.sqrt(plane_a**2 + plane_b**2 + plane_c**2)
        estimateTheta = np.pi / 2 - np.arccos(dist)
        estimateU = np.array([plane_a, plane_b, plane_c])

        if plot:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            plt.title("Gauss image")

            plot_gauss_image(ax, pts=self.normals, show=False)

            X, Y, Z = get_plane_wireframe(
                point=np.mean(self.normals, axis=1),
                normal=estimateU,
            )
            ax.plot_wireframe(X, Y, Z,
                              label="estimated",
                              color="red",
                              # rstride=2, cstride=2,
                              )

            plt.legend()
            plt.show()
            plt.close()
        return estimateU, estimateTheta

    def minimize_least_squares(self, x0: np.array, debug=True):
        assert len(x0) == 7
        eps = 0.1
        result = spo.least_squares(
            # fun=self.residuals_vutheta,
            fun=self.residuals_vutheta_weight,
            x0=x0,
            jac='3-point',
            method='lm',
            # method='trf',
            verbose=2,
            # bounds=(
            #     [-np.inf, -np.inf, -np.inf, -1, -1, -1, eps],
            #     [np.inf, np.inf, np.inf, 1, 1, 1, np.pi / 2 - eps]
            # ),
        )
        sol = result.x
        v = sol[:3]
        u = sol[3:6]
        theta = sol[6]
        if debug:
            # print(result)
            print("solution: ", result.x)
            print("x0 value: ", self.objectiveE_vutheta(params=x0))
            # print("sol value: ", result.fun)
            print("v u theta cos(theta): ", v, u, theta, np.cos(theta))
        return v, u, theta

    def minimize_trust_constr(self, x0: np.array, debug=False):
        assert len(x0) == 6
        result = spo.minimize(
            self.objectiveE_vw,
            x0=x0,
            # method="L-BFGS-B",
            # TODO: hessian / jacobian?
            # TODO: Gauss-Newton?
            # TODO: better way to add constraints? Do we need them at all?
            method="trust-constr",
            constraints=[spo.NonlinearConstraint(
                fun=lambda x: np.array([x[3] ** 2 + x[4] ** 2 + x[5] ** 2]),
                lb=np.array([1.0]),
                ub=np.array([10000.0])
            )],
            options={
                "maxiter": 2000,
                # "maxfun": 1000000,
                "disp": debug,
            }
        )
        minV = result.x[:3]
        minW = result.x[3:6]
        minU = minW / np.linalg.norm(minW)
        minTheta = np.arccos(1 / np.linalg.norm(minW))
        if debug:
            # print(result)
            print("solution: ", result.x)
            print("x0 value: ", self.objectiveE_vw(params=x0))
            print("sol value: ", result.fun)
            print("v u theta cos(theta): ", minV, minU, minTheta, np.cos(minTheta))
        return minV, minU, minTheta

    def minimize_lbfgsb(self, x0: np.array, debug=False):
        assert len(x0) == 7
        result = spo.minimize(
            self.objectiveE_vutheta_uweight,
            x0=x0,
            # jac=self.jacobian,
            method="L-BFGS-B",
            options={
                "maxiter": 200,
                "maxfun": 1000000,
                "disp": debug,
            },
            bounds=[
                (None, None),
                (None, None),
                (None, None),
                (None, None),
                (None, None),
                (None, None),
                (np.pi/32, np.pi/2 - 0.01),
            ]
        )
        sol = result.x
        v = sol[:3]
        u = sol[3:6]
        theta = sol[6]
        if debug:
            # print(result)
            print("solution: ", result.x)
            print("x0 value: ", self.objectiveE_vutheta(params=x0))
            print("sol value: ", result.fun)
            print("v u theta cos(theta): ", v, u, theta, np.cos(theta))
        return v, u, theta

    def estimate_vtheta(
            self,
            estimateU,
    ):
        estimateV = np.mean(self.points, axis=1)
        model = pyo.ConcreteModel()
        model.v = pyo.Var([0, 1, 2], initialize=estimateV.tolist())
        model.theta = pyo.Var(initialize=0.3, domain=pyo.NonNegativeReals, bounds=(-0.1, 1.6))

        model.xx = pyo.Var(initialize=1.5)

        def rosenbrock(m):
            return (1.6 - m.xx) ** 2

        model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)
        for i in range(self.points.shape[1] // 20):
            i_point = i * 20
            # TODO: something smarter than this (to decrease number of terms)
            x, y, z = self.points[0, i_point], self.points[1, i_point], self.points[2, i_point]
            model.obj += cone_distance(
                x=x, y=y, z=z,
                vx=model.v[0], vy=model.v[1], vz=model.v[2],
                u1=estimateU[0], u2=estimateU[1], u3=estimateU[2],
                theta=model.theta,
            )

        solver = pyo.SolverFactory('ipopt')
        solver.options['max_iter'] = 4000
        solver.options["tol"] = 1e-5
        status = solver.solve(model, tee=False, report_timing=False)

        optV = np.array([pyo.value(model.v[0]), pyo.value(model.v[1]), pyo.value(model.v[2]), ])
        optTheta = pyo.value(model.theta)

        return optV, optTheta

    def pyomo_fit(
            self,
            estimateU=None,
    ):
        estimateV = np.mean(self.points, axis=1)
        if estimateU is None:
            estimateU, estimateTheta = self.estimateUTheta_from_normals(plot=False)
        else:
            estimateV, estimateTheta = self.estimate_vtheta(estimateU=estimateU / np.linalg.norm(estimateU))
        model = pyo.ConcreteModel()
        model.v = pyo.Var([0, 1, 2], initialize=estimateV.tolist())
        model.u = pyo.Var([0, 1, 2], initialize=estimateU.tolist())
        model.theta = pyo.Var(initialize=estimateTheta, domain=pyo.NonNegativeReals, bounds=(0, 1.6))

        model.xx = pyo.Var(initialize=1.5)

        def rosenbrock(m):
            return (1.6 - m.xx) ** 2

        model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)
        for i_point in range(self.points.shape[1] // 100):
            # TODO: something smarter than this (to decrease number of terms)
            x, y, z = self.points[0, i_point * 100], self.points[1, i_point * 100], self.points[2, i_point * 100]
            model.obj += cone_distance(
                x=x, y=y, z=z,
                vx=model.v[0], vy=model.v[1], vz=model.v[2],
                u1=model.u[0], u2=model.u[1], u3=model.u[2],
                theta=model.theta,
            )

        def wRule1(m):
            return m.u[0] * m.u[0] + m.u[1] * m.u[1] + m.u[2] * m.u[2] >= 0.9

        def wRule2(m):
            return m.u[0] * m.u[0] + m.u[1] * m.u[1] + m.u[2] * m.u[2] <= 1.1

        model.Boundx1 = pyo.Constraint(rule=wRule1)
        model.Boundx2 = pyo.Constraint(rule=wRule2)

        solver = pyo.SolverFactory('ipopt')
        solver.options['max_iter'] = 4000
        solver.options["tol"] = 1e-5
        # solver.options["OTOL"] = 1e-5
        status = solver.solve(model, tee=True, report_timing=True)

        optV = np.array([pyo.value(model.v[0]), pyo.value(model.v[1]), pyo.value(model.v[2]), ])
        optU = np.array([pyo.value(model.u[0]), pyo.value(model.u[1]), pyo.value(model.u[2]), ])
        optTheta = pyo.value(model.theta)

        return optV, optU, optTheta

    def fit(
            self,
            plot=False,
            show=False,
    ):
        print("\n==== x0 ====")
        estimateV = np.mean(self.points, axis=1)
        estimateU, estimateTheta = self.estimateUTheta_from_normals(plot=plot)
        estimateW = estimateU / np.cos(estimateTheta)
        x0 = np.concatenate((estimateV, estimateW))
        if (self.method == "lbfgsb") or (self.method == "lm"):
            x0 = np.concatenate((estimateV, estimateU, np.array([estimateTheta])))
            x0val = self.objectiveE_vutheta_uweight(params=x0)
        else:
            x0val = self.objectiveE_vw(params=x0)
        print("x0: ", x0)
        print("Objective at x0:", x0val)

        print("\n==== SOLVE ====")
        optV, optU, optTheta = estimateV, estimateU, np.pi/4
        if self.method == "lbfgsb":
            optV, optU, optTheta = self.minimize_lbfgsb(x0=x0, debug=True)
        if self.method == "lm":
            optV, optU, optTheta = self.minimize_least_squares(x0=x0, debug=True)
        if self.method == "trust-constr":
            optV, optU, optTheta = self.minimize_trust_constr(x0=x0, debug=True)
        # fix orientation
        if np.dot(optU, self.points[:,0] - optV) < 0:
            print("-> U orientation fix")
            optU = -optU
        optW = optU / np.cos(optTheta)

        print("estV :", estimateV)
        print("optV :", optV)
        print("||optU|| :", np.linalg.norm(optU))
        print("optU :", optU)
        print("estU :", estimateU)
        print("oW :", optW)
        print("estW :", estimateW)
        print("estTheta :", estimateTheta)
        print("optTheta :", optTheta)
        print("cos(optTheta) :", np.cos(optTheta))

        # optTheta = np.arccos(1 / np.linalg.norm(optW))

        if plot:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            plt.title("Cone fit")
            ax.scatter3D(self.points[0, :], self.points[1, :], self.points[2, :], c="green")
            ax.scatter3D(optV[0], optV[1], optV[2], c='blue', marker="^", label="optV")
            ax.scatter3D(estimateV[0], estimateV[1], estimateV[2], c='orange', marker="^", label="estV")

            ax.plot3D(
                [estimateV[0], (estimateV + estimateU)[0]],
                [estimateV[1], (estimateV + estimateU)[1]],
                [estimateV[2], (estimateV + estimateU)[2]],
                c="orange", label="estU"
            )

            ax.plot3D(
                [optV[0], (optV + optU)[0]],
                [optV[1], (optV + optU)[1]],
                [optV[2], (optV + optU)[2]],
                c="blue", label="optU"
            )

            plot_cone(ax, v=optV, u=optU, theta=optTheta, h0=0.5, h1=2)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.set_aspect('equal')
            plt.legend()
            if show:
                plt.show()
        return optV, optU, optTheta

    @staticmethod
    def project_point_on_cone(
            x: float,
            y: float,
            z: float,
            t: np.array,
            v: np.array,
            u: np.array,
            theta: float,
            debug=False,
    ):
        # move cone vertex to origin
        cX = np.array([x - v[0], y - v[1], z - v[2]])
        cX2 = cX + t
        cv = np.array([0, 0, 0])

        ut = t / np.linalg.norm(t)
        uu = u / np.linalg.norm(u)

        # rotate coordinates s.t.cone axis is (0,0,1)
        ru, rotmatrix = rotate_axis_to_z(uu)
        rt = np.matmul(ut, rotmatrix)
        rX = np.matmul(cX, rotmatrix)
        rX2 = np.matmul(cX2, rotmatrix)
        # print("||ru|| = ", np.linalg.norm(ru))
        # print("||rt|| = ", np.linalg.norm(rt))

        # compute distance between skewed lines: rX + s * rt and origin + s * ru
        sol = skew_lines_dist(unit_a=rt, unit_b=ru, point_a=rX, point_b=cv)
        if sol is None:
            return
        dist = sol[2]
        point3 = rX + sol[0] * rt  # point on the projection line, closest to the axis
        point4 = cv + sol[1] * ru  # point on the axis, closest to the projection line
        if dist > np.abs(point4[2]) * np.tan(theta):
            print("projection too far")
            point3_back = np.matmul(point3, np.linalg.inv(rotmatrix)) + v
            return point3_back, point3_back

        tan2 = np.tan(theta)**2
        s1, s2 = solve_quadratic_equation(
            a=rt[0]**2 + rt[1]**2 - tan2*rt[2]**2,
            b=2 * (rX[0]*rt[0] + rX[1]*rt[1] - tan2 * rX[2]*rt[2]),
            c=rX[0]**2 + rX[1]**2 - tan2 * rX[2]**2,
        )

        # two points intersecting the cone
        point1 = rX + rt * s1
        point2 = rX + rt * s2

        # back to original coordinate system
        point1_back = np.matmul(point1, np.linalg.inv(rotmatrix)) + v
        point2_back = np.matmul(point2, np.linalg.inv(rotmatrix)) + v

        if np.isnan(point1_back).any():
            print("NaN")

        if debug:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            plot_cone(ax, v=cv, u=ru, theta=theta, h0=0.5, h1=2)
            ax.scatter3D(rX[0], rX[1], rX[2], c='k', marker='*')
            ax.scatter3D(point1[0], point1[1], point1[2], c='blue')
            ax.scatter3D(point2[0], point2[1], point2[2], c='blue')
            # ax.scatter3D(rX2[0], rX2[1], rX2[2], c='k', marker='^')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.plot3D(
                [rX[0], (rX + 4*rt)[0]],
                [rX[1], (rX + 4*rt)[1]],
                [rX[2], (rX + 4*rt)[2]],
                c="orange", label="proj"
            )

            ax.set_aspect('equal')
            plt.legend()
            plt.show()

        return point1_back, point2_back


def experiment_synthetic():
    trueV = np.array((1, 2, 3))
    trueU = np.array((3, 0, 3))
    trueU = trueU / np.linalg.norm(trueU)
    trueTheta = np.pi / 3
    trueW = trueU / np.cos(trueTheta)
    n_points = 500
    data, gtnormals = generate_cone_points(
        v=trueV,
        u=trueU,
        theta=trueTheta,
        h0=1,
        h1=2,
        n=n_points,
    )
    #data += (np.random.rand(data.shape[0], data.shape[1]) - 0.5) * 0.05
    normals = gtnormals #+ (np.random.rand(data.shape[0], data.shape[1]) - 0.5) * 0.1
    normals = normals / np.linalg.norm(normals, axis=0)

    print("points: ", data.shape)

    conefit = ConeFit(points=data, normals=normals, method="lm")
    oV, oU, oTheta = conefit.pyomo_fit(estimateU=trueU)
    # oV, oU, oTheta = conefit.fit(plot=True, show=True, method="trust-constr")
    # oV, oU, oTheta = conefit.fit(plot=True, show=True, method="lbfgsb")
    oW = oU / np.cos(oTheta)
    # oTheta = np.arccos(1 / np.linalg.norm(oW))

    print("\n\n=== SOLVE DONE ===")
    print("oV :", oV)
    print("trueV :", trueV)
    print("||oU|| :", np.linalg.norm(oU))
    print("oU :", oU)
    print("trueU :", trueU)
    print("oW :", oW)
    print("trueW :", trueW)
    print("oTheta :", oTheta)
    print("trueTheta: ", trueTheta)

    print("Objective at gt: ", cone_alignment_energy2(v=trueV, u=trueU, theta=trueTheta, points=data))
    print("Objective at opt: ", cone_alignment_energy2_vw(v=oV, w=oW, points=data))

    centered_points = data - oV.reshape(3,1)
    newu, rotmatrix = rotate_axis_to_z(oU)
    newpoints = np.matmul(centered_points.transpose(), rotmatrix).transpose()

    mypoint = np.array([3, 4, 4.5])
    myproj = np.array([-1, -2, -1])

    p1, p2 = conefit.project_point_on_cone(
        x=mypoint[0],
        y=mypoint[1],
        z=mypoint[2],
        t=myproj,
        u=trueU,
        v=trueV,
        theta=trueTheta,
        debug=True,
    )

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title("Cone fit with point projection")
    ax.scatter3D(data[0, :], data[1, :], data[2, :], c="green")
    ax.scatter3D(trueV[0], trueV[1], trueV[2], c='blue', marker="^", label="trueV")
    ax.plot3D(
        [trueV[0], (trueV + trueU)[0]],
        [trueV[1], (trueV + trueU)[1]],
        [trueV[2], (trueV + trueU)[2]],
        c="green", label="trueU"
    )
    ax.scatter3D(mypoint[0], mypoint[1], mypoint[2], c='black', marker="*", label="p")
    ax.scatter3D(p1[0], p1[1], p1[2], c='red', marker="*", label="p1")
    ax.scatter3D(p2[0], p2[1], p2[2], c='red', marker="*", label="p2")
    ax.plot3D(
        [mypoint[0], (mypoint + 2*myproj)[0]],
        [mypoint[1], (mypoint + 2*myproj)[1]],
        [mypoint[2], (mypoint + 2*myproj)[2]],
        c="orange", label="proj"
    )

    plot_cone(ax, v=oV, u=oU, theta=oTheta, h0=0.5, h1=2.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_aspect('equal')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    experiment_synthetic()
