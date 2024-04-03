import pyomo.environ as pyo


def distance_between_patches(
        params1,
        type1,
        params2,
        type2,
        x, y, z,
):
    """
    Evaluates distance between patches at (x, y, z)
    :param params1:
    :param type1:
    :param params2:
    :param type2:
    :param x:
    :param y:
    :param z:
    :return:
    """
    type_tuple = (type1, type2)
    if type_tuple == ("Plane", "Plane"):
        return plane_plane_distance(params1, type1, params2, type2, x, y, z)
    if (type_tuple == ("Plane", "Cylinder")) or (type_tuple == ("Cylinder", "Plane")):
        return plane_cylinder_distance(params1, type1, params2, type2, x, y, z)
    if (type_tuple == ("Cylinder", "Cylinder")):
        return cylinder_cylinder_distance(params1, type1, params2, type2, x, y, z)
    if "Other" in type_tuple:
        if type1 in ["Plane", "Cylinder"]:
            return evaluate_distance(x, y, z, params=params1, type=type1), 0
        if type2 in ["Plane", "Cylinder"]:
            return evaluate_distance(x, y, z, params=params2, type=type2), 0
        return 0, 0
    print(f"Distance {type1} - {type2} NOT IMPLEMENTED")
    raise NotImplemented


def evaluate_distance(
        x,
        y,
        z,
        params: list,
        type: str,
):
    """
    Evaluates distance between fixed point (x, y, z) and a patch
    :param x:
    :param y:
    :param z:
    :param params:
    :param type:
    :return:
    """
    if type == "Plane":
        return plane_distance(x, y, z, a=params[0], b=params[1], c=params[2], d=params[3], w_norm=0)
    if type == "Cylinder":
        return cylinder_distance(x, y, z,
                                 cx=params[0], cy=params[1], cz=params[2],
                                 w1=params[3], w2=params[4], w3=params[5],
                                 r2=params[6])
    if type == "Sphere":
        return sphere_distance(x, y, z,
                               cx=params[0], cy=params[1], cz=params[2],
                               r2=params[3],
                               )
    if type == "Other":
        return 0
    raise NotImplemented


def get_patch_axis(
        mp,
        p_patch,
        p_type,
):
    if p_type == "Plane":
        return mp[p_patch, 0], mp[p_patch, 1], mp[p_patch, 2]
    if p_type == "Cylinder":
        return mp[p_patch, 3], mp[p_patch, 4], mp[p_patch, 5]
    raise NotImplemented


def plane_distance(
        x, y, z, a, b, c, d, w_norm=2,
):
    znew = (a * x + b * y + d) / (-c)
    dist2 = (z - znew) ** 2
    # dist2 = (a * x + b * y + c * z + d) ** 2 / (a ** 2 + b ** 2 + c ** 2)
    # if len(dist2) > 1:
    #     dist2 = sum(dist2)
    # n = np.array([a, b, c])
    # norm_penalty = (1 - n.dot(n)) ** 2
    norm = a ** 2 + b ** 2 + c ** 2
    norm_penalty = (1 - norm) ** 2
    value = dist2 + w_norm * norm_penalty
    return value


def plane_plane_distance(
        params1,
        type1,
        params2,
        type2,
        x, y, z,
):
    assert type1 == "Plane"
    assert type2 == "Plane"
    a1, b1, c1, d1 = params1[0], params1[1], params1[2], params1[3]
    a2, b2, c2, d2 = params2[0], params2[1], params2[2], params2[3]
    z1 = (a1 * x + b1 * y + d1) / (-c1)
    z2 = (a2 * x + b2 * y + d2) / (-c2)
    dist2 = (z1 - z2) ** 2
    return dist2, (z1 + z2) / 2


def cylinder_distance(
        x, y, z,
        cx, cy, cz,
        w1, w2, w3,
        r2,
):
    # vector from point to cylinder axis: H = X - [(X-C) . W] * W
    xcdotw = (x - cx) * w1 + (y - cy) * w2 + (z - cz) * w3
    h1 = x - cx - xcdotw * w1
    h2 = y - cy - xcdotw * w2
    h3 = z - cz - xcdotw * w3
    normH = pyo.sqrt(h1 * h1 + h2 * h2 + h3 * h3)
    distance2 = (normH - pyo.sqrt(r2)) ** 2
    return distance2


def cone_distance(
        x, y, z,
        vx, vy, vz,
        u1, u2, u3,
        theta,
):
    # TODO: this is not actually distance, but a good proxy
    udotxv = u1 * (x - vx) + u2 * (y - vy) + u3 * (z - vz)
    normxv = pyo.sqrt(
        (x - vx) * (x - vx) + (y - vy) * (y - vy) + (z - vz) * (z - vz)
    )
    return (udotxv - normxv * pyo.cos(theta)) ** 2


def sphere_distance(
        x, y, z,
        cx, cy, cz,
        r2,
):
    xcnorm2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz)
    return (pyo.sqrt(xcnorm2) - pyo.sqrt(r2)) ** 2


def plane_cylinder_distance(
        params1,
        type1,
        params2,
        type2,
        x, y, z,
):
    if (type1 == "Cylinder") and (type2 == "Plane"):
        return plane_cylinder_distance(
            params2, type2, params1, type1, x, y, z,
        )
    assert type1 == "Plane"
    assert type2 == "Cylinder"
    a1, b1, c1, d1 = params1[0], params1[1], params1[2], params1[3]
    z1 = (a1 * x + b1 * y + d1) / (-c1)
    cx, cy, cz = params2[0], params2[1], params2[2]
    w1, w2, w3 = params2[3], params2[4], params2[5]
    r2 = params2[6]
    return cylinder_distance(
        x=x, y=y, z=z1,
        cx=cx, cy=cy, cz=cz,
        w1=w1, w2=w2, w3=w3,
        r2=r2,
    ), z1


def plane_any_distance(
        params1,
        type1,
        params2,
        type2,
        x, y, z,
):
    if type2 == "Plane":
        return plane_any_distance(
            params2, type2, params1, type1, x, y, z,
        )
    assert type1 == "Plane"
    assert type2 != "Plane"
    a1, b1, c1, d1 = params1[0], params1[1], params1[2], params1[3]
    z1 = (a1 * x + b1 * y + d1) / (-c1)
    return evaluate_distance(x, y, z1, params=params2, type=type2)


def cylinder_cylinder_distance(
        params1,
        type1,
        params2,
        type2,
        x, y, z,
):
    assert type1 == "Cylinder"
    assert type2 == "Cylinder"
    c1x, c1y, c1z = params1[0], params1[1], params1[2]
    w11, w12, w13 = params1[3], params1[4], params1[5]
    r1_2 = params1[6]
    dist2_1 = cylinder_distance(
        x=x, y=y, z=z,
        cx=c1x, cy=c1y, cz=c1z,
        w1=w11, w2=w12, w3=w13,
        r2=r1_2,
    )
    c2x, c2y, c2z = params2[0], params2[1], params2[2]
    w21, w22, w23 = params2[3], params2[4], params2[5]
    r2_2 = params2[6]
    dist2_2 = cylinder_distance(
        x=x, y=y, z=z,
        cx=c2x, cy=c2y, cz=c2z,
        w1=w21, w2=w22, w3=w23,
        r2=r2_2,
    )
    # TODO: cyl-cyl final distance (d1^2 + d2^2) - does it look right?
    return dist2_1 + dist2_2, 0


if __name__ == "__main__":
    cd = cylinder_distance(
        x=5, y=0, z=0,
        cx=5, cy=0, cz=0,
        w1=0, w2=0, w3=1,
        r2=4,
    )
    print(cd)
