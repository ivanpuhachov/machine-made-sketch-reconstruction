import numpy as np
import scipy.optimize as spo


def patch_params_to_axis(
        type_of_patch: str,
        params: np.array,
):
    """
    extracts axis from patch corresponding to its type
    :param type_of_patch:
    :param params:
    :return:
    """
    if type_of_patch not in ["Plane", "Cylinder", "Cone"]:
        raise Exception(f"Axis not supported for patch of type {type_of_patch}")
    if len(params) == 0:
        raise Exception(f"This patch of type {type_of_patch} appears to have no params! {params}")
    if type_of_patch == "Plane":
        axis = np.array(params[:3])
        # axis = axis / np.linalg.norm(axis)
        return axis
    if (type_of_patch == "Cone") or (type_of_patch == "Cylinder"):
        axis = np.array(params[3:6])
        # axis = axis / np.linalg.norm(axis)
        return axis


def pad_axis_gradient(
        type_of_patch: str,
        axis_grad: np.array,
):
    """
    given (1x3) gradient of axis pad it to (0,grad,0) array correspondingly to type
    :param type_of_patch:
    :param axis_grad:
    :return:
    """
    if type_of_patch not in ["Plane", "Cylinder", "Cone"]:
        raise Exception(f"Grad padding is not supported for patch of type {type_of_patch}")
    if axis_grad.shape != (3,):
        raise Exception(f"This grad wrong shape: {axis_grad.shape} --> {axis_grad}")
    if type_of_patch == "Plane":
        res = np.pad(axis_grad, [(0, 1)], mode='constant', constant_values=0)
        return res
    if (type_of_patch == "Cone") or (type_of_patch == "Cylinder"):
        res = np.pad(axis_grad, [(3, 1)], mode='constant', constant_values=0)
        return res


def axis_alignment_energy(
        a_type: str,
        a_params: np.array,
        b_type: str,
        b_params: np.array,
):
    w1 = patch_params_to_axis(type_of_patch=a_type, params=a_params)
    w2 = patch_params_to_axis(type_of_patch=b_type, params=b_params)
    energy = (1 - w1.dot(w2))**2
    return energy


def grad_axis_alignment(
        a_type: str,
        a_params: np.array,
        b_type: str,
        b_params: np.array,
):
    w1 = patch_params_to_axis(type_of_patch=a_type, params=a_params)
    w2 = patch_params_to_axis(type_of_patch=b_type, params=b_params)
    dEdw1 = 2 * (w1.dot(w2) - 1) * w2
    dEdw2 = 2 * (w1.dot(w2) - 1) * w1
    dw1_padded = pad_axis_gradient(type_of_patch=a_type, axis_grad=dEdw1)
    dw2_padded = pad_axis_gradient(type_of_patch=b_type, axis_grad=dEdw2)
    return dw1_padded, dw2_padded


def axis_othogonality_energy(
        a_type: str,
        a_params: np.array,
        b_type: str,
        b_params: np.array,
):
    w1 = patch_params_to_axis(type_of_patch=a_type, params=a_params)
    w2 = patch_params_to_axis(type_of_patch=b_type, params=b_params)
    energy = (w1.dot(w2)) ** 2
    return energy


def grad_axis_orthogonality(
        a_type: str,
        a_params: np.array,
        b_type: str,
        b_params: np.array,
):
    w1 = patch_params_to_axis(type_of_patch=a_type, params=a_params)
    w2 = patch_params_to_axis(type_of_patch=b_type, params=b_params)
    dEdw1 = 2 * (w1.dot(w2)) * w2
    dEdw2 = 2 * (w1.dot(w2)) * w1
    dw1_padded = pad_axis_gradient(type_of_patch=a_type, axis_grad=dEdw1)
    dw2_padded = pad_axis_gradient(type_of_patch=b_type, axis_grad=dEdw2)
    return dw1_padded, dw2_padded


if __name__ == "__main__":
    typeA = "Plane"
    paramsA = np.array([0.52590663, -0.07790949, -0.84696654, -2.90030857])
    typeB = "Plane"
    paramsB = np.array([-0.7732949, 0.00832372, -0.63399189, -2.06902011])

    en = axis_othogonality_energy(
        a_type=typeA,
        a_params=paramsA,
        b_type=typeB,
        b_params=paramsB,
    )

    gs = grad_axis_orthogonality(
        a_type=typeA,
        a_params=paramsA,
        b_type=typeB,
        b_params=paramsB,
    )

    num_grad1 = spo.approx_fprime(
        xk=paramsA,
        f=lambda x: axis_othogonality_energy(
            a_type=typeA,
            a_params=x,
            b_type=typeB,
            b_params=paramsB,
        ),
        epsilon=1.5e-8,
    )

    num_grad2 = spo.approx_fprime(
        xk=paramsB,
        f=lambda x: axis_othogonality_energy(
            a_type=typeA,
            a_params=paramsA,
            b_type=typeB,
            b_params=x,
        ),
    )

    print("num_grad 1: ", num_grad1)
    print("my  grad 1: ", gs[0])

    print("num_grad 2: ", num_grad2)
    print("my  grad 2: ", gs[1])

    err1 = spo.check_grad(
        x0=paramsA,
        func=lambda x: axis_othogonality_energy(
            a_type=typeA,
            a_params=x,
            b_type=typeB,
            b_params=paramsB,
        ),
        grad=lambda x: grad_axis_orthogonality(
            a_type=typeA,
            a_params=x,
            b_type=typeB,
            b_params=paramsB,
        )[0]
    )

    gg = lambda x: grad_axis_orthogonality(
        a_type=typeA,
        a_params=x,
        b_type=typeB,
        b_params=paramsB,
    )[0]
    print("gg: ", gg(paramsA))

    err2 = spo.check_grad(
        x0=paramsB,
        func=lambda x: axis_othogonality_energy(
            a_type=typeA,
            a_params=paramsA,
            b_type=typeB,
            b_params=x,
        ),
        grad=lambda x: grad_axis_orthogonality(
            a_type=typeA,
            a_params=paramsA,
            b_type=typeB,
            b_params=x,
        )[1]
    )

    print("error 1: ", err1)
    print("error 2: ", err2)

    print("energy: ", en)
