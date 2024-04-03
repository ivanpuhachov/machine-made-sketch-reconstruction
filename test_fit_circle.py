import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import skg


def generate_circle_points(
        center: np.array,
        n=10,
        r=3,
):
    random_phi = 0.5 * np.pi * np.random.rand(1, n)
    result = np.vstack([
        np.cos(random_phi),
        np.sin(random_phi)
    ]) * r + center.reshape(2,1)
    return result


def get_xy_circle(
        center: np.array,
        r=3,
):
    phi = np.linspace(0, 2*np.pi, num=36)
    x = r * np.cos(phi) + center[0]
    x = np.append(x, x[0])
    y = r * np.sin(phi) + center[1]
    y = np.append(y, y[0])
    return x, y


if __name__ == "__main__":
    trueC = np.array([2, 3])
    trueR = 2
    n_points = 10

    data = generate_circle_points(
        center=trueC,
        r=trueR,
        n=n_points,
    )

    data += (np.random.rand(2, n_points) - 0.5)*0.15*trueR

    print("data.shape ", data.shape)
    r, c = skg.nsphere_fit(data, axis=0, scaling=True)
    print(r, c)

    plt.scatter(data[0,:], data[1,:], c="blue")
    plt.scatter(trueC[0], trueC[1], c="green", marker="^", label="gt")
    plt.scatter(c[0], c[1], c="red", marker="^", label="estim")
    x, y = get_xy_circle(center=trueC, r=trueR)
    plt.plot(x, y, color="olive")
    x, y = get_xy_circle(center=c, r=r)
    plt.plot(x, y, color="gold")
    plt.axis("equal")
    plt.show()


