import numpy as np

NAME_ERROR = 'Ошибка KD'


def kd(f, u0, t0, t1, h, a1, a2, a3, s=0.5):

    u0 = u0.copy()

    h1 = h * s
    h2 = h * (1 - s)

    n = int((t1 - t0) / h)
    t = t0 + h * np.arange(n + 1)
    u = np.zeros((n + 1, len(u0)))

    u[0] = u0

    for i in range(n):
        x, y, z = u[i]

        x_m = x + h1 * a1 * y * z
        y_m = y + h1 * (x_m - a2 * y)
        z_m = z + h1 * (a3 - x_m * y_m)

        z_new = z_m + h2 * (a3 - x_m * y_m)
        y_new = (y_m + h2 * x_m) / (1 + a2 * h2)
        x_new = x_m + h2 * a1 * y_new * z_new

        u[i + 1] = np.array([x_new, y_new, z_new])

    return t, u