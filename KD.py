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

        x_new = x + h1 * a1 * y * z
        y_new = y + h1 * (x_new - a2 * y)
        z_new = z + h1 * (a3 - x_new * y_new)

        u_half = np.array([x_new, y_new, z_new])
        u_new = u_half.copy()

        for _ in range(4):
            k2 = f(u_new, a1, a2, a3)
            u_new = u_half + h2 * k2

        # y_n1 = (y_new + h2) / (1 + h2 * x_new)
        # x_n1 = x_new + h2 * (y_n1 * z_new)
        # z_n1 = z_new + h2 * (1 - x_n1 * y_n1)
        # u_new = np.array([x_n1, y_n1, z_n1])

        u[i + 1] = u_new

    return t, u