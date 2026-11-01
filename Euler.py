import numpy as np

NAME_ERROR = 'Ошибка Эйлера'


def euler(f, u0, t0, t1, h, a1, a2, a3):
    n = int(round((t1 - t0) / h))
    t = t0 + h * np.arange(n + 1)

    u = np.zeros((n + 1, 3))
    u[0] = u0

    for i in range(n):
        u[i + 1] = u[i] + h * f(u[i], a1, a2, a3)
    return t, u
