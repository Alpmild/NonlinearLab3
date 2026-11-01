import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

from Euler import euler
from EulerKromer import euler_kromer
from EMP import emp
from KD import kd
from RK4 import rk4

# ============================================================
# Настройки
# ============================================================

RES = 100                 # разрешение карты (RES x RES клеток)

h = 0.01

ct = 2000.0               # общее время интегрирования
NT = 0.4                  # окно между возвратами на расстояние eps
eps = 1e-8
transient = 0.7           # доля окон, отбрасываемая как переходный процесс

IC_RANGE = (0.0, 1.0)     # диапазон случайных начальных условий
SEED = 12345              # одинаковые старты для всех клеток, методов и RK4

# Значения параметров, которые не перебираются (по умолчанию 1)
BASE = {"a1": 1.0, "a2": 1.0, "a3": 1.0}

# Диапазоны перебора каждого параметра
RANGES = {
    "a1": (0.0, 3.0),
    "a2": (0.0, 3.0),
    "a3": (0.0, 3.0),
}

# Какие пары параметров строить: (по горизонтали, по вертикали).
# Для остальных параметров берутся значения из BASE.
PAIRS = [
    ("a1", "a2"),
    # ("a1", "a3"),
    # ("a2", "a3"),
]

VMIN, VMAX = -0.3, 0.3    # пределы цветовой шкалы LLE (None = автоматически)

CASE_NAME = "Case_B"
save_folder = "t3_lle_maps"
os.makedirs(save_folder, exist_ok=True)

PARAM_INDEX = {"a1": 0, "a2": 1, "a3": 2}


# ============================================================
# Система: Sprott Case C  (z' = a3 - x^2)
# ============================================================

@njit(cache=True)
def f(u, a1, a2, a3):
    x = u[0]
    y = u[1]
    z = u[2]

    return np.array([
        a1 * y * z,
        x - a2 * y,
        a3 - x * x
    ])


methods = [
    ("Euler", njit(euler)),
    ("Euler-Cromer", njit(euler_kromer)),
    ("EMP", njit(emp)),
    ("VSKD", njit(kd)),
    ("RK4", njit(rk4)),
]


# ============================================================
# LLE для одной клетки карты
# ============================================================

@njit(cache=True)
def lle_one(method, f, u_init, pert, h, NT, n_windows, n_skip, eps,
            a1, a2, a3):

    n_steps = int(round(NT / h))
    if n_steps < 1:
        n_steps = 1
    T_w = n_steps * h
    # +0.25 шага, чтобы и round, и int (в KD) дали ровно n_steps
    t_end = (n_steps + 0.25) * h

    x1 = u_init.copy()
    x2 = x1 + eps * pert

    total = 0.0
    used = 0

    for w in range(n_windows):
        t1_, s1 = method(f, x1.copy(), 0.0, t_end, h, a1, a2, a3)
        t2_, s2 = method(f, x2.copy(), 0.0, t_end, h, a1, a2, a3)

        x1 = s1[s1.shape[0] - 1].copy()
        x2 = s2[s2.shape[0] - 1].copy()

        diff = x2 - x1
        d = np.sqrt(np.sum(diff * diff))

        # решение разошлось -> значения нет
        if (not np.isfinite(d) or d == 0.0
                or np.max(np.abs(x1)) > 1e6):
            return np.nan

        if w >= n_skip:
            total += np.log(d / eps)
            used += 1

        x2 = x1 + diff / d * eps

    if used == 0:
        return np.nan
    return total / (used * T_w)


# ============================================================
# Карта LLE (параллельно по клеткам)
# ============================================================

@njit(parallel=True)
def scan_map(method, f, xs, ys, ix, iy, base1, base2, base3,
             u_init, pert, h, NT, n_windows, n_skip, eps):

    n1 = xs.shape[0]
    n2 = ys.shape[0]
    out = np.full((n2, n1), np.nan)

    for k in prange(n1 * n2):
        i2 = k // n1
        i1 = k % n1

        p = np.array([base1, base2, base3])
        p[ix] = xs[i1]
        p[iy] = ys[i2]

        out[i2, i1] = lle_one(method, f, u_init, pert, h, NT, n_windows,
                              n_skip, eps, p[0], p[1], p[2])

    return out


def run_scan(method, xname, yname, xs, ys, u_init, pert):
    n_windows = int(round(ct / NT))
    n_skip = int(round(transient * n_windows))
    return scan_map(
        method, f, xs, ys, PARAM_INDEX[xname], PARAM_INDEX[yname],
        BASE["a1"], BASE["a2"], BASE["a3"],
        u_init, pert, h, NT, n_windows, n_skip, eps
    )


# ============================================================
# График
# ============================================================

def plot_map(method_name, xname, yname, data):
    xmin, xmax = RANGES[xname]
    ymin, ymax = RANGES[yname]

    plt.figure(figsize=(9, 6))

    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("white")              # расходящиеся решения -> белым
    masked = np.ma.masked_invalid(data)

    plt.imshow(masked, origin="lower", aspect="auto", cmap=cmap,
               extent=(xmin, xmax, ymin, ymax), interpolation="nearest",
               vmin=VMIN, vmax=VMAX)
    plt.colorbar(label="LLE",
                 extend="both" if VMIN is not None else "neither")

    fixed = [p for p in ("a1", "a2", "a3") if p not in (xname, yname)]
    fixed_txt = ", ".join(f"{p} = {BASE[p]:g}" for p in fixed)

    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(f"Sprott {CASE_NAME}: {method_name}, LLE "
              f"({fixed_txt}, h = {h}, ct = {ct:g}, {RES}x{RES})")

    filepath = os.path.join(
        save_folder,
        f"LLE_map_{CASE_NAME}_{xname}_{yname}_{method_name}_{RES}x{RES}.png"
    )
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {filepath}")


if __name__ == "__main__":

    rng = np.random.default_rng(SEED)
    u_init = rng.uniform(IC_RANGE[0], IC_RANGE[1], size=3)
    pert = rng.random(size=3)   # x2 = x1 + eps * rand(0; 1)

    for xname, yname in PAIRS:
        xs = np.linspace(*RANGES[xname], RES)
        ys = np.linspace(*RANGES[yname], RES)

        for method_name, method in methods:
            print(f"{method_name}: {xname} x {yname}...")
            data = run_scan(method, xname, yname, xs, ys, u_init, pert)

            n_div = int(np.isnan(data).sum())
            print(f"  клеток без значения (расходимость): {n_div} из {RES * RES}")

            plot_map(method_name, xname, yname, data)

    print("\nПостроение всех карт LLE завершено.")