import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# Фиксированные коэффициенты
a1, a3 = 1.0, 1.0

# По горизонтали: a2, по вертикали: шаг h (оба линейно)
A2_MIN, A2_MAX = 0.0, 3.0
H_MIN, H_MAX = 1e-4, 0.1

ct = 2000.0               # общее время интегрирования
NT = 0.4                  # окно между возвратами на расстояние eps
eps = 1e-8
transient = 0.7           # доля окон, отбрасываемая как переходный процесс

IC_RANGE = (0.0, 1.0)     # диапазон случайных начальных условий
SEED = 12345              # одинаковые старты для всех клеток, методов и RK4

VMIN, VMAX = -0.3, 0.3    # пределы цветовой шкалы LLE (None = автоматически)

CASE_NAME = "Case_B"
save_folder = "t4_lle_maps"
os.makedirs(save_folder, exist_ok=True)

a2_values = np.linspace(A2_MIN, A2_MAX, RES)
h_values = np.linspace(H_MIN, H_MAX, RES)


# ============================================================
# Система: Sprott Case B  (z' = a3 - x*y)
# ВАЖНО: в EulerKromer.py и KD.py в уравнении для z тоже должно стоять
# x_new * y_new (а не x_new * x_new).
# ============================================================

@njit(cache=True)
def f(u, a1, a2, a3):
    x = u[0]
    y = u[1]
    z = u[2]

    return np.array([
        a1 * y * z,
        x - a2 * y,
        a3 - x * y
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
def scan_map(method, f, a2s, hs, a1, a3, u_init, pert, NT, n_windows,
             n_skip, eps):

    n1 = a2s.shape[0]   # по горизонтали: a2
    n2 = hs.shape[0]    # по вертикали: h
    out = np.full((n2, n1), np.nan)

    for k in prange(n1 * n2):
        # h меняется быстрее всего, чтобы дорогие клетки (малые h)
        # равномерно распределялись по потокам
        i2 = k % n2
        i1 = k // n2

        out[i2, i1] = lle_one(method, f, u_init, pert, hs[i2], NT,
                              n_windows, n_skip, eps,
                              a1, a2s[i1], a3)

    return out


def run_scan(method, u_init, pert):
    n_windows = int(round(ct / NT))
    n_skip = int(round(transient * n_windows))
    return scan_map(method, f, a2_values, h_values, a1, a3,
                    u_init, pert, NT, n_windows, n_skip, eps)


# ============================================================
# График
# ============================================================

def plot_map(method_name, data):
    plt.figure(figsize=(9, 6))

    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("lightgray")          # расходящиеся решения -> серым
    masked = np.ma.masked_invalid(data)

    plt.imshow(masked, origin="lower", aspect="auto", cmap=cmap,
               extent=(A2_MIN, A2_MAX, H_MIN, H_MAX),
               interpolation="nearest", vmin=VMIN, vmax=VMAX)
    plt.colorbar(label="LLE",
                 extend="both" if VMIN is not None else "neither")

    plt.legend(handles=[mpatches.Patch(facecolor="lightgray",
                                       edgecolor="black",
                                       label="решение расходится")],
               loc="upper right")

    plt.xlabel("a2")
    plt.ylabel("h")
    plt.title(f"Sprott {CASE_NAME}: {method_name}, LLE "
              f"(a1 = {a1:g}, a3 = {a3:g}, ct = {ct:g}, {RES}x{RES})")

    filepath = os.path.join(
        save_folder,
        f"LLE_map_{CASE_NAME}_a2_h_{method_name}_{RES}x{RES}.png"
    )
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {filepath}")


if __name__ == "__main__":

    rng = np.random.default_rng(SEED)
    u_init = rng.uniform(IC_RANGE[0], IC_RANGE[1], size=3)
    pert = rng.random(size=3)   # x2 = x1 + eps * rand(0; 1)

    for method_name, method in methods:
        print(f"{method_name}: a2 x h...")
        data = run_scan(method, u_init, pert)

        n_div = int(np.isnan(data).sum())
        print(f"  клеток без значения (расходимость): {n_div} из {RES * RES}")

        plot_map(method_name, data)

    print("\nПостроение всех карт LLE (a2, h) завершено.")