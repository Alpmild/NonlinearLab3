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

# Коэффициенты системы
a1, a2, a3 = 1.0, 1.0, 1.0

ct = 2000.0      # общее время интегрирования
NT = 0.4         # окно (время между возвратами на расстояние eps)
eps = 1e-8
transient = 0.7  # доля окон, отбрасываемая как переходный процесс

min_h = 1e-4
max_h = 0.1
res = 100       # число значений шага

# Линейная сетка шагов
h_values = np.linspace(min_h, max_h, res)

IC_RANGE = (0.0, 1.0)   # диапазон случайных начальных условий
SEED = 12345            # одинаковые старты для всех методов и RK4

save_folder = "t2_lle_h_plots"
os.makedirs(save_folder, exist_ok=True)


# ============================================================
# Система: Sprott Case C
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
]
rk4_jit = njit(rk4)


# ============================================================
# LLE для всех значений шага (параллельно по ядрам)
# ============================================================

@njit(parallel=True)
def lle_scan_h(method, f, h_values, u_init, pert, NT, n_windows, n_skip,
               eps, a1, a2, a3):

    n = h_values.shape[0]
    lle = np.full(n, np.nan)

    for k in prange(n):
        hk = h_values[k]

        # число шагов в окне; реальная длительность окна = n_steps * hk
        n_steps = int(round(NT / hk))
        if n_steps < 1:
            n_steps = 1
        T_w = n_steps * hk
        # +0.25 шага, чтобы и round, и int (в KD) дали ровно n_steps
        t_end = (n_steps + 0.25) * hk

        x1 = u_init.copy()
        x2 = x1 + eps * pert

        total = 0.0
        used = 0
        ok = True

        for w in range(n_windows):
            t1_, s1 = method(f, x1.copy(), 0.0, t_end, hk, a1, a2, a3)
            t2_, s2 = method(f, x2.copy(), 0.0, t_end, hk, a1, a2, a3)

            x1 = s1[s1.shape[0] - 1].copy()
            x2 = s2[s2.shape[0] - 1].copy()

            diff = x2 - x1
            d = np.sqrt(np.sum(diff * diff))

            # решение разошлось -> точки нет
            if not np.isfinite(d) or d == 0.0 or np.max(np.abs(x1)) > 1e6:
                ok = False
                break

            if w >= n_skip:
                total += np.log(d / eps)
                used += 1

            x2 = x1 + diff / d * eps

        if ok and used > 0:
            lle[k] = total / (used * T_w)

    return lle


def run_scan(method, u_init, pert):
    n_windows = int(round(ct / NT))
    n_skip = int(round(transient * n_windows))
    return lle_scan_h(method, f, h_values, u_init, pert, NT, n_windows,
                      n_skip, eps, a1, a2, a3)


# ============================================================
# Графики
# ============================================================

def plot_lle_h(method_name, lle_m, lle_rk4):
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color="black", linewidth=0.8)

    plt.plot(h_values, lle_m, color="blue", linewidth=1.0, label=method_name)
    plt.plot(h_values, lle_rk4, color="red", linewidth=1.0, alpha=0.6,
             label="RK4")

    plt.xlabel("h")
    plt.ylabel("LLE")
    plt.title(f"Sprott Case B: LLE от шага h, {method_name} и RK4")
    plt.legend(loc="best")
    plt.xlim(min_h, max_h)

    print(f"{method_name}: расхождение на {np.isnan(lle_m).sum()} из {len(lle_m)} точек")

    filepath = os.path.join(save_folder, f"LLE_h_{method_name}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {filepath}")


def plot_all(results, lle_rk4):
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color="black", linewidth=0.8)

    for name, lle_m in results.items():
        plt.plot(h_values, lle_m, linewidth=1.0, label=name)
    plt.plot(h_values, lle_rk4, color="black", linewidth=1.5, label="RK4")

    plt.xlabel("h")
    plt.ylabel("LLE")
    plt.title("Sprott Case B: LLE от шага h, все методы")
    plt.legend(loc="best")
    plt.xlim(min_h, max_h)

    filepath = os.path.join(save_folder, "LLE_h_all_methods.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {filepath}")


if __name__ == "__main__":

    rng = np.random.default_rng(SEED)
    u_init = rng.uniform(IC_RANGE[0], IC_RANGE[1], size=3)
    pert = rng.random(size=3)   # x2 = x1 + eps * rand(0; 1)

    print("RK4...")
    lle_rk4 = run_scan(rk4_jit, u_init, pert)

    results = {}
    for method_name, method in methods:
        print(f"{method_name}...")
        lle_m = run_scan(method, u_init, pert)
        results[method_name] = lle_m
        plot_lle_h(method_name, lle_m, lle_rk4)

    plot_all(results, lle_rk4)

    print("\nПостроение графиков LLE от шага завершено.")