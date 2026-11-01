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

h = 0.001

ct = 2000.0     # общее время интегрирования
NT = 0.4        # окно: через каждые NT траектории возвращаем на расстояние eps
eps = 1e-8
transient = 0.7 # доля окон, отбрасываемая как переходный процесс

IC_RANGE = (0.0, 1.0)   # диапазон случайных начальных условий
SEED = 12345            # фиксируем, чтобы у всех методов и RK4 были одинаковые старты

start_params = (1.0, 1.0, 1.0)

a1_values = np.linspace(0, 5, 100)
a2_values = np.linspace(0, 5, 100)
a3_values = np.linspace(0, 5, 100)

save_folder = "t1_lle_plots"
os.makedirs(save_folder, exist_ok=True)


# ============================================================
# Система: Sprott Case B
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

parameters = [
    ("a1", a1_values),
    ("a2", a2_values),
    ("a3", a3_values),
]


# ============================================================
# Расчёт LLE для всех значений параметра (параллельно по ядрам)
# ============================================================

@njit(parallel=True)
def lle_scan(method, f, values, which, u_init, pert, h, NT, n_windows,
             n_skip, eps, a1_0, a2_0, a3_0):

    n = values.shape[0]
    lle = np.full(n, np.nan)

    for k in prange(n):
        a1, a2, a3 = a1_0, a2_0, a3_0

        if which == 0:
            a1 = values[k]
        elif which == 1:
            a2 = values[k]
        else:
            a3 = values[k]

        x1 = u_init[k].copy()
        x2 = x1 + eps * pert[k]

        total = 0.0
        used = 0
        ok = True

        for w in range(n_windows):
            t1_, s1 = method(f, x1.copy(), 0.0, NT, h, a1, a2, a3)
            t2_, s2 = method(f, x2.copy(), 0.0, NT, h, a1, a2, a3)

            x1 = s1[s1.shape[0] - 1].copy()
            x2 = s2[s2.shape[0] - 1].copy()

            diff = x2 - x1
            d = np.sqrt(np.sum(diff * diff))

            # решение разошлось -> точки нет
            if (not np.isfinite(d) or d == 0.0
                    or np.max(np.abs(x1)) > 1e6):
                ok = False
                break

            if w >= n_skip:
                total += np.log(d / eps)
                used += 1

            # возвращаем вторую траекторию на расстояние eps по тому же направлению
            x2 = x1 + diff / d * eps

        if ok and used > 0:
            lle[k] = total / (used * NT)

    return lle


def run_scan(method, which, values, u_init, pert):
    n_windows = int(round(ct / NT))
    n_skip = int(round(transient * n_windows))
    return lle_scan(
        method, f, values, which, u_init, pert, h, NT, n_windows,
        n_skip, eps, start_params[0], start_params[1], start_params[2]
    )


# ============================================================
# График
# ============================================================

def plot_lle(method_name, parameter_name, params, lle_m, lle_rk4):
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color="black", linewidth=0.8)

    plt.plot(params, lle_m, color="blue", linewidth=1.0, label=method_name)
    plt.plot(params, lle_rk4, color="red", linewidth=1.0, alpha=0.6,
             label="RK4")

    plt.xlabel(parameter_name)
    plt.ylabel("LLE")
    plt.title(f"Sprott Case B: LLE от {parameter_name}, {method_name} и RK4")
    plt.legend(loc="best")
    plt.xlim(min(params), max(params))

    filepath = os.path.join(
        save_folder, f"LLE_Case_B_{parameter_name}_{method_name}.png"
    )
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"График сохранен: {filepath}")


if __name__ == "__main__":

    rng = np.random.default_rng(SEED)

    for which, (parameter_name, params) in enumerate(parameters):

        # Одни и те же случайные начальные условия для всех методов и RK4
        n = len(params)
        u_init = rng.uniform(IC_RANGE[0], IC_RANGE[1], size=(n, 3))
        pert = rng.random(size=(n, 3))   # x2 = x1 + eps * rand(0; 1)

        print(f"RK4, параметр {parameter_name}...")
        lle_rk4 = run_scan(rk4_jit, which, params, u_init, pert)

        for method_name, method in methods:
            print(f"{method_name}, параметр {parameter_name}...")
            lle_m = run_scan(method, which, params, u_init, pert)
            plot_lle(method_name, parameter_name, params, lle_m, lle_rk4)

    print("\nПостроение всех графиков LLE завершено.")