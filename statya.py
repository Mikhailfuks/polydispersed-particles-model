"""
Расчёт Sobol-индексов чувствительности (детерминированная модель).
Просеивание отключено — иначе стохастический шум уничтожает оценку.

Запуск: python sobol_run.py
Требуется: numpy, scipy
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc

RHO_AIR = 1.205
NU_AIR  = 1.5e-5
G       = 9.81


# ================================================================
#  МОДЕЛЬ
# ================================================================

class Particle:
    CONTACT_EPS = 1e-7

    def __init__(self, R, psi, phi, alpha_deg, A, nu, gamma,
                 k0=0.75, f=0.3, fs=0.45, rho_p=2650.0,
                 beta=2.5, k_psi=4.0,
                 d_hole=0.008, p0=0.7,
                 kappa=0.15, classic=False,
                 enable_sieve=False):   # ← отключено по умолчанию
        self.classic = classic
        self.enable_sieve = enable_sieve
        self.R = R
        self.psi = 1.0 if classic else psi
        self.R_eff = R / np.sqrt(self.psi)
        self.m = (4/3) * np.pi * self.R_eff**3 * rho_p
        if classic:
            self.S_m = np.pi * self.R_eff**2
            self.cd_form = 1.0
            self.lam = 1.0
            self.k = k0
        else:
            self.S_m = np.pi * self.R_eff**2 * (1 + 0.5*(1 - self.psi))
            self.cd_form = 1 + k_psi * (1 - self.psi)
            self.lam = 1 + beta * phi
            self.k = 0.55 + 0.2 * self.psi
        self.alpha = np.radians(alpha_deg)
        self.A = A
        self.nu = nu
        self.gamma = gamma
        self.w = 2 * np.pi * nu
        self.f, self.fs = f, fs
        self.d_hole = d_hole
        self.p0 = p0
        self.kappa = kappa

    def xi(self, t):
        return self.A * (np.sin(self.w*t) + self.gamma*np.sin(2*self.w*t))

    def dxi(self, t):
        return self.A*self.w*(np.cos(self.w*t) + 2*self.gamma*np.cos(2*self.w*t))

    def ddxi(self, t):
        return -self.A*self.w**2*(np.sin(self.w*t) + 4*self.gamma*np.sin(2*self.w*t))

    def prob_pass(self):
        return 0.0   # отключено

    def Cd(self, Re):
        if self.classic:
            return 0.0
        if Re < 1e-8:
            return 0.0
        if Re < 1.0:
            return 24.0 / Re
        if Re < 400.0:
            return 24.0/Re + 4.0/Re**(1/3)
        if Re < 1000.0:
            w = (Re - 400.0) / 600.0
            Cd_400 = 24.0/400.0 + 4.0/400.0**(1/3)
            return Cd_400*(1-w) + 0.44*w
        return 0.44

    def drag(self, vx, vy):
        if self.classic:
            return 0.0, 0.0
        v = np.hypot(vx, vy)
        if v < 1e-12:
            return 0.0, 0.0
        Re = 2 * self.R_eff * v / NU_AIR
        Cd = self.Cd(Re) * self.cd_form
        Fd = 0.5 * Cd * RHO_AIR * self.S_m * v**2
        return Fd*vx/v, Fd*vy/v

    def rhs_flight(self, t, s):
        x, y, vx, vy = s
        Fgx = self.m * G * np.sin(self.alpha)
        Fgy = -self.m * G * np.cos(self.alpha)
        Fdx, Fdy = self.drag(vx, vy)
        return [vx, vy, (Fgx + Fdx)/self.m, (Fgy + Fdy)/self.m]

    def rhs_slide(self, t, s):
        x, y, vx, vy = s
        Fgx = self.m * G * np.sin(self.alpha)
        Fgy = -self.m * G * np.cos(self.alpha) - self.m * self.ddxi(t)
        Fdx, Fdy = self.drag(vx, vy)
        N = max(0.0, -(Fgy + Fdy))
        if abs(vx) > 1e-4:
            Ftr = self.f * N * np.sign(vx)
        else:
            Fsh = Fgx + Fdx
            if abs(Fsh) <= self.fs * N:
                Ftr = Fsh
            else:
                Ftr = self.fs * N * np.sign(Fsh)
        ax = (Fgx + Fdx - Ftr)/self.m
        return [vx, self.dxi(t), ax, self.ddxi(t)]

    def ev_land(self, t, s):
        return s[1] - (self.xi(t) + self.R_eff)
    ev_land.terminal = True
    ev_land.direction = -1

    def simulate(self, t_max=5.0, L=0.5, rtol=1e-6, atol=1e-8,
                 y_offset=1e-4, seed=None):
        """Детерминированный расчёт (без просеивания)."""
        T_period = 1.0 / max(self.nu, 1e-6)
        max_step = T_period / 100.0

        def ev_stop_L(t, s):
            return s[0] - L
        ev_stop_L.terminal = True
        ev_stop_L.direction = 1

        t = 0.0
        y0 = self.xi(0.0) + self.R_eff + y_offset
        s = np.array([0.0, y0, 0.0, 0.0])
        hits = 0
        t_flight = 0.0
        phase = "flight"

        for _ in range(2000):
            if t >= t_max or s[0] >= L - 1e-9:
                break

            if phase == "flight":
                sol = solve_ivp(
                    self.rhs_flight, [t, t_max], s,
                    method='RK45', rtol=rtol, atol=atol,
                    events=[self.ev_land, ev_stop_L],
                    max_step=max_step,
                )
                t_new = sol.t[-1]
                if t_new <= t + 1e-14:
                    break
                t_flight += t_new - t
                t = t_new
                s = sol.y[:, -1]

                if s[0] >= L - 1e-9:
                    break

                landed = (sol.status == 1 and len(sol.t_events[0]) > 0)
                if not landed:
                    break

                xi_t = self.xi(t)
                dxi_t = self.dxi(t)
                vy_rel = s[3] - dxi_t
                if vy_rel < 0:
                    s[3] = dxi_t - self.k * vy_rel
                    s[2] -= min(self.fs*abs(vy_rel), abs(s[2])) * np.sign(s[2])
                    hits += 1
                s[1] = xi_t + self.R_eff
                phase = "slide"

            else:
                Fgy = -self.m*G*np.cos(self.alpha) - self.m*self.ddxi(t)
                _, Fdy = self.drag(s[2], s[3])
                N_now = -(Fgy + Fdy)
                if N_now <= 0:
                    phase = "flight"
                    continue

                def ev_detach(t, s):
                    Fgy = -self.m*G*np.cos(self.alpha) - self.m*self.ddxi(t)
                    _, Fdy = self.drag(s[2], s[3])
                    return -(Fgy + Fdy)
                ev_detach.terminal = True
                ev_detach.direction = -1

                sol = solve_ivp(
                    self.rhs_slide, [t, t_max], s,
                    method='RK45', rtol=rtol, atol=atol,
                    events=[ev_detach, ev_stop_L],
                    max_step=max_step,
                )
                t_new = sol.t[-1]
                if t_new <= t + 1e-14:
                    phase = "flight"
                    continue
                t = t_new
                s = sol.y[:, -1]

                if s[0] >= L - 1e-9:
                    break

                detach_fired = (sol.status == 1 and len(sol.t_events[0]) > 0)
                if detach_fired:
                    phase = "flight"
                else:
                    break

        tc = max(0.0, t - t_flight)
        t_sl = tc + self.kappa * t_flight
        T_lam = t + (self.lam - 1.0) * t_sl
        return T_lam, hits, tc


# ================================================================
#  SOBOL
# ================================================================

def run_one(d_mm, psi, phi):
    """Один детерминированный расчёт."""
    R = d_mm / 2 / 1000
    p = Particle(R=R, psi=psi, phi=phi,
                 alpha_deg=15, A=0.002, nu=25, gamma=0.2,
                 enable_sieve=False)
    T, _, _ = p.simulate(L=0.5)
    return T


def sobol_analysis(N=128, seed=42):
    D = 3
    total_runs = (2 + D) * N
    print(f"=== Sobol-анализ (детерминированная модель) ===")
    print(f"  N = {N}, D = {D}")
    print(f"  Всего расчётов: {total_runs}")
    print()

    sampler = qmc.Sobol(d=2*D, scramble=True, seed=seed)
    AB = sampler.random(2*N)
    lo = np.array([0.8, 0.6, 0.2] * 2)
    hi = np.array([8.0, 1.0, 0.5] * 2)
    AB = qmc.scale(AB, lo, hi)
    A_mat = AB[:N, :D]
    B_mat = AB[N:2*N, :D]

    t0 = time.time()
    counter = [0]

    def run(row):
        counter[0] += 1
        if counter[0] % 32 == 0 or counter[0] == total_runs:
            el = time.time() - t0
            frac = counter[0] / total_runs
            eta = el * (1 - frac) / max(frac, 1e-9)
            print(f"    [{counter[0]:>4}/{total_runs}]  "
                  f"elapsed={el:>6.1f} с  ETA={eta:>6.1f} с")
        return run_one(row[0], row[1], row[2])

    print("  Расчёт A ...")
    y_A = np.array([run(A_mat[i]) for i in range(N)])
    print("  Расчёт B ...")
    y_B = np.array([run(B_mat[i]) for i in range(N)])

    var_all = np.var(np.concatenate([y_A, y_B]))
    print(f"\n  Var(T) = {var_all:.6f}")
    print(f"  <T>    = {np.mean(np.concatenate([y_A, y_B])):.4f} с\n")

    S_i = np.zeros(D)
    S_Ti = np.zeros(D)

    for j in range(D):
        AB_j = A_mat.copy()
        AB_j[:, j] = B_mat[:, j]
        print(f"  Расчёт AB_{j+1} (замена столбца {j+1}) ...")
        y_ABj = np.array([run(AB_j[i]) for i in range(N)])
        S_i[j]  = np.mean(y_B * (y_ABj - y_A)) / var_all
        S_Ti[j] = 0.5 * np.mean((y_A - y_ABj)**2) / var_all

    el = time.time() - t0
    print(f"\n  Завершено за {el:.1f} с\n")

    names = ["d (размер)", "psi (форма)", "phi (концентрация)"]
    print("=" * 55)
    print(f"  {'Параметр':>22} {'S_i':>12} {'S_Ti':>12}")
    print("-" * 55)
    for j, nm in enumerate(names):
        print(f"  {nm:>22} {S_i[j]:>12.4f} {S_Ti[j]:>12.4f}")
    print("=" * 55)
    print(f"  {'Сумма':>22} {S_i.sum():>12.4f} {S_Ti.sum():>12.4f}")

    return S_i, S_Ti


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    S_i, S_Ti = sobol_analysis(N=128)

    print("\nИнтерпретация:")
    names = ["d (размер)", "psi (форма)", "phi (концентрация)"]
    for j, nm in enumerate(names):
        if S_i[j] < 0.01:
            st = "не влияет"
        elif S_i[j] < 0.10:
            st = "слабый вклад"
        elif S_i[j] < 0.30:
            st = "умеренный вклад"
        else:
            st = "доминирующий фактор"
        print(f"  {nm:>22}: {st} (S_i = {S_i[j]:.4f})")

    total = S_i.sum()
    print(f"\n  Сумма главных индексов = {total:.4f}")
    if total < 0.9:
        print(f"  → есть взаимодействия (необъяснено {(1-total)*100:.1f}%)")
    else:
        print(f"  → модель практически аддитивна")