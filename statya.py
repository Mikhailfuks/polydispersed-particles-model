
import os
import sys
import time
import sqlite3
import argparse
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc, t as t_dist

import matplotlib
matplotlib.use("Agg")  # без GUI, чтобы работало на сервере
import matplotlib.pyplot as plt


# ============================================================================
#  КОНСТАНТЫ
# ============================================================================

RHO_AIR = 1.205       # плотность воздуха, кг/м3
NU_AIR  = 1.5e-5      # кинематическая вязкость воздуха, м2/с
G       = 9.81        # ускорение свободного падения, м/с2

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grokhot.db")

# Файлы графиков
PLOT_SIZE = "plot_size.png"
PLOT_PSI  = "plot_psi.png"
PLOT_GAMMA = "plot_gamma.png"
PLOT_MC   = "plot_mc_hist.png"
PLOT_SOBOL = "plot_sobol.png"
PLOT_VERIFY = "plot_verify.png"
PLOT_CONV = "plot_convergence.png"

# Параметры задачи по умолчанию
DEFAULTS = dict(
    alpha_deg = 15.0,     # угол наклона, град
    A         = 0.002,    # амплитуда, м
    nu        = 25.0,     # частота, Гц
    gamma     = 0.2,      # параметр асимметрии
    f         = 0.3,      # коэффициент трения скольжения
    fs        = 0.45,     # коэффициент трения покоя
    rho_p     = 2650.0,   # плотность частицы, кг/м3
    beta      = 2.5,      # коэф. взаимодействия
    k_psi     = 4.0,      # коэф. формы в Cd
    d_hole    = 0.008,    # размер отверстия, м
    p0        = 0.7,      # базовая вероятность прохода
    kappa     = 0.15,     # доля t_flight в t_сл
    L         = 0.5,      # длина сита, м
)


class Particle:
    
    CONTACT_EPS = 1e-7

    def __init__(self, R, psi, phi, alpha_deg=None, A=None, nu=None,
                 gamma=None, f=None, fs=None, rho_p=None, beta=None,
                 k_psi=None, d_hole=None, p0=None, kappa=None,
                 classic=False, enable_sieve=True):
        # --- форма ---
        self.classic = classic
        self.enable_sieve = enable_sieve
        self.R = R
        self.psi = 1.0 if classic else psi
        self.R_eff = R / np.sqrt(self.psi)

        # --- масса и геометрия ---
        rho_p = rho_p if rho_p is not None else DEFAULTS["rho_p"]
        self.m = (4.0/3.0) * np.pi * self.R_eff**3 * rho_p

        if classic:
            self.S_m     = np.pi * self.R_eff**2
            self.cd_form = 1.0
            self.lam     = 1.0
            self.k       = 0.75
        else:
            self.S_m     = np.pi * self.R_eff**2 * (1 + 0.5*(1 - self.psi))
            k_psi        = k_psi if k_psi is not None else DEFAULTS["k_psi"]
            self.cd_form = 1 + k_psi * (1 - self.psi)
            beta         = beta if beta is not None else DEFAULTS["beta"]
            self.lam     = 1 + beta * phi
            self.k       = 0.55 + 0.2 * self.psi

        # --- кинематика поверхности ---
        self.alpha = np.radians(alpha_deg if alpha_deg is not None else DEFAULTS["alpha_deg"])
        self.A     = A     if A     is not None else DEFAULTS["A"]
        self.nu    = nu    if nu    is not None else DEFAULTS["nu"]
        self.gamma = gamma if gamma is not None else DEFAULTS["gamma"]
        self.w     = 2 * np.pi * self.nu

        # --- трение ---
        self.f  = f  if f  is not None else DEFAULTS["f"]
        self.fs = fs if fs is not None else DEFAULTS["fs"]

        # --- просеивание ---
        self.d_hole = d_hole if d_hole is not None else DEFAULTS["d_hole"]
        self.p0     = p0     if p0     is not None else DEFAULTS["p0"]

        # --- взаимодействие ---
        self.kappa = kappa if kappa is not None else DEFAULTS["kappa"]

    # ------- кинематика поверхности -------
    def xi(self, t):
        """Смещение поверхности."""
        return self.A * (np.sin(self.w*t) + self.gamma * np.sin(2*self.w*t))

    def dxi(self, t):
        """Скорость поверхности."""
        return self.A * self.w * (np.cos(self.w*t) +
                                   2*self.gamma * np.cos(2*self.w*t))

    def ddxi(self, t):
        """Ускорение поверхности."""
        return -self.A * self.w**2 * (np.sin(self.w*t) +
                                       4*self.gamma * np.sin(2*self.w*t))

    # ------- вероятность прохода -------
    def prob_pass(self):
        """Вероятность просеивания за один период."""
        if not self.enable_sieve:
            return 0.0
        ratio = (2 * self.R) / self.d_hole
        if ratio >= 0.95:
            return 0.0
        return self.p0 * np.exp(-(ratio / 0.5)**2)

    # ------- аэродинамика -------
    def Cd(self, Re):
        """Коэффициент сопротивления (аппроксимация Клячко)."""
        if self.classic or Re < 1e-8:
            return 0.0
        if Re < 1.0:
            return 24.0 / Re                              # Стокс
        if Re < 400.0:
            return 24.0/Re + 4.0/Re**(1.0/3.0)            # Клячко
        if Re < 1000.0:
            # сглаженный переход к 0.44
            w = (Re - 400.0) / 600.0
            Cd_400 = 24.0/400.0 + 4.0/400.0**(1.0/3.0)
            return Cd_400 * (1-w) + 0.44 * w
        return 0.44

    def drag(self, vx, vy):
        """Сила аэродинамического сопротивления."""
        if self.classic:
            return 0.0, 0.0
        v = np.hypot(vx, vy)
        if v < 1e-12:
            return 0.0, 0.0
        Re = 2 * self.R_eff * v / NU_AIR
        Cd = self.Cd(Re) * self.cd_form
        Fd = 0.5 * Cd * RHO_AIR * self.S_m * v**2
        return Fd * vx / v, Fd * vy / v

    # ------- правая часть ОДУ: полёт -------
    def rhs_flight(self, t, s):
        """Уравнения движения в фазе полёта."""
        x, y, vx, vy = s
        Fgx =  self.m * G * np.sin(self.alpha)
        Fgy = -self.m * G * np.cos(self.alpha)
        Fdx, Fdy = self.drag(vx, vy)
        return [vx, vy, (Fgx+Fdx)/self.m, (Fgy+Fdy)/self.m]

    # ------- правая часть ОДУ: скольжение -------
    def rhs_slide(self, t, s):
        """Уравнения движения в фазе скольжения."""
        x, y, vx, vy = s
        Fgx =  self.m * G * np.sin(self.alpha)
        Fgy = -self.m * G * np.cos(self.alpha) - self.m * self.ddxi(t)
        Fdx, Fdy = self.drag(vx, vy)

        N = max(0.0, -(Fgy + Fdy))

        # трение
        if abs(vx) > 1e-4:
            Ftr = self.f * N * np.sign(vx)
        else:
            Fsh = Fgx + Fdx
            Ftr = Fsh if abs(Fsh) <= self.fs * N else self.fs * N * np.sign(Fsh)

        ax = (Fgx + Fdx - Ftr) / self.m
        return [vx, self.dxi(t), ax, self.ddxi(t)]

    # ------- события -------
    def ev_land(self, t, s):
        """Касание поверхности."""
        return s[1] - (self.xi(t) + self.R_eff)
    ev_land.terminal = True
    ev_land.direction = -1

    # ------- главный метод -------
    def simulate(self, t_max=5.0, L=None, rtol=1e-6, atol=1e-8,
                 y_offset=1e-4, seed=None):
        """
        Численное интегрирование.

        Возвращает:
            T:           время пребывания, с
            hits:        число ударов
            tc:          суммарное время контакта, с
            x_exit:      координата выхода
            sieve_passed: True, если частица просеялась
        """
        L = L if L is not None else DEFAULTS["L"]
        rng = np.random.default_rng(seed)
        max_step = 1.0 / (self.nu * 100.0)

        # однократное разыгрывание "судьбы"
        will_sieve = rng.random() < self.prob_pass()
        if will_sieve:
            x_target = float(rng.uniform(0.10 * L, 0.95 * L))
        else:
            x_target = float('inf')

        def ev_stop(t, s):
            return s[0] - L
        ev_stop.terminal = True
        ev_stop.direction = 1

        t = 0.0
        y0 = self.xi(0.0) + self.R_eff + y_offset
        s = np.array([0.0, y0, 0.0, 0.0])

        hits = 0
        t_flight = 0.0
        phase = "flight"
        sieve_passed = False
        t_sieve = None
        x_sieve = None

        for _ in range(2000):
            # проверки завершения
            if t >= t_max or s[0] >= L - 1e-9:
                break
            if s[0] >= x_target:
                sieve_passed = True
                t_sieve = t
                x_sieve = float(s[0])
                break

            if phase == "flight":
                sol = solve_ivp(
                    self.rhs_flight, [t, t_max], s,
                    method='RK45', rtol=rtol, atol=atol,
                    events=[self.ev_land, ev_stop],
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
                if s[0] >= x_target:
                    sieve_passed = True
                    t_sieve = t
                    x_sieve = float(s[0])
                    break

                landed = (sol.status == 1 and len(sol.t_events[0]) > 0)
                if not landed:
                    break

                # удар
                dxi_t = self.dxi(t)
                vy_rel = s[3] - dxi_t
                if vy_rel < 0:
                    s[3] = dxi_t - self.k * vy_rel
                    s[2] -= min(self.fs * abs(vy_rel), abs(s[2])) * np.sign(s[2])
                    hits += 1
                s[1] = self.xi(t) + self.R_eff
                phase = "slide"

            else:  # slide
                Fgy = -self.m * G * np.cos(self.alpha) - self.m * self.ddxi(t)
                _, Fdy = self.drag(s[2], s[3])
                if -(Fgy + Fdy) <= 0:
                    phase = "flight"
                    continue

                def ev_detach(t, s):
                    Fgy = -self.m * G * np.cos(self.alpha) - self.m * self.ddxi(t)
                    _, Fdy = self.drag(s[2], s[3])
                    return -(Fgy + Fdy)
                ev_detach.terminal = True
                ev_detach.direction = -1

                sol = solve_ivp(
                    self.rhs_slide, [t, t_max], s,
                    method='RK45', rtol=rtol, atol=atol,
                    events=[ev_detach, ev_stop],
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
                if s[0] >= x_target:
                    sieve_passed = True
                    t_sieve = t
                    x_sieve = float(s[0])
                    break

                detach_fired = (sol.status == 1 and len(sol.t_events[0]) > 0)
                if detach_fired:
                    phase = "flight"
                else:
                    break

        # время контакта и полное время
        tc = max(0.0, t - t_flight)
        T_full = t + (self.lam - 1.0) * (tc + self.kappa * t_flight)

        if sieve_passed and t_sieve is not None:
            return max(0.05, t_sieve), hits, tc, x_sieve, True
        return T_full, hits, tc, float(s[0]), False


# ============================================================================
#  БАЗА ДАННЫХ SQLITE
# ============================================================================

def db_init():
    """Создать/проверить схему БД."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            run_type   TEXT    NOT NULL,
            mode       TEXT    NOT NULL,
            d_mm       REAL,
            psi        REAL,
            phi        REAL,
            alpha      REAL,
            A          REAL,
            nu         REAL,
            gamma      REAL,
            T          REAL,
            M          INTEGER,
            tc         REAL,
            sieve      INTEGER,
            rtol       REAL,
            atol       REAL,
            timestamp  TEXT
        )
    """)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    if "sieve" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN sieve INTEGER DEFAULT 0")
    conn.commit()
    return conn


def db_clear(conn, run_type):
    """Очистить результаты определённого типа."""
    conn.execute("DELETE FROM results WHERE run_type = ?", (run_type,))
    conn.commit()


def db_save(conn, run_type, mode, d_mm, psi, phi, alpha, A, nu, gamma,
            T, M, tc, sieve=False, rtol=None, atol=None):
    """Сохранить результат."""
    conn.execute("""
        INSERT INTO results
        (run_type, mode, d_mm, psi, phi, alpha, A, nu, gamma,
         T, M, tc, sieve, rtol, atol, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (run_type, mode, d_mm, psi, phi, alpha, A, nu, gamma,
          T, M, tc, int(sieve), rtol, atol, datetime.now().isoformat()))
    conn.commit()


def db_fetch(conn, run_type, mode=None, order_by="id"):
    """Извлечь результаты."""
    sql = ("SELECT d_mm, psi, phi, alpha, A, nu, gamma, T, M, tc, "
           "sieve, rtol, atol, mode FROM results WHERE run_type = ?")
    args = [run_type]
    if mode:
        sql += " AND mode = ?"
        args.append(mode)
    sql += f" ORDER BY {order_by}"
    rows = conn.execute(sql, args).fetchall()
    cols = ["d_mm", "psi", "phi", "alpha", "A", "nu",
            "gamma", "T", "M", "tc", "sieve", "rtol", "atol", "mode"]
    return [dict(zip(cols, r)) for r in rows]


# ============================================================================
#  ВЕРИФИКАЦИЯ
# ============================================================================

def test_analytic():
    """Аналитическая проверка: T = sqrt(2L/(g*sin(alpha)))."""
    print("=" * 70)
    print("АНАЛИТИЧЕСКАЯ ПРОВЕРКА")
    print("=" * 70)

    alpha_deg = 15.0
    L = 0.5
    T_analytic = np.sqrt(2 * L / (G * np.sin(np.radians(alpha_deg))))

    p = Particle(R=0.001, psi=1.0, phi=0.0,
                 alpha_deg=alpha_deg, A=0.0, nu=1.0, gamma=0.0,
                 f=0.0, fs=0.0, classic=True, enable_sieve=False)
    T_num, M, tc, x_exit, _ = p.simulate(t_max=5.0, L=L, y_offset=1e-4)

    err = 100 * abs(T_num - T_analytic) / T_analytic
    print(f"  T_аналит = {T_analytic:.4f} с")
    print(f"  T_числ   = {T_num:.4f} с")
    print(f"  отклонение = {err:.2f} %")
    print()

    return err


def test_verification():
    """Верификация по эксперименту Standish & Meta (1985)."""
    print("=" * 70)
    print("ВЕРИФИКАЦИЯ: Standish & Meta (1985)")
    print("=" * 70)

    # ферросплав -4.55 мм, alpha=11.5, nu=24, эксперимент 1.2 ± 0.15 с
    p = Particle(R=0.0035/2, psi=0.75, phi=0.0,
                 alpha_deg=11.5, A=0.006, nu=24, gamma=0.0,
                 beta=0.0, rho_p=4000.0, enable_sieve=False)
    T, M, tc, _, _ = p.simulate(t_max=5.0, L=0.5)

    print(f"  эксперимент: 1.2 ± 0.15 с")
    print(f"  расчёт:      {T:.3f} с")
    ok = 1.05 <= T <= 1.35
    print(f"  попадание в интервал: {'да' if ok else 'нет'}")
    print()

    return T


# ============================================================================
#  СВИПЫ ПАРАМЕТРОВ
# ============================================================================

def run_convergence(conn, quiet=False):
    """Сеточная сходимость."""
    if not quiet:
        print("=" * 70)
        print("СЕТОЧНАЯ СХОДИМОСТЬ")
        print("=" * 70)
        print(f"{'rtol':>8} {'atol':>10} {'T, с':>8} {'M':>5}")
    db_clear(conn, "convergence")

    for rtol, atol in [(1e-4, 1e-6), (1e-5, 1e-7), (1e-6, 1e-8),
                       (1e-7, 1e-9), (1e-8, 1e-10)]:
        p = Particle(R=0.0005, psi=0.75, phi=0.35,
                     enable_sieve=False)
        T, M, tc, _, _ = p.simulate(rtol=rtol, atol=atol, L=0.5)
        if not quiet:
            print(f"{rtol:>8.0e} {atol:>10.0e} {T:>8.3f} {M:>5d}")
        db_save(conn, "convergence", "full", 1.0, 0.75, 0.35,
                15, 2, 25, 0.2, T, M, tc, rtol=rtol, atol=atol)
    if not quiet:
        print()


def run_size_sweep(conn, n_real=30, quiet=False):
    """Влияние размера частиц."""
    if not quiet:
        print("=" * 70)
        print("ВЛИЯНИЕ РАЗМЕРА ЧАСТИЦ")
        print("=" * 70)
        print(f"{'d, мм':>7} {'P(d)':>7} {'T, с':>8} {'M':>5} {'%пр':>7}")
    db_clear(conn, "size")

    for d_mm in [0.8, 1.0, 2.0, 3.0, 5.0, 8.0]:
        R = d_mm / 2 / 1000
        T_list, M_list, S_list = [], [], []
        for sd in range(n_real):
            p = Particle(R=R, psi=0.75, phi=0.35)
            T, M, tc, _, sv = p.simulate(L=0.5, seed=42+sd)
            T_list.append(T); M_list.append(M); S_list.append(sv)
        Tm = float(np.mean(T_list))
        Mm = float(np.mean(M_list))
        Sf = float(np.mean(S_list))
        p_model = Particle(R=R, psi=0.75, phi=0.35).prob_pass()
        db_save(conn, "size", "full", d_mm, 0.75, 0.35,
                15, 2, 25, 0.2, Tm, int(round(Mm)), 0.0, sieve=(Sf > 0.5))
        if not quiet:
            print(f"{d_mm:>7.1f} {p_model:>7.3f} {Tm:>8.3f} "
                  f"{Mm:>5.1f} {100*Sf:>6.1f}%")
    if not quiet:
        print()


def run_psi_sweep(conn, n_real=3, quiet=False):
    """Влияние фактора формы."""
    if not quiet:
        print("=" * 70)
        print("ВЛИЯНИЕ ФАКТОРА ФОРМЫ (d=2 мм)")
        print("=" * 70)
        print(f"{'psi':>6} {'k(psi)':>8} {'T, с':>8}")
    db_clear(conn, "psi")

    for psi in [1.0, 0.85, 0.75, 0.65, 0.6]:
        T_list = []
        for sd in range(n_real):
            p = Particle(R=0.001, psi=psi, phi=0.35,
                         enable_sieve=False)
            T, _, _, _, _ = p.simulate(L=0.5, seed=42+sd)
            T_list.append(T)
        Tm = float(np.mean(T_list))
        db_save(conn, "psi", "full", 2.0, psi, 0.35,
                15, 2, 25, 0.2, Tm, 0, 0.0)
        if not quiet:
            print(f"{psi:>6.2f} {0.55+0.2*psi:>8.3f} {Tm:>8.3f}")
    if not quiet:
        print()


def run_gamma_sweep(conn, n_real=3, quiet=False):
    """Влияние асимметрии."""
    if not quiet:
        print("=" * 70)
        print("ВЛИЯНИЕ АСИММЕТРИИ (d=2 мм)")
        print("=" * 70)
        print(f"{'gamma':>7} {'T, с':>8}")
    db_clear(conn, "gamma")

    for gamma in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        T_list = []
        for sd in range(n_real):
            p = Particle(R=0.001, psi=0.75, phi=0.35,
                         gamma=gamma, enable_sieve=False)
            T, _, _, _, _ = p.simulate(L=0.5, seed=42+sd)
            T_list.append(T)
        Tm = float(np.mean(T_list))
        db_save(conn, "gamma", "full", 2.0, 0.75, 0.35,
                15, 2, 25, gamma, Tm, 0, 0.0)
        if not quiet:
            print(f"{gamma:>7.2f} {Tm:>8.3f}")
    if not quiet:
        print()


# ============================================================================
#  АНСАМБЛЬ (LHS) И КОРРЕЛЯЦИИ
# ============================================================================

def run_monte_carlo(conn, n=200, seed=42, quiet=False):
    """Параметрическое исследование ансамбля."""
    if not quiet:
        print("=" * 70)
        print(f"АНСАМБЛЬ LHS (N = {n})")
        print("=" * 70)
    db_clear(conn, "monte_carlo")

    sampler = qmc.LatinHypercube(d=3, seed=seed)
    X = qmc.scale(sampler.random(n),
                  np.array([0.8, 0.6, 0.2]),
                  np.array([8.0, 1.0, 0.5]))

    T_arr, M_arr = [], []
    sieve_count = 0
    t0 = time.time()

    for i, (d_mm, psi, phi) in enumerate(X):
        R = d_mm / 2 / 1000
        p = Particle(R=R, psi=psi, phi=phi)
        T, M, tc, _, sv = p.simulate(L=0.5, seed=1000+i)
        T_arr.append(T); M_arr.append(M)
        if sv:
            sieve_count += 1
        db_save(conn, "monte_carlo", "full", d_mm, psi, phi,
                15, 2, 25, 0.2, T, M, tc, sieve=sv)
        if not quiet and (i+1) % 40 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{n}]  {el:.1f} с, просеяно={sieve_count}, "
                  f"T_mean={np.mean(T_arr):.3f}")

    T_arr = np.array(T_arr)
    M_arr = np.array(M_arr, dtype=float)

    r_d   = float(np.corrcoef(T_arr, X[:, 0])[0, 1])
    r_psi = float(np.corrcoef(T_arr, X[:, 1])[0, 1])
    r_phi = float(np.corrcoef(T_arr, X[:, 2])[0, 1])

    if not quiet:
        print(f"\n  T: mean={T_arr.mean():.3f}  std={T_arr.std():.3f}  "
              f"min={T_arr.min():.3f}  max={T_arr.max():.3f}")
        print(f"  M: mean={M_arr.mean():.2f}  std={M_arr.std():.2f}")
        print(f"  Просеялось: {sieve_count}/{n} "
              f"({100*sieve_count/n:.1f}%)\n")
        print(f"  Корреляции:")
        for nm, r_val in [("T-d", r_d), ("T-psi", r_psi), ("T-phi", r_phi)]:
            ts = r_val * np.sqrt((n-2) / (1 - r_val**2 + 1e-12))
            pv = 2 * (1 - t_dist.cdf(abs(ts), df=n-2))
            sig = "значима" if pv < 0.05 else "не значима"
            print(f"    {nm:>7}: r={r_val:+.3f}, p={pv:.2e} — {sig}")
        print()

    return T_arr, X


# ============================================================================
#  SOBOL-АНАЛИЗ (схема Солтэлли)
# ============================================================================

def run_sobol(conn, N=128, seed=42, quiet=False):
    """Sobol-анализ чувствительности (детерминированная модель)."""
    if not quiet:
        print("=" * 70)
        print(f"SOBOL-АНАЛИЗ (N={N}, детерминированная модель)")
        print("=" * 70)

    D = 3
    sampler = qmc.Sobol(d=2*D, scramble=True, seed=seed)
    AB = sampler.random(2*N)
    lo = np.array([0.8, 0.6, 0.2] * 2)
    hi = np.array([8.0, 1.0, 0.5] * 2)
    AB = qmc.scale(AB, lo, hi)
    A_mat = AB[:N, :D]
    B_mat = AB[N:2*N, :D]

    def run_one(row):
        d_mm, psi, phi = row[0], row[1], row[2]
        R = d_mm / 2 / 1000
        p = Particle(R=R, psi=psi, phi=phi, enable_sieve=False)
        T, _, _, _, _ = p.simulate(L=0.5)
        return T

    t0 = time.time()

    if not quiet:
        print("  Матрица A ...")
    y_A = np.array([run_one(A_mat[i]) for i in range(N)])
    if not quiet:
        print("  Матрица B ...")
    y_B = np.array([run_one(B_mat[i]) for i in range(N)])

    var_all = np.var(np.concatenate([y_A, y_B]))

    S_i  = np.zeros(D)
    S_Ti = np.zeros(D)

    for j in range(D):
        AB_j = A_mat.copy(); AB_j[:, j] = B_mat[:, j]
        if not quiet:
            print(f"  Матрица AB_{j+1} ...")
        y_ABj = np.array([run_one(AB_j[i]) for i in range(N)])
        S_i[j]  = np.mean(y_B * (y_ABj - y_A)) / var_all
        S_Ti[j] = 0.5 * np.mean((y_A - y_ABj)**2) / var_all

    if not quiet:
        el = time.time() - t0
        print(f"\n  Завершено за {el:.1f} с")
        print(f"  Var(T) = {var_all:.6f}\n")
        names = ["d (размер)", "psi (форма)", "phi (концентрация)"]
        print(f"  {'Параметр':>22} {'S_i':>10} {'S_Ti':>10}")
        for j, nm in enumerate(names):
            print(f"  {nm:>22} {S_i[j]:>10.4f} {S_Ti[j]:>10.4f}")
        print(f"  {'Сумма':>22} {S_i.sum():>10.4f} {S_Ti.sum():>10.4f}")
        print()

    return S_i, S_Ti


# ============================================================================
#  ГРАФИКИ
# ============================================================================

def _series(conn, run_type, x_key):
    """Извлечь (x, T) для графика."""
    rows = db_fetch(conn, run_type)
    x = [r[x_key] for r in rows]
    T = [r["T"] for r in rows]
    return x, T


def plot_all(conn, S_i=None, S_Ti=None):
    """Построить все графики."""
    print("=" * 70)
    print("ГРАФИКИ")
    print("=" * 70)

    # --- 1. Сеточная сходимость ---
    rows = db_fetch(conn, "convergence", order_by="rtol DESC")
    if rows:
        fig, ax = plt.subplots(figsize=(7, 5))
        rtol = [r["rtol"] for r in rows]
        T = [r["T"] for r in rows]
        ax.plot(range(len(rtol)), T, 'o-', color='C3')
        ax.set_xticks(range(len(rtol)))
        ax.set_xticklabels([f"{r:.0e}" for r in rtol], rotation=30)
        ax.set_xlabel("rtol"); ax.set_ylabel("T, с")
        ax.set_title("Сеточная сходимость")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(PLOT_CONV, dpi=150); plt.close(fig)
        print(f"  {PLOT_CONV}")

    # --- 2. Размер ---
    x, T = _series(conn, "size", "d_mm")
    if x:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, T, 's-', color='C0')
        ax.set_xlabel("d, мм"); ax.set_ylabel("T, с")
        ax.set_title("Влияние размера частиц")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(PLOT_SIZE, dpi=150); plt.close(fig)
        print(f"  {PLOT_SIZE}")

    # --- 3. Форма ---
    x, T = _series(conn, "psi", "psi")
    if x:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, T, 's-', color='C1')
        ax.set_xlabel("psi"); ax.set_ylabel("T, с")
        ax.set_title("Влияние фактора формы (d=2 мм)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(PLOT_PSI, dpi=150); plt.close(fig)
        print(f"  {PLOT_PSI}")

    # --- 4. Асимметрия ---
    x, T = _series(conn, "gamma", "gamma")
    if x:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, T, 's-', color='C2')
        ax.set_xlabel("gamma"); ax.set_ylabel("T, с")
        ax.set_title("Влияние асимметрии (d=2 мм)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(PLOT_GAMMA, dpi=150); plt.close(fig)
        print(f"  {PLOT_GAMMA}")

    # --- 5. Гистограмма ансамбля ---
    rows = db_fetch(conn, "monte_carlo")
    if rows:
        T = [r["T"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(T, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel("T, с"); ax.set_ylabel("Частота")
        ax.set_title(f"Распределение T (N={len(T)})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(PLOT_MC, dpi=150); plt.close(fig)
        print(f"  {PLOT_MC}")

    # --- 6. Sobol ---
    if S_i is not None and S_Ti is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(3); w = 0.35
        ax.bar(x - w/2, S_i,  w, label="S_i",  color='C0')
        ax.bar(x + w/2, S_Ti, w, label="S_Ti", color='C1')
        ax.set_xticks(x)
        ax.set_xticklabels(["d", "psi", "phi"])
        ax.set_ylabel("Индекс")
        ax.set_title("Sobol-индексы чувствительности")
        ax.grid(True, alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(PLOT_SOBOL, dpi=150); plt.close(fig)
        print(f"  {PLOT_SOBOL}")

    print()


# ============================================================================
#  ГЛАВНАЯ
# ============================================================================

def run_all(quick=False):
    """Полный цикл: верификация + свипы + MC + Sobol + графики."""
    t0 = time.time()

    # 1. Верификация
    test_analytic()
    test_verification()

    # 2. БД и расчёты
    conn = db_init()

    run_convergence(conn, quiet=quick)
    run_size_sweep(conn, n_real=(10 if quick else 30), quiet=quick)
    run_psi_sweep(conn, n_real=(2 if quick else 3), quiet=quick)
    run_gamma_sweep(conn, n_real=(2 if quick else 3), quiet=quick)
    run_monte_carlo(conn, n=(50 if quick else 200), quiet=quick)

    # 3. Sobol
    if quick:
        S_i, S_Ti = run_sobol(conn, N=32, quiet=True)
    else:
        S_i, S_Ti = run_sobol(conn, N=128, quiet=False)

    # 4. Графики
    plot_all(conn, S_i, S_Ti)

    conn.close()

    el = time.time() - t0
    print(f"Готово за {el:.1f} с")
    print(f"Файл БД: {os.path.abspath(DB_FILE)}")


def run_sobol_only():
    """Только Sobol-анализ."""
    conn = db_init()
    S_i, S_Ti = run_sobol(conn, N=128, quiet=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, S_i,  w, label="S_i",  color='C0')
    ax.bar(x + w/2, S_Ti, w, label="S_Ti", color='C1')
    ax.set_xticks(x); ax.set_xticklabels(["d", "psi", "phi"])
    ax.set_ylabel("Индекс"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOT_SOBOL, dpi=150); plt.close(fig)
    print(f"  {PLOT_SOBOL}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Модель движения частиц на вибрирующей поверхности")
    parser.add_argument("--quick", action="store_true",
                        help="быстрый режим (~30 с)")
    parser.add_argument("--sobol-only", action="store_true",
                        help="только Sobol-анализ")
    args = parser.parse_args()

    if args.sobol_only:
        run_sobol_only()
    else:
        run_all(quick=args.quick)


if __name__ == "__main__":
    main()
