import math
from typing import Dict, Sequence

DEFAULT_TC_CURVES: Dict[str, Sequence[float]] = dict(
    speed_ratio=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    torque_ratio=(2.4, 2.2, 2.0, 1.8, 1.65, 1.5, 1.35, 1.2, 1.1, 1.02, 1.0),
    k_norm=(1.00, 1.02, 1.04, 1.07, 1.10, 1.13, 1.16, 1.18, 1.20, 1.21, 1.22),
    lockup_sr=0.92,
)


def interp_lin(x: float, xp: Sequence[float], fp: Sequence[float]) -> float:
    if not xp:
        return float(fp[0]) if fp else 0.0
    if x <= xp[0]:
        return float(fp[0])
    if x >= xp[-1]:
        return float(fp[-1])
    lo, hi = 0, len(xp) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x >= xp[mid]:
            lo = mid
        else:
            hi = mid
    x0, x1 = xp[lo], xp[hi]
    y0, y1 = fp[lo], fp[hi]
    if x1 == x0:
        return float(y0)
    t = (x - x0) / (x1 - x0)
    return float(y0 + t * (y1 - y0))


def bisect_root(fun, a: float, b: float, tol: float = 1e-6, itmax: int = 60) -> float:
    fa, fb = fun(a), fun(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        return a if abs(fa) < abs(fb) else b
    left, right = a, b
    f_left, f_right = fa, fb
    for _ in range(itmax):
        mid = 0.5 * (left + right)
        f_mid = fun(mid)
        if abs(f_mid) < tol or 0.5 * (right - left) < tol:
            return mid
        if f_left * f_mid <= 0.0:
            right, f_right = mid, f_mid
        else:
            left, f_left = mid, f_mid
    return 0.5 * (left + right)


def solve_tc_state(omega_turb: float,
                   Te_fn,
                   tc_curves: Dict[str, Sequence[float]],
                   stall_rpm: float) -> Dict[str, float]:
    sr_grid = tc_curves.get("speed_ratio", DEFAULT_TC_CURVES["speed_ratio"])
    TR_grid = tc_curves.get("torque_ratio", DEFAULT_TC_CURVES["torque_ratio"])
    Knorm_grid = tc_curves.get("k_norm", DEFAULT_TC_CURVES["k_norm"])
    lockup_sr = float(tc_curves.get("lockup_sr", DEFAULT_TC_CURVES.get("lockup_sr", 0.92)))

    stall_rpm = float(max(stall_rpm, 1.0))
    omega_stall = stall_rpm * 2.0 * math.pi / 60.0
    T_stall = max(float(Te_fn(omega_stall)), 1e-6)
    K0 = stall_rpm / math.sqrt(T_stall)

    def K(sr: float) -> float:
        return K0 * interp_lin(sr, sr_grid, Knorm_grid)

    sr_min = 1e-3
    sr_max = 1.0

    def resid(sr: float) -> float:
        sr = max(min(sr, 1.0), sr_min)
        omega_pump = omega_turb / sr if sr > 0.0 else 0.0
        N_pump = omega_pump * 60.0 / (2.0 * math.pi)
        Ksr = max(K(sr), 1e-9)
        T_pump_req = (N_pump / Ksr) ** 2
        return float(Te_fn(omega_pump) - T_pump_req)

    sr = bisect_root(resid, sr_min, sr_max)
    if not math.isfinite(sr):
        sr = 1.0
    sr = max(min(sr, 1.0), sr_min)

    if sr >= lockup_sr and lockup_sr < 1.0:
        omega_engine = omega_turb
        T_pump = float(Te_fn(omega_engine))
        return dict(sr=1.0, omega_pump=omega_engine, T_pump=T_pump,
                    T_turb=T_pump, TR=1.0, eta_tc=1.0)

    TR = interp_lin(sr, sr_grid, TR_grid)
    omega_engine = omega_turb / sr if sr > 0.0 else 0.0
    T_pump = float(Te_fn(omega_engine))
    T_turb = TR * T_pump
    eta_tc = TR * sr
    return dict(sr=sr, omega_pump=omega_engine, T_pump=T_pump,
                T_turb=T_turb, TR=TR, eta_tc=eta_tc)
