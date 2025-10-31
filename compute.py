import numpy as np

import tc

from typing import Dict, Tuple

from scipy.interpolate import RegularGridInterpolator



# ----- Basic vehicle helpers -----

def compute_tire_radius(v: Dict) -> float:

    # meters; (rim radius + sidewall height)

    return ((v["tire_rim_in"] * 25.4) / 2.0 + (v["tire_ar"] / 100.0) * v["tire_w"]) / 1000.0



def update_derived(v: Dict) -> Dict:

    Krpm2kmh = (2.0 * np.pi * v["re"] / 60.0) * 3.6 / v["fd"]

    v["top_speed_kmh"] = (v["redline_rpm"] * Krpm2kmh) / v["gears"][-1]

    return v



def F_roll(v: Dict, vel_mps):

    return (0.013 + 6.5e-6 * vel_mps**2) * v["m"] * v["g"]



def F_aero(v: Dict, vel_mps):

    return 0.5 * v["rho_air"] * v["Af"] * v["Cd"] * vel_mps**2



# ----- Maps -----

def build_map_from_engine_struct(E: Dict) -> Dict:

    tq = np.asarray(E["fuel_map_torque_Nm"], dtype=float).ravel()

    sp = np.asarray(E["fuel_map_speed_radps"], dtype=float).ravel()

    gps = np.asarray(E["fuel_map_gps"], dtype=float)

    # Make sure shape is [len(tq), len(sp)]

    if gps.shape != (tq.size, sp.size):

        gps = gps.reshape(tq.size, sp.size)


    T, W = np.meshgrid(tq, sp, indexing="ij")

    P_kW = (T * W) / 1000.0

    bsfc = (gps * 3600.0) / np.maximum(P_kW, 1e-12)

    bad = (gps <= 0) | (P_kW <= 0)

    bsfc[bad] = np.nan


    M = dict(

        torque_Nm=tq,

        speed_radps=sp,

        bsfc_map_gpkWh=bsfc,

        wot_speed_radps=np.asarray(E["full_throttle_speed_radps"], dtype=float).ravel(),

        wot_torque_Nm=np.asarray(E["full_throttle_torque_Nm"], dtype=float).ravel(),

        ct_speed_radps=np.asarray(E["closed_throttle_speed_radps"], dtype=float).ravel(),

        ct_torque_Nm=np.asarray(E["closed_throttle_torque_Nm"], dtype=float).ravel(),

    )

    converter = E.get("converter")

    if converter:

        M["converter"] = dict(

            speed_ratio=np.asarray(converter.get("speed_ratio", []), dtype=float).ravel(),

            torque_ratio=np.asarray(converter.get("torque_ratio", []), dtype=float).ravel(),

            k_norm=np.asarray(converter.get("k_norm", []), dtype=float).ravel(),

            lockup_sr=float(converter.get("lockup_sr", 0.92)),

        )

    return M



def rebuild_maps(v: Dict, M: Dict) -> Dict:

    bsfc = np.asarray(M["bsfc_map_gpkWh"], dtype=float)

    eta = (3.6e6) / ((bsfc / 1000.0) * v["LHV"])

    eta[~np.isfinite(bsfc)] = np.nan

    M["eta_map"] = eta

    return M


def _tc_curves_from_map(M: Dict) -> Dict[str, np.ndarray]:

    curves = (M or {}).get("converter")

    if not curves:

        curves = tc.DEFAULT_TC_CURVES

    return dict(

        speed_ratio=np.array(curves.get("speed_ratio", tc.DEFAULT_TC_CURVES["speed_ratio"]), dtype=float),

        torque_ratio=np.array(curves.get("torque_ratio", tc.DEFAULT_TC_CURVES["torque_ratio"]), dtype=float),

        k_norm=np.array(curves.get("k_norm", tc.DEFAULT_TC_CURVES["k_norm"]), dtype=float),

        lockup_sr=float(curves.get("lockup_sr", tc.DEFAULT_TC_CURVES.get("lockup_sr", 0.92))),

    )



def interpolators(M: Dict):

    tq = np.asarray(M["torque_Nm"], dtype=float)

    sp = np.asarray(M["speed_radps"], dtype=float)

    z  = np.asarray(M["bsfc_map_gpkWh"], dtype=float)


    # clamp outside -> nan

    rgi = RegularGridInterpolator((tq, sp), z, bounds_error=False, fill_value=np.nan)


    def bsfc_fn(T: np.ndarray, W: np.ndarray) -> np.ndarray:

        pts = np.column_stack([T, W])

        return rgi(pts)


    w_wot = np.asarray(M["wot_speed_radps"], dtype=float)

    T_wot = np.asarray(M["wot_torque_Nm"], dtype=float)

    w_ct  = np.asarray(M["ct_speed_radps"], dtype=float)

    T_ct  = np.asarray(M["ct_torque_Nm"], dtype=float)


    def _interp1(xq, x, y):

        return np.interp(xq, x, y, left=y[0], right=y[-1])


    T_WOT = lambda w: _interp1(np.asarray(w, dtype=float), w_wot, T_wot)

    T_CT  = lambda w: _interp1(np.asarray(w, dtype=float),  w_ct,  T_ct)

    return bsfc_fn, T_WOT, T_CT



# ----- Cruise / FE -----

def steady_cruise_curve(v: Dict, M: Dict, g_ix: int):

    Krpm2kmh = (2*np.pi*v["re"]/60.0)*3.6 / v["fd"]

    vmin = max(1e-3, (v["idle_rpm"] * Krpm2kmh)/v["gears"][g_ix])

    vmax = (v["redline_rpm"] * Krpm2kmh)/v["gears"][g_ix]

    v_kmh = np.linspace(vmin, vmax, 200)

    v_mps = v_kmh/3.6

    R = F_roll(v, v_mps) + F_aero(v, v_mps)

    T_wheel = R * v["re"]

    omega = (v_mps / v["re"]) * (v["gears"][g_ix]*v["fd"])

    Treq = T_wheel / (v["gears"][g_ix]*v["fd"]*v["eta_dl"])

    rpm = omega * 60/(2*np.pi)


    tq = np.asarray(M["torque_Nm"], dtype=float)

    sp = np.asarray(M["speed_radps"], dtype=float)

    mask = (omega>=sp.min()) & (omega<=sp.max()) & (Treq>=tq.min()) & (Treq<=tq.max())

    rpm[~mask] = np.nan

    Treq[~mask] = np.nan

    return rpm, Treq, v_kmh



def steady_FE_in_gear(v: Dict, M: Dict, g_ix: int, v_kmh):

    v_mps = v_kmh/3.6

    R = F_roll(v, v_mps) + F_aero(v, v_mps)

    T_wheel = R * v["re"]

    omega = (v_mps / v["re"]) * (v["gears"][g_ix]*v["fd"])

    rpm = omega * 60/(2*np.pi)

    Treq = T_wheel / (v["gears"][g_ix]*v["fd"]*v["eta_dl"])


    tq = np.asarray(M["torque_Nm"], dtype=float)

    sp = np.asarray(M["speed_radps"], dtype=float)

    in_range = (omega>=sp.min()) & (omega<=sp.max()) & (Treq>=tq.min()) & (Treq<=tq.max())

    L100 = np.full_like(v_kmh, np.nan, dtype=float)

    if in_range.any():

        bsfc_fn, _, _ = interpolators(M)

        P_kW = (Treq[in_range] * omega[in_range]) / 1000.0

        bsfc = bsfc_fn(Treq[in_range], omega[in_range])       # g/kWh

        mdot = bsfc * P_kW / 3600.0                           # g/s

        Lps  = mdot / (v["rho_fuel"]*1000.0)                  # L/s

        L100[in_range] = (Lps / (v_kmh[in_range]/3600.0)) * 100.0

    return L100, rpm, Treq



# ----- Acceleration integrator (records gear, rpm, torque, shift points) -----

def accel_fuel_to_speed_core(v: Dict, M: Dict, target_kmh: float,

                             shift_kmh, lam: float, opts: Dict) -> Dict:

    rho_fuel_kg_per_L = opts.get("rho_fuel_kg_per_L", v["rho_fuel"])

    idle_rpm          = opts.get("idle_rpm", v["idle_rpm"])

    redline_rpm       = opts.get("redline_rpm", v["redline_rpm"])

    dv_kmh            = opts.get("dv_kmh", 0.25)

    force_redline     = opts.get("force_redline", True)


    nG = len(v["gears"])

    shift_kmh = list(shift_kmh or [])

    if len(shift_kmh) < nG-1:

        # pad strictly increasing if user gave fewer

        base = list(shift_kmh)

        for k in range(len(base), nG-1):

            base.append((k+1) * 30.0)

        shift_kmh = base[:nG-1]


    bsfc_fn, T_WOT, T_CT = interpolators(M)

    idle_w = idle_rpm*2*np.pi/60.0; red_w = redline_rpm*2*np.pi/60.0

    Krpm2kmh = (2*np.pi*v["re"]/60.0)*3.6 / v["fd"]

    v_top_kmh = (redline_rpm * Krpm2kmh)/v["gears"][-1]

    v_goal_kmh = min(target_kmh, v_top_kmh)

    auto_enabled = bool(v.get("auto_trans_enabled"))
    stall_rpm_user = v.get("stall_speed_rpm", 2600.0)
    try:
        stall_rpm_user = float(stall_rpm_user)
    except (TypeError, ValueError):
        stall_rpm_user = 2600.0
    if stall_rpm_user <= 0:
        stall_rpm_user = 2600.0
    stall_rpm_floor = max(float(idle_rpm), 1000.0)
    stall_rpm_use = max(stall_rpm_user, stall_rpm_floor)
    tc_curves = _tc_curves_from_map(M) if auto_enabled else None


    def gear_of(vk):

        gi = 0

        while gi < nG-1 and vk >= shift_kmh[gi]:

            gi += 1

        return gi


    t = 0.0; v_kmh = 0.0; fuel_cum_L = 0.0

    dv_mps = dv_kmh/3.6

    t_vec  = [0.0]; v_vec = [0.0]

    rpm_hist = []; T_hist = []; a_hist = []; bsfc_hist = []; fuel_hist = [0.0]; gear_hist = []

    def clamp_engine_speed(omega):
        return min(max(float(omega), idle_w), red_w)

    def engine_torque_val(omega):
        w = clamp_engine_speed(omega)
        val = T_CT(w) + lam*(T_WOT(w) - T_CT(w))
        return float(max(0.0, val))



    while v_kmh < v_goal_kmh - 1e-12:

        gi_sched = gear_of(v_kmh)

        v_mps = v_kmh/3.6

        gi = gi_sched

        while True:

            gear_ratio = v["gears"][gi]

            omega_turb = (v_mps / v["re"]) * (gear_ratio * v["fd"])

            if auto_enabled:

                res = tc.solve_tc_state(omega_turb, engine_torque_val, tc_curves, stall_rpm_use)

                omega_engine = float(res.get("omega_pump", omega_turb))

                if not np.isfinite(omega_engine) or omega_engine <= 0.0:

                    omega_engine = omega_turb

                T_eng = float(res.get("T_pump", engine_torque_val(omega_engine)))

                T_turb = float(res.get("T_turb", T_eng))

                F_wheel = (T_turb * (gear_ratio * v["fd"] * v["eta_dl"])) / v["re"]

            else:

                omega_engine = clamp_engine_speed(omega_turb)

                T_eng = engine_torque_val(omega_engine)

                F_wheel = T_eng * (gear_ratio * v["fd"] * v["eta_dl"]) / v["re"]

            if force_redline and gi < nG-1 and omega_engine >= red_w:

                gi += 1

                continue

            break

        omega_map = clamp_engine_speed(omega_engine)

        Rtot    = F_roll(v, v_mps) + F_aero(v, v_mps)

        a = (F_wheel - Rtot) / v["m"]

        if a <= 1e-12:

            break

        dt = dv_mps / a

        P_kW = T_eng * omega_map / 1000.0

        bsfc_gpkWh = bsfc_fn(np.array([T_eng]), np.array([omega_map]))[0]

        if not np.isfinite(bsfc_gpkWh):

            break

        mdot_gps = bsfc_gpkWh * P_kW / 3600.0

        fuel_step_L = (mdot_gps * dt) / (rho_fuel_kg_per_L*1000.0)

        fuel_cum_L += max(0.0, fuel_step_L)

        t += dt

        v_kmh = min(v_kmh + dv_kmh, v_goal_kmh)

        rpm_hist.append(omega_engine*60.0/(2*np.pi))

        T_hist.append(T_eng)

        a_hist.append(a)

        bsfc_hist.append(bsfc_gpkWh)

        fuel_hist.append(fuel_cum_L)

        gear_hist.append(gi+1)

        t_vec.append(t); v_vec.append(v_kmh)

    R = dict(

        reached_target=v_kmh >= target_kmh - 1e-9,

        final_speed_kmh=v_kmh,

        time_s=t,

        fuel_L=fuel_cum_L,

        distance_m=np.trapz(np.asarray(v_vec)/3.6, np.asarray(t_vec)),

        trace=dict(

            t_s=np.asarray(t_vec),

            v_kmh=np.asarray(v_vec),

            gear=np.asarray(gear_hist, dtype=int),

            rpm=np.asarray(rpm_hist),

            T_engine_Nm=np.asarray(T_hist),

            a_mps2=np.asarray(a_hist),

            bsfc_gpkWh=np.asarray(bsfc_hist),

            fuel_cum_L=np.asarray(fuel_hist),

        )

    )

    return R

