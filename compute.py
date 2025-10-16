import numpy as np

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

    return M



def rebuild_maps(v: Dict, M: Dict) -> Dict:

    bsfc = np.asarray(M["bsfc_map_gpkWh"], dtype=float)

    eta = (3.6e6) / ((bsfc / 1000.0) * v["LHV"])

    eta[~np.isfinite(bsfc)] = np.nan

    M["eta_map"] = eta

    return M



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


    def gear_of(vk):

        gi = 0

        while gi < nG-1 and vk >= shift_kmh[gi]:

            gi += 1

        return gi


    t = 0.0; v_kmh = 0.0; fuel_cum_L = 0.0

    dv_mps = dv_kmh/3.6

    t_vec  = [0.0]; v_vec = [0.0]

    rpm_hist = []; T_hist = []; a_hist = []; bsfc_hist = []; fuel_hist = [0.0]; gear_hist = []


    while v_kmh < v_goal_kmh - 1e-12:

        gi_sched = gear_of(v_kmh)

        v_mps = v_kmh/3.6

        gi = gi_sched

        w_from_wheels = (v_mps / v["re"]) * (v["gears"][gi]*v["fd"])

        if force_redline and w_from_wheels >= red_w and gi < nG-1:

            gi += 1

            w_from_wheels = (v_mps / v["re"]) * (v["gears"][gi]*v["fd"])

        w_eff = min(max(w_from_wheels, idle_w), red_w)


        T_eng = max(0.0, T_CT(w_eff) + lam*(T_WOT(w_eff) - T_CT(w_eff)))

        F_wheel = T_eng * (v["gears"][gi]*v["fd"]*v["eta_dl"]) / v["re"]

        Rtot    = F_roll(v, v_mps) + F_aero(v, v_mps)

        a = (F_wheel - Rtot) / v["m"]

        if a <= 1e-12:

            break

        dt = dv_mps / a


        P_kW = T_eng*w_eff/1000.0

        bsfc_gpkWh = bsfc_fn(np.array([T_eng]), np.array([w_eff]))[0]

        if not np.isfinite(bsfc_gpkWh):

            # outside map; bail

            break

        mdot_gps = bsfc_gpkWh * P_kW / 3600.0

        fuel_step_L = (mdot_gps * dt) / (rho_fuel_kg_per_L*1000.0)

        fuel_cum_L += max(0.0, fuel_step_L)


        t += dt

        v_kmh = min(v_kmh + dv_kmh, v_goal_kmh)


        rpm_hist.append(w_eff*60.0/(2*np.pi))

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

