
from pathlib import Path

from typing import Dict, Any, List


import numpy as np

from dash import Dash, dcc, html, Input, Output, State, no_update, callback, ctx

from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc

import plotly.graph_objects as go

import plotly.io as pio


import compute as C

import engine_io as EIO



# ---------- Theme (dark, desaturated) ----------

DARK_PAPER = "#0d1117"

DARK_PLOT  = "#111418"

DARK_GRID  = "#2a2f36"

DARK_TEXT  = "#e6e7ea"

DARK_MUTED = "#a2a9b4"

DARK_AXIS  = "#c9cdd3"

pio.templates.default = "plotly_dark"


def apply_dark(fig: go.Figure, height=None, title=None):

    fig.update_layout(

        paper_bgcolor=DARK_PAPER,

        plot_bgcolor=DARK_PLOT,

        font=dict(color=DARK_TEXT, size=13),

        legend=dict(bgcolor="rgba(0,0,0,0)"),

        margin=dict(t=50, r=12, l=60, b=50),

        xaxis=dict(gridcolor=DARK_GRID, linecolor=DARK_GRID, zerolinecolor=DARK_GRID,

                   title_font=dict(color=DARK_AXIS), tickfont=dict(color=DARK_TEXT)),

        yaxis=dict(gridcolor=DARK_GRID, linecolor=DARK_GRID, zerolinecolor=DARK_GRID,

                   title_font=dict(color=DARK_AXIS), tickfont=dict(color=DARK_TEXT)),

    )

    if height is not None:

        fig.update_layout(height=height)

    if title:

        fig.update_layout(title=title)

    return fig



# ---------- App ----------

app = Dash(

    __name__,

    external_stylesheets=[dbc.themes.BOOTSTRAP],

    suppress_callback_exceptions=True,  # harmless now, but we keep it

    title="Vehicle Efficiency (Web)",

)

server = app.server


BASE_DIR = Path(__file__).resolve().parent

ENG_DIR  = BASE_DIR / "engines"



# ---------- Defaults ----------

def default_vehicle() -> Dict[str, Any]:

    v = {

        "m": 1340.0,

        "Cd": 0.28,

        "Af": 2.0,

        "rho_air": 1.23,

        "rho_fuel": 0.73,

        "LHV": 42.887e6,

        "g": 9.81,

        "eta_dl": 0.89,

        "tire_w": 215.0, "tire_ar": 45.0, "tire_rim_in": 17.0,

        "fd": 3.85,

        "gears": [3.55, 1.95, 1.30, 1.03, 0.84, 0.68],

        "idle_rpm": 900.0, "redline_rpm": 6500.0

    }

    v["re"] = C.compute_tire_radius(v)

    v = C.update_derived(v)

    return v



def initial_engine_state() -> Dict[str, Any]:

    items: List[Dict[str, Any]] = [EIO.load_builtin_engine()]

    try:

        scanned = EIO.scan_engines_folder(ENG_DIR)

        seen = {items[0]["name"]}

        for e in scanned:

            if e["name"] not in seen:

                items.append(e); seen.add(e["name"])

    except Exception as ex:

        print("Engine scan failed:", ex)

    names = [e["name"] for e in items]

    return {"names": names, "items": items, "active": 0}



def maps_from_state(vstate: Dict[str, Any], eng_state: Dict[str, Any]) -> Dict[str, Any]:

    if not eng_state or not eng_state.get("items"):

        return {}

    e = eng_state["items"][eng_state.get("active", 0)]

    M = C.build_map_from_engine_struct(e)

    M = C.rebuild_maps(vstate, M)

    out = {}

    for k, val in M.items():

        out[k] = val.tolist() if isinstance(val, np.ndarray) else val

    return out



# ---------- Layout helpers ----------

def num(id_, val, step=None, min_=None, max_=None, width="140px"):

    return dbc.Input(id=id_, type="number", value=float(val), step=step, min=min_, max=max_, style={"width": width})


def txt(id_, val, width="340px"):

    return dbc.Input(id=id_, type="text", value=val, style={"width": width})



def tab1_layout(v, eng):

    names = eng.get("names", []) if eng else []

    active_val = names[eng.get("active", 0)] if names else None

    return html.Div([

        dbc.Row([

            dbc.Col(html.Div([

                html.H5("Vehicle & Constants"),

                dbc.Row([dbc.Col(html.Label("Mass [kg]")), dbc.Col(num("mass", v["m"]))]),

                dbc.Row([dbc.Col(html.Label("Cd")), dbc.Col(num("cd", v["Cd"], step=0.01))]),

                dbc.Row([dbc.Col(html.Label("Af [m²]")), dbc.Col(num("af", v["Af"], step=0.01))]),

                dbc.Row([dbc.Col(html.Label("Air ρ [kg/m³]")), dbc.Col(num("rho_air", v["rho_air"], step=0.01))]),

                dbc.Row([dbc.Col(html.Label("Fuel ρ [kg/L]")), dbc.Col(num("rho_fuel", v["rho_fuel"], step=0.001))]),

                dbc.Row([dbc.Col(html.Label("Fuel LHV [MJ/kg]")), dbc.Col(num("lhv_mjpkg", v["LHV"]/1e6, step=0.001))]),

                dbc.Row([dbc.Col(html.Label("g [m/s²]")), dbc.Col(num("g", v["g"], step=0.01))]),

                dbc.Row([dbc.Col(html.Label("η_dl")), dbc.Col(num("eta_dl", v["eta_dl"], step=0.01, min_=0, max_=1))]),

                html.Hr(),

                html.H6("Tire & Driveline"),

                dbc.Row([dbc.Col(html.Label("Final drive")), dbc.Col(num("fd", v["fd"], step=0.01))]),

                dbc.Row([dbc.Col(html.Label("Tire width [mm]")), dbc.Col(num("tire_w", v["tire_w"]))]),

                dbc.Row([dbc.Col(html.Label("Aspect [%]")), dbc.Col(num("tire_ar", v["tire_ar"], step=1))]),

                dbc.Row([dbc.Col(html.Label("Rim [in]")), dbc.Col(num("tire_rim_in", v["tire_rim_in"], step=0.5))]),

                html.Hr(),

                dbc.Row([dbc.Col(html.Label("Gear ratios (space-separated)")), dbc.Col(txt("gear_ratios", " ".join(f"{g:g}" for g in v["gears"])))]),

                dbc.Row([dbc.Col(html.Label("Idle RPM")), dbc.Col(num("idle_rpm", v["idle_rpm"], step=50))]),

                dbc.Row([dbc.Col(html.Label("Redline RPM")), dbc.Col(num("redline_rpm", v["redline_rpm"], step=50))]),

                html.Br(),

                dbc.Button("Apply", id="btn-apply", color="success", className="me-2"),

                dbc.Button("Reset to Defaults", id="btn-reset", color="secondary"),

                html.Br(), html.Br(),

                html.Div(id="derived-label", className="derived")

            ]), md=6),


            dbc.Col(html.Div([

                html.H5("Engine Map"),

                dbc.Row([dbc.Col(html.Label("Select engine")), dbc.Col(

                    dcc.Dropdown(

                        id="engine-select",

                        options=[{"label": n, "value": n} for n in names],

                        value=active_val,

                        placeholder="Select engine...",

                        className="dropdown-dark",

                    ),

                    width=8)]),

                html.Div(className="small-note", children=[

                    f"Engines are auto-loaded from {ENG_DIR} (JSON library)."

                ])

            ]), md=6)

        ])

    ], style={"padding": "12px"})



def tab2_layout(_v):

    return html.Div([

        html.Div([

            html.Label("Loads λ (space-separated)"),

            dbc.Input(id="loads", type="text", value="0.2 0.4 0.6 0.8 1.0", style={"width": "200px"}),

            html.Span("  "),

            dbc.Button("Update plots", id="btn-update-gear", color="primary", size="sm")

        ], className="bar"),

        dbc.Row([

            dbc.Col(dcc.Graph(id="ax-distance-time"), md=6),

            dbc.Col(dcc.Graph(id="ax-speedrpm"), md=6),

        ]),

        dbc.Row([

            dbc.Col(dcc.Graph(id="ax-speed-time-gear"), md=6),

            dbc.Col(dcc.Graph(id="ax-torquespeed"), md=6),

        ]),

        dbc.Row([

            dbc.Col(dcc.Graph(id="ax-accelspeed"), md=6),

            dbc.Col(dcc.Graph(id="ax-accelbottom"), md=6),

        ]),

    ], style={"padding": "12px"})



def tab3_layout(v, _eng):

    nG = len(v.get("gears", [])) if v else 0

    return html.Div([

        dbc.Row([

            dbc.Col(dcc.Graph(id="ax-map", style={"height": "560px"}), md=7),

            dbc.Col([

                dcc.Graph(id="ax-speed-time3", style={"height": "270px"}),

                dcc.Graph(id="ax-fuel-time",  style={"height": "270px"}),

            ], md=5)

        ]),

        html.Div(className="bar-sticky", children=[

            html.Div([

                dcc.Dropdown(

                    id="map-mode",

                    options=[{"label": "Efficiency (%)", "value": "eta"},

                             {"label": "BSFC (g/kWh)", "value": "bsfc"}],

                    value="eta",

                    className="dropdown-dark",

                    style={"width": "220px"}

                ),

            ], className="bar-cell"),

            html.Div([

                html.Label("Cruise gear:"),

                dcc.Dropdown(

                    id="cruise-gear",

                    options=[{"label": str(i+1), "value": i+1} for i in range(nG)],

                    value=nG if nG else None,

                    className="dropdown-dark",

                    style={"width": "110px", "display": "inline-block", "marginLeft": "8px"}

                ),

                dcc.Checklist(

                    id="show-cruise",

                    options=[{"label": " Cruise overlay", "value": 1}],

                    value=[1],

                    style={"display": "inline-block", "marginLeft": "12px"}

                ),

                html.Label("Label step [km/h]:", style={"marginLeft": "12px"}),

                dbc.Input(id="cruise-label-step", type="number", value=10, min=1, step=1,

                          style={"width": "90px", "display": "inline-block", "marginLeft": "6px"}),

            ], className="bar-cell"),

            html.Div([

                html.Label("Shift speeds [km/h]:"),

                dbc.Input(id="shift-speeds", type="text", value="40 65 100 130 160", style={"width": "320px"}),

            ], className="bar-cell"),

            html.Div([

                html.Label("λ:"),

                dbc.Input(id="lambda", type="number", value=0.72, min=0, max=1, step=0.01, style={"width": "90px"}),

                html.Span("  "),

                html.Label("Target [km/h]:"),

                dbc.Input(id="target-kmh", type="number", value=120, min=1, max=400, step=1, style={"width": "110px"}),

                dcc.Checklist(id="force-redline",

                              options=[{"label": " Upshift at redline", "value": 1}],

                              value=[1],

                              style={"display": "inline-block", "marginLeft": "10px"}),

            ], className="bar-cell"),

            html.Div([

                dbc.Button("Add sequence", id="btn-add-seq", color="primary", className="me-2"),

                dbc.Button("Clear sequences", id="btn-clear-seq", color="danger"),

            ], className="bar-cell"),

        ])

    ], style={"padding": "12px"})



def tab4_layout(_v):

    return html.Div([

        html.Div(className="bar", children=[

            html.Label("Label step [km/h]:"),

            dbc.Input(id="fe-label-step", type="number", value=10, min=1, step=1, style={"width": "100px"}),

            html.Span("  "),

            html.Label("Show gears:"),

            dcc.Checklist(id="fe-show-gears", inline=True, style={"marginLeft": "8px"})

        ]),

        dcc.Graph(id="ax-fe", style={"height": "720px"})

    ], style={"padding": "12px"})



# ---------- Static pages: keep all tabs mounted ----------

v0   = default_vehicle()

eng0 = initial_engine_state()

maps0 = maps_from_state(v0, eng0)


app.layout = html.Div([
    dcc.Store(id="store-vehicle", data=v0),
    dcc.Store(id="store-engine",  data=eng0),
    dcc.Store(id="store-maps",    data=maps0),
    dcc.Store(id="store-sequences", data=[]),
    dcc.Interval(id="iv-scan", interval=8000, n_intervals=0),

    # Only the tab headers/toggles live inside dcc.Tabs.
    dcc.Tabs(
        id="tabs",
        value="tab1",
        parent_className="tabs",
        children=[
            dcc.Tab(label="1) Vehicle & Simulation", value="tab1"),
            dcc.Tab(label="2) Gearing & Accel",      value="tab2"),
            dcc.Tab(label="3) Maps & Sequences",     value="tab3"),
            dcc.Tab(label="4) Steady-State FE",      value="tab4"),
        ],
    ),

    # The actual tab page contents are siblings of Tabs, always mounted.
    html.Div(id="page-tab1", children=tab1_layout(v0, eng0), style={"display": "block"}),
    html.Div(id="page-tab2", children=tab2_layout(v0),       style={"display": "none"}),
    html.Div(id="page-tab3", children=tab3_layout(v0, eng0), style={"display": "none"}),
    html.Div(id="page-tab4", children=tab4_layout(v0),       style={"display": "none"}),
])


# ---------- Show/Hide pages ----------

@callback(

    Output("page-tab1", "style"),

    Output("page-tab2", "style"),

    Output("page-tab3", "style"),

    Output("page-tab4", "style"),

    Input("tabs", "value"),

)
def _toggle_pages(active):

    styles = [{"display": "none"} for _ in range(4)]

    idx = {"tab1": 0, "tab2": 1, "tab3": 2, "tab4": 3}.get(active, 0)

    styles[idx] = {"display": "block"}

    return tuple(styles)


# ---------- Utils ----------

def dash_ctx_trigger():
    try:
        triggered = ctx.triggered_id
    except Exception:
        triggered = None
    if triggered:
        if isinstance(triggered, dict):
            return triggered.get("id", "")
        return str(triggered)
    trig = getattr(ctx, "triggered", None) or []
    if trig:
        prop = trig[0].get("prop_id", "")
        if prop:
            return prop.split(".")[0]
    return ""


def _arr(a): return np.asarray(a, dtype=float)



# ---------- Apply / Reset (Tab 1) ----------

@callback(

    Output("store-vehicle", "data"),

    Output("derived-label", "children"),

    Output("fe-show-gears", "options"),

    Output("fe-show-gears", "value"),

    Input("btn-apply", "n_clicks"),

    Input("btn-reset", "n_clicks"),

    State("mass", "value"), State("cd", "value"), State("af", "value"),

    State("rho_air", "value"), State("rho_fuel", "value"), State("lhv_mjpkg", "value"),

    State("g", "value"), State("eta_dl", "value"),

    State("tire_w", "value"), State("tire_ar", "value"), State("tire_rim_in", "value"),

    State("fd", "value"), State("gear_ratios", "value"),

    State("idle_rpm", "value"), State("redline_rpm", "value"),

    prevent_initial_call=True

)

def apply_or_reset(n_apply, n_reset, m, Cd, Af, rho_air, rho_fuel, lhv_mjpkg, g, eta_dl,

                   tire_w, tire_ar, tire_rim_in, fd, gear_ratios, idle_rpm, redline_rpm):

    ctx_id = dash_ctx_trigger()
    if not ctx_id:
        n_apply = n_apply or 0
        n_reset = n_reset or 0
        if n_reset > n_apply:
            ctx_id = "btn-reset"
        elif n_apply > n_reset:
            ctx_id = "btn-apply"
    if ctx_id == "btn-reset":

        v = default_vehicle()

    elif ctx_id == "btn-apply":

        try:

            gears = [float(x) for x in str(gear_ratios).strip().split()]

            assert all(g > 0 for g in gears)

        except Exception:

            gears = default_vehicle()["gears"]

        v = {

            "m": float(m), "Cd": float(Cd), "Af": float(Af),

            "rho_air": float(rho_air), "rho_fuel": float(rho_fuel),

            "LHV": float(lhv_mjpkg) * 1e6, "g": float(g), "eta_dl": float(eta_dl),

            "tire_w": float(tire_w), "tire_ar": float(tire_ar), "tire_rim_in": float(tire_rim_in),

            "fd": float(fd), "gears": gears, "idle_rpm": float(idle_rpm), "redline_rpm": float(redline_rpm)

        }

        v["re"] = C.compute_tire_radius(v)

        v = C.update_derived(v)

    else:

        raise PreventUpdate


    label = f"re = {v['re']:.3f} m | top-gear @ redline ≈ {v['top_speed_kmh']:.1f} km/h"

    opts = [{"label": f"G{i+1}", "value": i+1} for i in range(len(v["gears"]))]

    return v, label, opts, [len(v["gears"])] if v["gears"] else []

@callback(

    Output("mass", "value"),

    Output("cd", "value"),

    Output("af", "value"),

    Output("rho_air", "value"),

    Output("rho_fuel", "value"),

    Output("lhv_mjpkg", "value"),

    Output("g", "value"),

    Output("eta_dl", "value"),

    Output("tire_w", "value"),

    Output("tire_ar", "value"),

    Output("tire_rim_in", "value"),

    Output("fd", "value"),

    Output("gear_ratios", "value"),

    Output("idle_rpm", "value"),

    Output("redline_rpm", "value"),

    Input("store-vehicle", "data"),

    prevent_initial_call=True

)

def sync_vehicle_inputs(v):

    if not v:

        raise PreventUpdate

    ratios = " ".join(f"{g:g}" for g in v.get("gears", []))

    return (

        float(v.get("m", 0.0)),

        float(v.get("Cd", 0.0)),

        float(v.get("Af", 0.0)),

        float(v.get("rho_air", 0.0)),

        float(v.get("rho_fuel", 0.0)),

        float(v.get("LHV", 0.0)) / 1e6,

        float(v.get("g", 0.0)),

        float(v.get("eta_dl", 0.0)),

        float(v.get("tire_w", 0.0)),

        float(v.get("tire_ar", 0.0)),

        float(v.get("tire_rim_in", 0.0)),

        float(v.get("fd", 0.0)),

        ratios,

        float(v.get("idle_rpm", 0.0)),

        float(v.get("redline_rpm", 0.0))

    )




# ---------- Engine dropdown / periodic rescan ----------

@callback(

    Output("store-engine", "data"),

    Output("engine-select", "options"),

    Output("engine-select", "value"),

    Input("engine-select", "value"),

    Input("iv-scan", "n_intervals"),

    State("store-engine", "data"),

    prevent_initial_call=True

)

def engine_picker(selected_name, _tick, eng_state):

    if eng_state is None:

        eng_state = initial_engine_state()


    # periodic folder rescan (does not change active)

    try:

        scanned = EIO.scan_engines_folder(ENG_DIR)

        known = {e["name"] for e in eng_state["items"]}

        for e in scanned:

            if e["name"] not in known:

                eng_state["items"].append(e)

                eng_state["names"].append(e["name"])

                known.add(e["name"])

    except Exception as ex:

        print("Scan failed:", ex)


    # selection

    names = eng_state.get("names", [])

    if selected_name in names:

        eng_state["active"] = names.index(selected_name)


    opts = [{"label": n, "value": n} for n in eng_state.get("names", [])]

    value = eng_state["names"][eng_state["active"]] if eng_state.get("names") else None

    return eng_state, opts, value



# ---------- Rebuild maps whenever vehicle or engine changes ----------

@callback(

    Output("store-maps", "data"),

    Input("store-vehicle", "data"),

    Input("store-engine", "data"),

)

def rebuild_maps_store(v, eng):

    if not v or not eng:

        raise PreventUpdate

    return maps_from_state(v, eng)



# ---------- Tab 2: gearing & accel ----------

@callback(

    Output("ax-distance-time", "figure"),

    Output("ax-speed-time-gear", "figure"),

    Output("ax-speedrpm", "figure"),

    Output("ax-torquespeed", "figure"),

    Output("ax-accelspeed", "figure"),

    Output("ax-accelbottom", "figure"),

    Input("tabs", "value"),

    Input("store-vehicle", "data"),

    Input("store-maps", "data"),

    Input("btn-update-gear", "n_clicks"),

    Input("loads", "value"),

)

def update_gearing(active_tab, v, M, _nclicks, loads_txt):

    if active_tab != "tab2":

        return no_update, no_update, no_update, no_update, no_update, no_update

    if not v or not M:

        raise PreventUpdate



    bsfc_fn, T_WOT, T_CT = C.interpolators(M)

    nG = len(v["gears"])

    if nG == 0:

        raise PreventUpdate



    gear_colors = [f"hsl({int(360*i/nG)},60%,60%)" for i in range(nG)]

    styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

    Krpm2kmh = (2*np.pi*v["re"]/60.0) * 3.6 / v["fd"]



    rpm_grid = np.linspace(0, v["redline_rpm"], 400)

    fig_speedrpm = go.Figure()

    for gi in range(nG):

        v_kmh = (rpm_grid * Krpm2kmh) / v["gears"][gi]

        fig_speedrpm.add_trace(go.Scatter(

            x=v_kmh,

            y=rpm_grid,

            mode="lines",

            name=f"G{gi+1} ({v['gears'][gi]:.2f})",

            line=dict(width=2, color=gear_colors[gi])

        ))

    fig_speedrpm.add_hline(y=v["redline_rpm"], line=dict(color="#ff5d5d", dash="dash"), annotation_text="Redline")

    apply_dark(fig_speedrpm, title="Gear capability: vehicle speed vs engine RPM")



    rpm_grid2 = np.linspace(max(1, v["idle_rpm"]), v["redline_rpm"], 420)

    w_grid = rpm_grid2 * 2*np.pi/60

    Twot = np.maximum(0, T_WOT(w_grid))

    Tmot = T_CT(w_grid)



    try:

        loads = [float(x) for x in str(loads_txt).strip().split()]

    except Exception:

        loads = [0.2, 0.4, 0.6, 0.8, 1.0]

    loads = [min(1.0, max(0.0, v_)) for v_ in loads]



    fig_torque = go.Figure()

    fig_accel = go.Figure()

    fig_force = go.Figure()

    fig_speedtime = go.Figure()

    fig_dist = go.Figure()



    def cumtrap(x, y):

        x = np.asarray(x, dtype=float)

        y = np.asarray(y, dtype=float)

        if x.size < 2:

            return np.zeros_like(x)

        dx = np.diff(x)

        avg = 0.5 * (y[1:] + y[:-1])

        return np.concatenate(([0.0], np.cumsum(avg * dx)))



    for li, lam in enumerate(loads):

        Tvec = np.maximum(0, Tmot + lam*(Twot - Tmot))

        style = styles[li % len(styles)]

        for gi in range(nG):

            gear = v["gears"][gi]

            v_mps = ((rpm_grid2 * 2*np.pi/60) * v["re"]) / (gear * v["fd"])

            v_kmh = v_mps * 3.6



            fig_torque.add_trace(go.Scatter(

                x=v_kmh,

                y=Tvec,

                mode="lines",

                line=dict(width=1.6, color=gear_colors[gi], dash=style),

                showlegend=(gi == 0),

                name=f"λ={lam:.2f}"

            ))



            F_wheel = Tvec * (gear * v["fd"] * v["eta_dl"]) / v["re"]

            R = C.F_roll(v, v_mps) + C.F_aero(v, v_mps)

            a_mps2 = (F_wheel - R) / v["m"]

            a_g = a_mps2 / v["g"]

            a_plot = a_g.copy()

            a_plot[a_plot < 0] = np.nan



            fig_accel.add_trace(go.Scatter(

                x=v_kmh,

                y=a_plot,

                mode="lines",

                line=dict(width=1.6, color=gear_colors[gi], dash=style),

                showlegend=(gi == 0),

                name=f"λ={lam:.2f}"

            ))



            valid = np.isfinite(a_mps2) & (a_mps2 > 1e-3)

            if np.count_nonzero(valid) < 2:

                continue



            v_mps_valid = v_mps[valid]

            v_kmh_valid = v_kmh[valid]

            a_valid = a_mps2[valid]

            t = cumtrap(v_mps_valid, 1.0 / a_valid)

            dist = cumtrap(t, v_mps_valid)



            fig_speedtime.add_trace(go.Scatter(

                x=t,

                y=v_kmh_valid,

                mode="lines",

                line=dict(width=1.6, color=gear_colors[gi], dash=style),

                showlegend=False,

                legendgroup=f"λ{li}",

                name=f"λ={lam:.2f}"

            ))



            fig_dist.add_trace(go.Scatter(

                x=t,

                y=dist,

                mode="lines",

                line=dict(width=1.6, color=gear_colors[gi], dash=style),

                showlegend=(gi == 0),

                legendgroup=f"λ{li}",

                name=f"λ={lam:.2f}"

            ))



    apply_dark(fig_torque, title="Engine torque vs vehicle speed (per gear) at normalized loads")

    apply_dark(fig_accel, title="Acceleration vs vehicle speed (per gear) at selected loads")

    apply_dark(fig_force, title="Available tractive force (λ=1) and resistance vs speed")

    apply_dark(fig_speedtime, title="Vehicle speed vs time (per gear at selected loads)")

    apply_dark(fig_dist, title="Vehicle distance vs time (per gear at selected loads)")



    fig_speedtime.update_xaxes(title="Time (s)")

    fig_speedtime.update_yaxes(title="Vehicle speed (km/h)")

    fig_dist.update_xaxes(title="Time (s)")

    fig_dist.update_yaxes(title="Distance (m)")



    v_all_kmh = np.linspace(0, v["top_speed_kmh"], 600)

    v_all_mps = v_all_kmh / 3.6

    R_all = C.F_roll(v, v_all_mps) + C.F_aero(v, v_all_mps)

    fig_force.add_trace(go.Scatter(

        x=v_all_kmh,

        y=R_all,

        name="Resistance",

        line=dict(color="#d0d3d8", width=2)

    ))



    Tvec_WOT = np.maximum(0, Twot)

    for gi in range(nG):

        v_mps = ((rpm_grid2 * 2*np.pi/60) * v["re"]) / (v["gears"][gi] * v["fd"])

        v_kmh = v_mps * 3.6

        F_wheel = Tvec_WOT * (v["gears"][gi] * v["fd"] * v["eta_dl"]) / v["re"]

        fig_force.add_trace(go.Scatter(

            x=v_kmh,

            y=F_wheel,

            name=f"G{gi+1}",

            line=dict(width=1.6, color=gear_colors[gi])

        ))



    return fig_dist, fig_speedtime, fig_speedrpm, fig_torque, fig_accel, fig_force



# ---------- Tab 3: engine map + sequences ----------

@callback(

    Output("ax-map", "figure"),

    Output("ax-speed-time3", "figure"),

    Output("ax-fuel-time", "figure"),

    Input("tabs", "value"),

    Input("store-maps", "data"),

    Input("store-sequences", "data"),

    Input("map-mode", "value"),

    Input("show-cruise", "value"),

    Input("cruise-gear", "value"),

    Input("cruise-label-step", "value"),

    State("store-vehicle", "data"),

)

def update_map_and_time(active_tab, M, seqs, map_mode, show_cruise, cruise_gear, label_step, v):

    if active_tab != "tab3":

        return no_update, no_update, no_update

    if not M:

        raise PreventUpdate


    torque = _arr(M["torque_Nm"])

    w = _arr(M["speed_radps"])

    rpm_map = w * 60.0 / (2*np.pi)

    Zeta  = np.array(M.get("eta_map"), dtype=float) if ("eta_map" in M) else None

    Zbsfc = _arr(M["bsfc_map_gpkWh"])


    fig = go.Figure()

    if (map_mode == "eta") and (Zeta is not None):

        Z = Zeta * 100.0

        fig.add_contour(x=rpm_map, y=torque, z=Z,

                        contours=dict(coloring="heatmap", showlines=False),

                        colorbar=dict(title="Efficiency (%)", len=0.6, y=0.75))

        fig.add_contour(x=rpm_map, y=torque, z=Z,

                        contours=dict(coloring="none", showlines=True, start=10, end=45, size=1, showlabels=False),

                        line=dict(color="rgba(230,230,230,0.25)", width=1),

                        showscale=False)

        title = "Engine map — Efficiency"

    else:

        Z = Zbsfc

        fig.add_contour(x=rpm_map, y=torque, z=Z,

                        contours=dict(coloring="heatmap", showlines=False),

                        colorbar=dict(title="BSFC (g/kWh)", len=0.6, y=0.75))

        zmin = np.nanmin(Z); zmax = np.nanmax(Z)

        if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:

            fig.add_contour(x=rpm_map, y=torque, z=Z,

                            contours=dict(coloring="none", showlines=True,

                                          start=zmin+10, end=zmax-10, size=10, showlabels=False),

                            line=dict(color="rgba(230,230,230,0.25)", width=1),

                            showscale=False)

        title = "Engine map — BSFC"


    fig.add_trace(go.Scatter(x=_arr(M["wot_speed_radps"]) * 60/(2*np.pi),

                             y=_arr(M["wot_torque_Nm"]),

                             mode="lines", line=dict(color="#ff5d5d", width=2), name="WOT"))

    fig.add_trace(go.Scatter(x=_arr(M["ct_speed_radps"]) * 60/(2*np.pi),

                             y=_arr(M["ct_torque_Nm"]),

                             mode="lines", line=dict(color="#c8ccd1", width=1.5, dash="dash"),

                             name="Closed throttle"))

    fig.update_layout(legend=dict(y=0.02, yanchor="bottom", x=1.02, xanchor="left"))

    fig.update_xaxes(title="Engine speed (RPM)")

    fig.update_yaxes(title="Torque (N·m)")

    apply_dark(fig, height=560, title=title)


    # Cruise overlay with labels

    if show_cruise and 1 in show_cruise and cruise_gear:

        rpm, Treq, v_kmh = C.steady_cruise_curve(v, M, int(cruise_gear) - 1)

        ok = np.isfinite(rpm) & np.isfinite(Treq)

        if ok.any():

            fig.add_trace(go.Scatter(x=rpm[ok], y=Treq[ok], mode="lines",

                                     line=dict(color="white", width=2), name=f"Cruise G{cruise_gear}"))

            step = max(1, int(label_step or 10))

            vk = v_kmh[ok]

            if vk.size:

                marks = np.arange(step*np.ceil(vk.min()/step), step*np.floor(vk.max()/step)+1e-9, step)

                annotations = []

                for s in marks:

                    j = np.nanargmin(np.abs(v_kmh - s))

                    if np.isfinite(Treq[j]) and np.isfinite(rpm[j]):

                        annotations.append(dict(

                            x=float(rpm[j]), y=float(Treq[j]),

                            xref="x", yref="y", text=f" {int(round(v_kmh[j]))}",

                            showarrow=False, font=dict(color="white", size=10)

                        ))

                fig.update_layout(annotations=annotations)


    # Time panes + sequences + shift markers

    fig_v = apply_dark(go.Figure(), height=270, title="Vehicle speed vs time")

    fig_f = apply_dark(go.Figure(), height=270, title="Cumulative fuel vs time")

    seen = set()

    for i, S in enumerate(seqs or []):

        nm = S.get("name", f"seq{i+1}")

        if nm in seen:

            continue

        seen.add(nm)

        col = f"hsl({int(360*(i%12)/12)},70%,60%)"

        Tt = S.get("trace", {})

        rp = np.asarray(Tt.get("rpm", [])); Te = np.asarray(Tt.get("T_engine_Nm", []))

        if rp.size and Te.size:

            fig.add_trace(go.Scatter(x=rp, y=Te, mode="lines", line=dict(color=col, width=2),

                                     name=nm, legendgroup=nm))

            gear = np.asarray(Tt.get("gear", []))

            if gear.size:

                change = np.nonzero(np.r_[False, np.diff(gear)!=0])[0]

                fig.add_trace(go.Scatter(x=rp[change], y=Te[change], mode="markers",

                                         marker=dict(symbol="square", size=8, color=col, line=dict(color="white", width=1)),

                                         showlegend=False, legendgroup=nm))

        ts = np.asarray(Tt.get("t_s", []))

        vk = np.asarray(Tt.get("v_kmh", []))

        fk = np.asarray(Tt.get("fuel_cum_L", []))

        fig_v.add_trace(go.Scatter(x=ts, y=vk, mode="lines", line=dict(color=col, width=2), name=nm, legendgroup=nm))

        fig_f.add_trace(go.Scatter(x=ts, y=fk, mode="lines", line=dict(color=col, width=2), name=nm, legendgroup=nm))

        gear = np.asarray(Tt.get("gear", []))

        if gear.size:

            change = np.nonzero(np.r_[False, np.diff(gear)!=0])[0]

            fig_v.add_trace(go.Scatter(x=ts[change], y=vk[change], mode="markers",

                                       marker=dict(symbol="circle", size=7, color=col, line=dict(color="white", width=1)),

                                       showlegend=False, legendgroup=nm))

            fig_f.add_trace(go.Scatter(x=ts[change], y=fk[change], mode="markers",

                                       marker=dict(symbol="circle", size=7, color=col, line=dict(color="white", width=1)),

                                       showlegend=False, legendgroup=nm))


    return fig, fig_v, fig_f



# ---------- Add / clear sequences ----------

@callback(

    Output("store-sequences", "data"),

    Input("btn-add-seq", "n_clicks"),

    Input("btn-clear-seq", "n_clicks"),

    State("shift-speeds", "value"),

    State("lambda", "value"),

    State("target-kmh", "value"),

    State("force-redline", "value"),

    State("store-vehicle", "data"),

    State("store-maps", "data"),

    State("store-sequences", "data"),

    State("tabs", "value"),

    prevent_initial_call=True

)

def sequences(add, clear, shift_txt, lam, target, force_redline, v, M, seqs, active_tab):

    if active_tab != "tab3":

        raise PreventUpdate

    which = dash_ctx_trigger()

    if which == "btn-clear-seq":

        return []

    if which != "btn-add-seq":

        raise PreventUpdate

    if not v or not M:

        raise PreventUpdate


    try:

        shift_kmh = [float(x) for x in str(shift_txt).split()]

    except Exception:

        shift_kmh = []


    lam = float(lam or 0.7)

    opts = dict(

        rho_fuel_kg_per_L=v["rho_fuel"],

        idle_rpm=v["idle_rpm"], redline_rpm=v["redline_rpm"],

        dv_kmh=0.25, verbose=False,

        force_redline=(1 in (force_redline or [])),

    )

    R = C.accel_fuel_to_speed_core(v, M, float(target or 120.0), shift_kmh, lam, opts)


    idx = len(seqs or []) + 1

    entry = {

        "name": f"seq{idx}",

        "lambda": lam,

        "shift_kmh": shift_kmh,

        "target": target,

        "R": {k: (float(v) if np.isscalar(v) else v) for k, v in R.items() if k in ["reached_target","final_speed_kmh","time_s","fuel_L","distance_m"]},

        "trace": {k: (np.asarray(v).tolist()) for k, v in R.get("trace", {}).items()}

    }

    return (seqs or []) + [entry]



# ---------- FE (Tab 4) ----------

@callback(

    Output("ax-fe", "figure"),

    Input("tabs", "value"),

    Input("fe-label-step", "value"),

    Input("fe-show-gears", "value"),

    State("store-vehicle", "data"),

    State("store-maps", "data"),

)

def fe_plot(active_tab, lbl_step, show_gears, v, M):

    if active_tab != "tab4":

        return no_update

    if not v or not M:

        raise PreventUpdate


    nG = len(v["gears"])

    if not show_gears:

        show_gears = [nG]


    fig = go.Figure()

    lbl_step = int(lbl_step or 10)

    for gi in range(nG):

        if (gi+1) not in show_gears:

            continue

        Krpm2kmh = (2*np.pi*v["re"]/60.0) * 3.6 / v["fd"]

        v_idle = max(0, (v["idle_rpm"] * Krpm2kmh)/v["gears"][gi])

        v_redl = (v["redline_rpm"] * Krpm2kmh)/v["gears"][gi]

        v_kmh = np.linspace(v_idle, v_redl, 160)

        L100, rpm, Treq = C.steady_FE_in_gear(v, M, gi, v_kmh)

        fig.add_trace(go.Scatter(x=v_kmh, y=L100, mode="lines", name=f"G{gi+1}", line=dict(width=2)))


        mask = np.isfinite(L100)

        if mask.any():

            j = np.nanargmin(L100[mask])

            xv = v_kmh[mask][j]; yv = L100[mask][j]

            fig.add_trace(go.Scatter(x=[xv], y=[yv], mode="markers",

                                     marker=dict(color="#ffd54a", line=dict(color="black", width=1.5), size=10),

                                     showlegend=False))

            lbs = np.arange(lbl_step*np.ceil(v_idle/lbl_step), lbl_step*np.floor(v_redl/lbl_step)+1e-9, lbl_step)

            for s in lbs:

                jj = np.nanargmin(np.abs(v_kmh - s))

                if np.isfinite(L100[jj]):

                    fig.add_trace(go.Scatter(x=[v_kmh[jj]], y=[L100[jj]], mode="text",

                                             text=[f" {int(round(v_kmh[jj]))}"], showlegend=False,

                                             textfont=dict(color=DARK_MUTED, size=10)))

    apply_dark(fig, height=720, title="Steady-state fuel consumption per gear")

    return fig



# ---------- Cruise-gear options (Tab 3) ----------

@callback(

    Output("cruise-gear", "options"),

    Output("cruise-gear", "value"),

    Input("store-vehicle", "data"),

)

def cruise_gear_options(v):

    nG = len(v.get("gears", [])) if v else 0

    opts = [{"label": str(i+1), "value": i+1} for i in range(nG)]

    val = nG or None

    return opts, val



# ---------- Run ----------

if __name__ == "__main__":

    app.run(debug=True, host="0.0.0.0", port=25560)

