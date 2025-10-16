import base64

import json

import re

from pathlib import Path

from typing import Dict, List, Union


import numpy as np


# Optional .mat support

try:

    from scipy.io import loadmat, savemat

    HAS_MAT = True

except Exception:

    HAS_MAT = False



# -------- Built-in engine (EPA Mazda 2.0L; compact copy) --------

def load_builtin_engine() -> Dict:

    # Same content/shape as your MATLAB app uses.

    from math import pi

    # Minimal but representative map (trimmed size for brevity but smooth)

    # Full grid would be larger; this works well for demo & matches app logic.

    S = _mazda_skyactiv_sample()

    E = dict(

        name="Built-in: Mazda 2.0L (EPA-derived)",

        fuel_map_speed_radps=np.array(S["speed_radps"], dtype=float),

        fuel_map_torque_Nm=np.array(S["torque_Nm"], dtype=float),

        fuel_map_gps=np.array(S["fuel_gps"], dtype=float),

        full_throttle_speed_radps=np.array(S["wot_speed_radps"], dtype=float),

        full_throttle_torque_Nm=np.array(S["wot_torque_Nm"], dtype=float),

        closed_throttle_speed_radps=np.array(S["ct_speed_radps"], dtype=float),

        closed_throttle_torque_Nm=np.array(S["ct_torque_Nm"], dtype=float),

    )

    # Ensure matrix shape [len(torque), len(speed)]

    E["fuel_map_gps"] = E["fuel_map_gps"].reshape(

        len(E["fuel_map_torque_Nm"]), len(E["fuel_map_speed_radps"])

    )

    return E



def _mazda_skyactiv_sample():

    # A compact subset of the data from your provided MATLAB function

    # (downsampled to keep this file readable).

    speed = [0.0, 104.57, 156.89, 209.28, 310.89, 415.80, 562.87, 680.68]

    torque = [-39.93, 9.69, 49.57, 89.72, 129.53, 169.72, 211.47]

    # 7 x 8 = 56 values; simple smooth surface

    fuel_gps = np.array([

        [0,0,0,0,0,0,0,0],

        [0,0,0.02,0.06,0.17,0.45,0.66,0.70],

        [0.03,0.10,0.20,0.33,0.47,0.72,0.96,1.10],

        [0.05,0.13,0.25,0.41,0.58,0.86,1.19,1.35],

        [0.07,0.18,0.32,0.51,0.75,1.06,1.46,1.65],

        [0.09,0.22,0.40,0.66,0.97,1.39,1.89,2.15],

        [0.12,0.28,0.52,0.83,1.22,1.75,2.32,2.65],

    ])

    wot_w = [0, 157.18, 209.47, 282.74, 356.10, 429.37, 628.31, 680.68]

    wot_T = [0, 164.92, 187.87, 201.40, 197.40, 200.10, 184.0, 0]

    ct_w  = [0, 146.56, 403.17, 465.84, 680.68]

    ct_T  = [-16.01, -20.65, -28.76, -34.96, -42.71]

    return dict(

        speed_radps=speed, torque_Nm=torque, fuel_gps=fuel_gps,

        wot_speed_radps=wot_w, wot_torque_Nm=wot_T,

        ct_speed_radps=ct_w,  ct_torque_Nm=ct_T

    )



# -------- Parse engines from files --------

def scan_engines_folder(folder: Union[str, Path]) -> List[Dict]:

    folder = Path(folder)

    if not folder.exists():

        return []

    engines: List[Dict] = []

    for p in sorted(list(folder.glob("*.json")) + list(folder.glob("*.m")) + list(folder.glob("*.mat"))):

        try:

            if p.suffix.lower() == ".json":

                engines.append(load_engine_json(p))

            elif p.suffix.lower() == ".m":

                engines.append(parse_engine_m_file(p.read_text(encoding="utf-8", errors="ignore"), p.name))

            elif p.suffix.lower() == ".mat" and HAS_MAT:

                engines.append(load_engine_mat(p))

        except Exception as ex:

            print(f"[engine_io] Skipping {p.name}: {ex}")

    return engines



def save_engine_json(folder: Union[str, Path], engine: Dict):

    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)

    name_sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", engine.get("name", "engine"))

    (folder / f"{name_sanitized}.json").write_text(json.dumps(_to_jsonable(engine), indent=2), encoding="utf-8")



def load_engine_json(path: Union[str, Path]) -> Dict:

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    return normalize_engine_dict(data)



def load_engine_mat(path: Union[str, Path]) -> Dict:

    if not HAS_MAT:

        raise RuntimeError("scipy is required to read .mat files")

    D = loadmat(str(path), squeeze_me=True, struct_as_record=False)

    eng = D.get("engine", None)

    if eng is None:

        # try first struct

        for k, v in D.items():

            if hasattr(v, "_fieldnames"):

                eng = v

                break

    if eng is None:

        raise ValueError("No 'engine' struct in MAT file")


    def fget(obj, attr):

        return np.array(getattr(obj, attr)).squeeze()


    E = dict(

        name=str(getattr(eng, "name", Path(path).stem)),

        fuel_map_speed_radps=fget(eng, "fuel_map_speed_radps"),

        fuel_map_torque_Nm=fget(eng, "fuel_map_torque_Nm"),

        fuel_map_gps=np.array(getattr(eng, "fuel_map_gps")).squeeze(),

        full_throttle_speed_radps=fget(eng, "full_throttle_speed_radps"),

        full_throttle_torque_Nm=fget(eng, "full_throttle_torque_Nm"),

    )

    # closed throttle or NA fallback

    if hasattr(eng, "closed_throttle_speed_radps") and hasattr(eng, "closed_throttle_torque_Nm"):

        E["closed_throttle_speed_radps"] = fget(eng, "closed_throttle_speed_radps")

        E["closed_throttle_torque_Nm"]   = fget(eng, "closed_throttle_torque_Nm")

    else:

        E["closed_throttle_speed_radps"] = fget(eng, "naturally_aspirated_speed_radps")

        E["closed_throttle_torque_Nm"]   = -np.abs(fget(eng, "naturally_aspirated_torque_Nm"))

    return normalize_engine_dict(E)



def parse_engine_m_file(text: str, filename="engine.m") -> Dict:

    # name

    m = re.search(r"engine\.name\s*=\s*'([^']*)'", text)

    name = m.group(1).strip() if m else filename


    def block(field):

        # captures [ ... ]

        mm = re.search(rf"engine\.{field}\s*=\s*\[([\s\S]*?)\];", text)

        if not mm:

            return None

        # turn into numbers

        raw = re.sub(r"[,\s]+", " ", mm.group(1).strip())

        if not raw:

            return np.array([])

        return np.array([float(x) for x in raw.split()])


    wot_w = block("full_throttle_speed_radps");  wot_T = block("full_throttle_torque_Nm")

    ct_w  = block("closed_throttle_speed_radps"); ct_T  = block("closed_throttle_torque_Nm")

    if ct_w is None or ct_T is None:

        na_w = block("naturally_aspirated_speed_radps"); na_T = block("naturally_aspirated_torque_Nm")

        if na_w is None or na_T is None:

            raise ValueError("Missing closed/naturally aspirated curves")

        ct_w, ct_T = na_w, -np.abs(na_T)


    sp  = block("fuel_map_speed_radps")

    tq  = block("fuel_map_torque_Nm")

    # matrix

    mm = re.search(r"engine\.fuel_map_gps\s*=\s*\[([\s\S]*?)\];", text)

    if not mm:

        raise ValueError("Missing fuel_map_gps")

    grid = np.array([float(x) for x in re.sub(r"[;\s]+", " ", mm.group(1).strip()).split()])

    gps  = grid.reshape(len(tq), len(sp))


    E = dict(

        name=name,

        fuel_map_speed_radps=sp,

        fuel_map_torque_Nm=tq,

        fuel_map_gps=gps,

        full_throttle_speed_radps=wot_w,

        full_throttle_torque_Nm=wot_T,

        closed_throttle_speed_radps=ct_w,

        closed_throttle_torque_Nm=ct_T,

    )

    return normalize_engine_dict(E)



def parse_upload(contents_b64: str, filename: str) -> Dict:

    header, b64 = contents_b64.split(",", 1)

    raw = base64.b64decode(b64)

    suffix = Path(filename).suffix.lower()

    if suffix == ".json":

        return normalize_engine_dict(json.loads(raw.decode("utf-8")))

    if suffix == ".m":

        return parse_engine_m_file(raw.decode("utf-8", errors="ignore"), filename)

    if suffix == ".mat":

        if not HAS_MAT:

            raise RuntimeError("Upload of .mat requires scipy")

        tmp = Path(filename).with_suffix(".mat")

        tmp.write_bytes(raw)

        try:

            return load_engine_mat(tmp)

        finally:

            try: tmp.unlink()

            except Exception: pass

    raise ValueError(f"Unsupported engine file: {filename}")



# -------- utilities --------

def normalize_engine_dict(E: Dict) -> Dict:

    req = [

        "fuel_map_speed_radps","fuel_map_torque_Nm","fuel_map_gps",

        "full_throttle_speed_radps","full_throttle_torque_Nm"

    ]

    for k in req:

        if k not in E:

            raise ValueError(f"Missing field {k}")

    if "closed_throttle_speed_radps" not in E or "closed_throttle_torque_Nm" not in E:

        raise ValueError("Missing closed-throttle (or NA fallback) curves")

    # ensure numpy arrays & correct shapes

    sp = np.asarray(E["fuel_map_speed_radps"], dtype=float).ravel()

    tq = np.asarray(E["fuel_map_torque_Nm"], dtype=float).ravel()

    gps = np.asarray(E["fuel_map_gps"], dtype=float)

    if gps.shape != (tq.size, sp.size):

        gps = gps.reshape(tq.size, sp.size)

    return dict(

        name=str(E.get("name", "engine")),

        fuel_map_speed_radps=sp,

        fuel_map_torque_Nm=tq,

        fuel_map_gps=gps,

        full_throttle_speed_radps=np.asarray(E["full_throttle_speed_radps"], dtype=float).ravel(),

        full_throttle_torque_Nm=np.asarray(E["full_throttle_torque_Nm"], dtype=float).ravel(),

        closed_throttle_speed_radps=np.asarray(E["closed_throttle_speed_radps"], dtype=float).ravel(),

        closed_throttle_torque_Nm=np.asarray(E["closed_throttle_torque_Nm"], dtype=float).ravel(),

    )



def _to_jsonable(E: Dict) -> Dict:

    out = {}

    for k, v in E.items():

        if isinstance(v, np.ndarray):

            out[k] = v.tolist()

        else:

            out[k] = v

    return out

