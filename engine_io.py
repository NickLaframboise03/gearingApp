import base64

import json

import re

from pathlib import Path

from typing import Dict, List, Union, Any


import numpy as np

import tc


# Optional .mat support

try:

    from scipy.io import loadmat, savemat

    HAS_MAT = True

except Exception:

    HAS_MAT = False


DEFAULT_ENGINE_FILENAME = "2014_Mazda_2_0L_SKYACTIV_Engine_Tier_2_Fuel.engine.json"


# -------- Built-in engine (EPA Mazda 2.0L) --------

def load_builtin_engine() -> Dict:

    default_path = Path(__file__).resolve().parent / "engines" / DEFAULT_ENGINE_FILENAME

    if not default_path.exists():

        raise FileNotFoundError(f"Missing built-in engine at {default_path}")

    return load_engine_json(default_path)



# -------- Parse engines from files --------

def scan_engines_folder(folder: Union[str, Path]) -> List[Dict]:

    folder = Path(folder)

    if not folder.exists():

        return []

    engines: List[Dict] = []

    seen: set[str] = set()

    def add_engine(E: Dict):

        name = str(E.get("name", "")).strip()

        if not name:

            return

        if name in seen:

            return

        engines.append(E)

        seen.add(name)

    for p in sorted(folder.glob("*.json")):

        try:

            add_engine(load_engine_json(p))

        except Exception as ex:

            print(f"[engine_io] Skipping {p.name}: {ex}")

    for p in sorted(folder.glob("*.m")):

        try:

            e = parse_engine_m_file(p.read_text(encoding="utf-8", errors="ignore"), p.name)

            save_engine_json(folder, e)

            add_engine(e)

        except Exception as ex:

            print(f"[engine_io] Skipping {p.name}: {ex}")

    if HAS_MAT:

        for p in sorted(folder.glob("*.mat")):

            try:

                e = load_engine_mat(p)

                save_engine_json(folder, e)

                add_engine(e)

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

        raw = re.sub(r"[;,\s]+", " ", mm.group(1).strip())

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



# Helper to normalize optional torque-converter data
def _normalize_converter_block(conv: Any) -> Dict[str, Any]:
    base = tc.DEFAULT_TC_CURVES
    conv = conv or {}
    def arr(key: str) -> np.ndarray:
        src = conv.get(key, base[key])
        return np.asarray(src, dtype=float).ravel()
    return dict(
        speed_ratio=arr("speed_ratio"),
        torque_ratio=arr("torque_ratio"),
        k_norm=arr("k_norm"),
        lockup_sr=float(conv.get("lockup_sr", base.get("lockup_sr", 0.92))),
    )

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

    converter = _normalize_converter_block(E.get("converter"))

    return dict(

        name=str(E.get("name", "engine")),

        fuel_map_speed_radps=sp,

        fuel_map_torque_Nm=tq,

        fuel_map_gps=gps,

        full_throttle_speed_radps=np.asarray(E["full_throttle_speed_radps"], dtype=float).ravel(),

        full_throttle_torque_Nm=np.asarray(E["full_throttle_torque_Nm"], dtype=float).ravel(),

        closed_throttle_speed_radps=np.asarray(E["closed_throttle_speed_radps"], dtype=float).ravel(),

        closed_throttle_torque_Nm=np.asarray(E["closed_throttle_torque_Nm"], dtype=float).ravel(),

        converter=converter,

    )



def _to_jsonable(E: Dict) -> Dict:

    def convert(val):

        if isinstance(val, np.ndarray):

            return val.tolist()

        if isinstance(val, dict):

            return {kk: convert(vv) for kk, vv in val.items()}

        return val

    out = {}

    for k, v in E.items():

        out[k] = convert(v)

    return out

