# Vehicle Efficiency Web App (Dash)

A self-hosted, MATLAB-free web application that mirrors your **VehicleEfficiencyApp**:
- 4 tabs: Vehicle & Simulation, Gearing & Accel, Maps & Sequences, Steady-State FE
- Engine map import: `.m` (EPA/REVS-style) and `.mat` (struct with `engine.*`), plus JSON export
- Contour maps (efficiency / BSFC), cruise overlays, shift-sequence integration (accel & fuel), per-gear FE
- Works on any device via a browser

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8050

## Engine files

- Upload `.m` files in the EPA/REVS style (see `engines/example_engine.engine.json` for fields)
- Upload `.mat` files that contain a struct named `engine` (or a single top-level struct) with fields:
  - `fuel_map_speed_radps` (N2)
  - `fuel_map_torque_Nm` (N1)
  - `fuel_map_gps` (N1 x N2)
  - `full_throttle_speed_radps`, `full_throttle_torque_Nm`
  - `closed_throttle_*` **or** `naturally_aspirated_*` (CT = -abs(NA))
- Convert `.m`/`.mat` to portable JSON via **Convert to JSON** in the Engine section.

## Notes

- The UI layout mirrors the MATLAB app as closely as practical in a browser: same tabs, same controls/labels.
- All computations are reimplemented in NumPy/SciPy to match MATLAB behavior.
- If your `.mat` files require HDF5 v7.3 features, ensure your SciPy install supports them.
