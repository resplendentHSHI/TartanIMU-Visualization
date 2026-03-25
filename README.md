# IMU Processing & Visualization Pipeline

Utilities to align IMU data with ground-truth trajectories, batch-process Lamar sequences, and visualize the results in a real-time 3D web viewer.

## Scripts

### Alignment
- **`alignment_original.py`** — Original single-sequence alignment script. Reads IMU + GT CSVs, applies gravity alignment, interpolates to 200 Hz, outputs `aligned_imu.csv`, `aligned_gt.csv`, and NPZ.
- **`alignment_library.py`** — Refactored alignment as an importable library (no hardcoded paths). Accepts pre-loaded DataFrames and a `timestamp_unit` parameter.
- **`lamar-v2-processing.py`** — Batch processor for Lamar dataset. Walks directories looking for `fused_imu.txt` + `trajectories.txt`, runs alignment on each, outputs flat NPZ files.

### Visualization
- **`visualize_live.py`** — Real-time 3D Foxglove-style web visualizer. Loads an NPZ and serves an interactive browser UI with animated trajectory, coordinate frames, IMU charts, and playback controls.
- **`visualize_npz.py`** — Static matplotlib dashboard generator. Produces PNG plots of trajectory, IMU, orientation, and gravity check.

## Setup

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Core dependencies
uv pip install numpy pandas scipy matplotlib

# Clone EgoVerse (used by visualizers for pose utilities)
git clone https://github.com/GaTech-RL2/EgoVerse.git
```

## Usage

### Live 3D Visualizer
```bash
python3 visualize_live.py Helmet_Poser_Aligned/A_c/A_c.npz
python3 visualize_live.py debug_dataset/car/0/0.npz --port 9000
```
Opens a browser at `http://localhost:8888` with:
- 3D viewport with trajectory trail and animated pose frame
- View presets (3D / Top / Side / Front)
- Side panels: accelerometer, gyroscope, accel magnitude (gravity check), euler angles
- Playback controls with adjustable speed (0.25x–10x)

### Static Plots
```bash
python3 visualize_npz.py Helmet_Poser_Aligned/A_c/A_c.npz
```

### Alignment
```bash
# Single sequence (edit paths at top of file first)
python3 alignment_original.py

# Batch Lamar (edit BASE_INPUT_DIR / BASE_OUTPUT_DIR first)
python3 lamar-v2-processing.py
```

## NPZ Format

All output NPZ files share these keys:
| Key | Shape | Description |
|-----|-------|-------------|
| `retargetted_ts` | `(N,)` | Timestamps (seconds) |
| `retargetted_imu` | `(N, 6)` | `[ax, ay, az, gx, gy, gz]` — accel (m/s²) + gyro (rad/s) |
| `retargetted_pos` | `(N, 3)` | `[x, y, z]` — position (meters) |
| `retargetted_quat` | `(N, 4)` | `[qx, qy, qz, qw]` — orientation (Hamilton XYZW) |

## Sample Data

The repo includes one sample from each dataset for testing:
- `Helmet_Poser_Aligned/A_c/` — Helmet poser sequence with NPZ + aligned CSVs
- `debug_dataset/car/0/` — Short car motion sequence
- `lamar-v2-processed-NO-GRAVITY-npz/` — One Lamar CAB building sequence (no gravity alignment)
