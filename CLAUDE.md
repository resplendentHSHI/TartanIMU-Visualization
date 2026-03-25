# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IMU processing pipeline for aligning inertial measurement unit (IMU) data with ground-truth trajectories, plus real-time 3D visualization. Focused on Lamar dataset sequences and helmet-mounted IMU pose estimation.

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install numpy pandas scipy matplotlib
git clone https://github.com/GaTech-RL2/EgoVerse.git  # dependency for visualizers
```

No build system, test suite, or linter is configured.

## Scripts

- `alignment_original.py` — Original single-sequence alignment (hardcoded paths at top). Reads IMU+GT CSVs, gravity-aligns, interpolates to 200 Hz, outputs CSVs + NPZ.
- `alignment_library.py` — Refactored alignment as importable library. Accepts DataFrames + `timestamp_unit` param. Used by `lamar-v2-processing.py`.
- `lamar-v2-processing.py` — Batch Lamar processor. Walks dirs for `fused_imu.txt` + `trajectories.txt`, combines user accel with gravity vector, converts microsecond timestamps, runs alignment. Edit `BASE_INPUT_DIR`/`BASE_OUTPUT_DIR` at top.
- `visualize_live.py` — Real-time 3D web visualizer (Three.js). Loads NPZ, serves Foxglove-style UI at localhost. Uses EgoVerse `pose_utils` for quaternion→euler.
- `visualize_npz.py` — Static matplotlib dashboard (PNG output). Uses EgoVerse `pose_utils` and `ColorsPalette`.

## NPZ Format

All NPZ files share these keys:
- `retargetted_ts` — (N,) timestamps in seconds
- `retargetted_imu` — (N,6) `[ax, ay, az, gx, gy, gz]` m/s² and rad/s
- `retargetted_pos` — (N,3) `[x, y, z]` meters
- `retargetted_quat` — (N,4) `[qx, qy, qz, qw]` Hamilton XYZW

## EgoVerse Dependency

Visualizers import from `EgoVerse/egomimic/utils/pose_utils.py` (quaternion↔matrix↔euler conversions) and `EgoVerse/egomimic/scripts/plotting/plotting.py` (color palette). EgoVerse is gitignored — clone it locally per setup instructions.

## Column Name Conventions

Alignment scripts handle multiple IMU column naming schemes:
- Accel: `accel_x/y/z`, `ax/ay/az`; gravity-aligned output appends `_ga` suffix
- Gyro: `gyro_x/y/z`, `gx/gy/gz`; gravity-aligned output appends `_ga` suffix
- GT quaternion: `qx,qy,qz,qw` (xyzw) or `w,x,y,z` (wxyz, auto-converted)
- Lamar fused_imu.txt: 14+ columns with user accel, rotation rate, magnetometer, gravity vector

## Sample Data

One sample per dataset is included for testing:
- `Helmet_Poser_Aligned/A_c/` — NPZ + aligned CSVs
- `debug_dataset/car/0/0.npz` — Short car motion sequence
- `lamar-v2-processed-NO-GRAVITY-npz/` — One Lamar CAB sequence
