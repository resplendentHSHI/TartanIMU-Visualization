#!/usr/bin/env python3
"""
Visualize NPZ data produced by alignment_original.py using EgoVerse utilities.

Usage:
    python visualize_npz.py <npz_file> [--output_dir <dir>]
    python visualize_npz.py Helmet_Poser_Aligned/A_c/A_c.npz
    python visualize_npz.py lamar-v2-processed-NO-GRAVITY-npz/CAB_ios_2021-06-02_14.21.49_temp_CAB_ios_2021-06-02_14.21.49.npz
    python visualize_npz.py debug_dataset/car/0/0.npz
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add EgoVerse to path for pose_utils
sys.path.insert(0, str(Path(__file__).resolve().parent / "EgoVerse"))
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix, _matrix_to_xyzypr, xyzw_to_wxyz
from egomimic.scripts.plotting.plotting import ColorsPalette


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    ts = data["retargetted_ts"]
    imu = data["retargetted_imu"]    # Nx6: ax,ay,az,gx,gy,gz
    pos = data["retargetted_pos"]    # Nx3: x,y,z
    quat = data["retargetted_quat"]  # Nx4: qx,qy,qz,qw (XYZW)
    return ts, imu, pos, quat


def quat_to_euler(quat_xyzw):
    """Convert XYZW quaternions to yaw/pitch/roll using EgoVerse pose_utils."""
    wxyz = xyzw_to_wxyz(quat_xyzw)  # -> WXYZ
    # Build 4x4 SE(3) with identity translation for rotation extraction
    xyzwxyz = np.hstack([np.zeros((len(wxyz), 3)), wxyz])  # [0,0,0, qw,qx,qy,qz]
    mats = _xyzwxyz_to_matrix(xyzwxyz)
    xyzypr = _matrix_to_xyzypr(mats)
    return xyzypr[:, 3:]  # yaw, pitch, roll


def make_time_relative(ts):
    """Convert timestamps to seconds from start."""
    return ts - ts[0]


def subsample(arrays, max_points=5000):
    """Subsample arrays for plotting performance."""
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return [a[idx] for a in arrays]


def plot_3d_trajectory(ax, pos, ts_rel, title="3D Trajectory"):
    """Plot 3D position trajectory colored by time."""
    pos_sub, ts_sub = subsample([pos, ts_rel])
    sc = ax.scatter(
        pos_sub[:, 0], pos_sub[:, 1], pos_sub[:, 2],
        c=ts_sub, cmap="viridis", s=1, alpha=0.6
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="Time (s)", shrink=0.6)


def plot_imu_timeseries(axes, ts_rel, imu, colors):
    """Plot accelerometer and gyroscope on two subplots."""
    ts_sub, imu_sub = subsample([ts_rel, imu])
    accel_labels = ["ax", "ay", "az"]
    gyro_labels = ["gx", "gy", "gz"]

    for i in range(3):
        axes[0].plot(ts_sub, imu_sub[:, i], color=colors[i], linewidth=0.5, label=accel_labels[i])
    axes[0].set_ylabel("Accel (m/s²)")
    axes[0].set_title("Accelerometer")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    for i in range(3):
        axes[1].plot(ts_sub, imu_sub[:, 3 + i], color=colors[i], linewidth=0.5, label=gyro_labels[i])
    axes[1].set_ylabel("Gyro (rad/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Gyroscope")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].grid(True, alpha=0.3)


def plot_position_timeseries(ax, ts_rel, pos, colors):
    """Plot x/y/z position over time."""
    ts_sub, pos_sub = subsample([ts_rel, pos])
    for i, label in enumerate(["x", "y", "z"]):
        ax.plot(ts_sub, pos_sub[:, i], color=colors[i], linewidth=0.5, label=label)
    ax.set_ylabel("Position (m)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Position")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_orientation_timeseries(ax, ts_rel, euler, colors):
    """Plot yaw/pitch/roll over time."""
    ts_sub, euler_sub = subsample([ts_rel, euler])
    euler_deg = np.degrees(euler_sub)
    for i, label in enumerate(["yaw", "pitch", "roll"]):
        ax.plot(ts_sub, euler_deg[:, i], color=colors[i], linewidth=0.5, label=label)
    ax.set_ylabel("Angle (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Orientation (Euler)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_accel_magnitude(ax, ts_rel, imu):
    """Plot accelerometer magnitude to check gravity alignment."""
    ts_sub, imu_sub = subsample([ts_rel, imu])
    mag = np.linalg.norm(imu_sub[:, :3], axis=1)
    ax.plot(ts_sub, mag, color=ColorsPalette.PACBLUE[5], linewidth=0.5)
    ax.axhline(y=9.81, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="g=9.81")
    ax.set_ylabel("|accel| (m/s²)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Accel Magnitude (gravity check)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Visualize alignment NPZ data")
    parser.add_argument("npz_file", help="Path to NPZ file")
    parser.add_argument("--output_dir", default=None, help="Output directory for plots (default: same as NPZ)")
    args = parser.parse_args()

    npz_path = Path(args.npz_file)
    if not npz_path.exists():
        print(f"Error: {npz_path} not found")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem

    print(f"Loading {npz_path} ...")
    ts, imu, pos, quat = load_npz(npz_path)
    ts_rel = make_time_relative(ts)
    euler = quat_to_euler(quat)

    print(f"  Samples: {len(ts)}, Duration: {ts_rel[-1]:.1f}s")
    print(f"  Position range: {pos.min(axis=0)} to {pos.max(axis=0)}")
    print(f"  Accel magnitude mean: {np.linalg.norm(imu[:, :3], axis=1).mean():.3f} m/s²")

    # Use EgoVerse color palette
    xyz_colors = [ColorsPalette.PACBLUE[5], ColorsPalette.WILLOWGREEN[5], ColorsPalette.TIGERFLAME[5]]

    # --- Figure 1: Full dashboard ---
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"NPZ Visualization: {npz_path.name}", fontsize=14, fontweight="bold")

    # 3D trajectory
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    plot_3d_trajectory(ax3d, pos, ts_rel)

    # Position time series
    ax_pos = fig.add_subplot(2, 3, 2)
    plot_position_timeseries(ax_pos, ts_rel, pos, xyz_colors)

    # Orientation
    ax_ori = fig.add_subplot(2, 3, 3)
    plot_orientation_timeseries(ax_ori, ts_rel, euler, xyz_colors)

    # Accelerometer
    ax_acc = fig.add_subplot(2, 3, 4)
    plot_imu_timeseries([ax_acc, ax_acc], ts_rel, imu, xyz_colors)  # just accel on this one
    # override: do accel only
    ax_acc.clear()
    ts_sub, imu_sub = subsample([ts_rel, imu])
    for i, label in enumerate(["ax", "ay", "az"]):
        ax_acc.plot(ts_sub, imu_sub[:, i], color=xyz_colors[i], linewidth=0.5, label=label)
    ax_acc.set_ylabel("Accel (m/s²)")
    ax_acc.set_title("Accelerometer")
    ax_acc.legend(loc="upper right", fontsize=7)
    ax_acc.grid(True, alpha=0.3)

    # Gyroscope
    ax_gyro = fig.add_subplot(2, 3, 5)
    for i, label in enumerate(["gx", "gy", "gz"]):
        ax_gyro.plot(ts_sub, imu_sub[:, 3 + i], color=xyz_colors[i], linewidth=0.5, label=label)
    ax_gyro.set_ylabel("Gyro (rad/s)")
    ax_gyro.set_xlabel("Time (s)")
    ax_gyro.set_title("Gyroscope")
    ax_gyro.legend(loc="upper right", fontsize=7)
    ax_gyro.grid(True, alpha=0.3)

    # Accel magnitude
    ax_mag = fig.add_subplot(2, 3, 6)
    plot_accel_magnitude(ax_mag, ts_rel, imu)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / f"{stem}_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # --- Figure 2: Top-down (XY) trajectory ---
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    pos_sub, ts_sub = subsample([pos, ts_rel])
    sc = ax2.scatter(pos_sub[:, 0], pos_sub[:, 1], c=ts_sub, cmap="viridis", s=2, alpha=0.6)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title(f"Top-Down Trajectory: {npz_path.name}")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax2, label="Time (s)")
    out_path2 = out_dir / f"{stem}_trajectory_xy.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out_path2}")


if __name__ == "__main__":
    main()
