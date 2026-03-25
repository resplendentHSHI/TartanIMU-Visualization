#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

# NOTE: All global constants (paths, etc.) have been removed.
# They will be passed in as arguments by the script that imports this library.


def compute_gravity_rotation_from_accels(accel_samples):
    print("fixed gravity rotation")
    mean_acc = np.mean(accel_samples, axis=0).astype(float)
    if np.linalg.norm(mean_acc) < 1e-8:
        return np.eye(3)
    
    v_from = mean_acc / np.linalg.norm(mean_acc)
    v_to = np.array([0.0, 0.0, -1.0]) # Target: gravity points down Z axis
    
    if np.allclose(v_from, v_to, atol=1e-6):
        return np.eye(3) # Already aligned

    axis = np.cross(v_from, v_to)
    s = np.linalg.norm(axis) # sine of angle
    c = np.dot(v_from, v_to) # cosine of angle
    
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        else:
            if abs(v_from[0]) < 0.9:
                axis = np.cross(v_from, np.array([1.0, 0.0, 0.0]))
            else:
                axis = np.cross(v_from, np.array([0.0, 1.0, 0.0]))
            axis = axis / np.linalg.norm(axis)
            return -np.eye(3) + 2.0 * np.outer(axis, axis)
            
    axis = axis / s
    K = np.array([[0, -axis[2], axis[1]], 
                  [axis[2], 0, -axis[0]], 
                  [-axis[1], axis[0], 0]])
    
    R = np.eye(3) + K * s + K.dot(K) * (1 - c)
    return np.eye(3)
    #return R


def align_and_interpolate(
    imu_df, 
    gt_df, 
    timestamp_unit,
    imu_gravity,
    out_dir,
    gravity_tf_csv=None, 
    target_rate=200.0, 
    estimate_gravity_samples=200
):
    """
    Main alignment function.
    Accepts pre-loaded and pre-processed DataFrames.
    """
    
    # --- MODIFICATION ---
    # CSV reading is removed. We now assume imu_df and gt_df are passed in.
    # --- END MODIFICATION ---
    
    imu_df['timestamp'] = pd.to_numeric(imu_df['timestamp'], errors='coerce')
    gt_df['timestamp'] = pd.to_numeric(gt_df['timestamp'], errors='coerce')
    imu_df = imu_df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    gt_df = gt_df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    #find overlap
    imu_start, imu_end = imu_df['timestamp'].iloc[0], imu_df['timestamp'].iloc[-1]
    gt_start, gt_end = gt_df['timestamp'].iloc[0], gt_df['timestamp'].iloc[-1]
    overlap_start = max(imu_start, gt_start)
    overlap_end = min(imu_end, gt_end)
    
    print(f"IMU timestamp range: {imu_start} to {imu_end} ({len(imu_df)} samples)")
    print(f"GT timestamp range:  {gt_start} to {gt_end} ({len(gt_df)} samples)")
    print(f"Overlap range: {overlap_start} to {overlap_end}")
    
    if overlap_end <= overlap_start:
        raise ValueError(f'No overlap: IMU [{imu_start}, {imu_end}] vs GT [{gt_start}, {gt_end}]')

    accel_cols = ['accel_x','accel_y','accel_z']
    gyro_cols = ['gyro_x','gyro_y','gyro_z']

    # determine gravity alignment rotation R
    R = np.eye(3)
    gpath = Path(gravity_tf_csv) if gravity_tf_csv is not None else None
    if gpath is not None and gpath.exists():
        try:
            gdf = pd.read_csv(gpath)
            if all(c in gdf.columns for c in ('qx','qy','qz','qw')):
                qx, qy, qz, qw = gdf[['qx','qy','qz','qw']].iloc[0].values.astype(float)
            elif all(c in gdf.columns for c in ('qx','qy','qz')) and 'w' in gdf.columns:
                qx, qy, qz, qw = gdf[['qx','qy','qz','w']].iloc[0].values.astype(float)
            else:
                raise ValueError('Quaternion columns not found in gravity TF CSV')

            q = np.array([qx, qy, qz, qw], dtype=float)
            q = q / np.linalg.norm(q)
            x, y, z, w = q
            R = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
                [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
                [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
            ])
            print(f"Using gravity TF from {gpath} with quaternion ({qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f})")
        except Exception as e:
            print(f"Warning: failed to read gravity TF ({gpath}): {e} -- falling back to accel-based estimate")
            gpath = None

    if gpath is None:
        accel_fallback = ['ax','ay','az']
        chosen_accel = accel_cols if all(c in imu_df.columns for c in accel_cols) else (accel_fallback if all(c in imu_df.columns for c in accel_fallback) else None)
        if chosen_accel is not None:
            n = min(estimate_gravity_samples, len(imu_df))
            samples = imu_df[chosen_accel].iloc[:n].values
            R = compute_gravity_rotation_from_accels(samples)
        else:
            R = np.eye(3)

    imu_aligned = imu_df.copy()
    accel_candidates = [accel_cols, ['ax','ay','az']]
    gyro_candidates = [gyro_cols, ['gx','gy','gz'], ['gyro_x','gyro_y','gyro_z']]

    def _pick_cols(df, candidates):
        for cand in candidates:
            if all(c in df.columns for c in cand):
                return cand
        return None

    src_accel = _pick_cols(imu_df, accel_candidates)
    src_gyro = _pick_cols(imu_df, gyro_candidates)

    if src_accel is not None:
        acc = imu_df[src_accel].values.T
        rotated = R.dot(acc).T 
        try:
            n = min(estimate_gravity_samples, len(imu_df))
            mean_vec = imu_df[src_accel].iloc[:n].values.mean(axis=0).astype(float)
            acc_mean_norm = np.linalg.norm(mean_vec)
            if acc_mean_norm > 1e-8:
                scale = imu_gravity / acc_mean_norm
            else:
                scale = 1.0
        except Exception:
            scale = 1.0
        print(f"Warning: original code used scale {scale}")


        if abs(scale - 1.0) > 1e-6:
            print(f"Applying Livox gravity scale: {scale:.6f} (acc_mean_norm={acc_mean_norm:.6f})")

        rotated = rotated * scale
        imu_aligned['accel_x_ga'] = rotated[:,0]
        imu_aligned['accel_y_ga'] = rotated[:,1]
        imu_aligned['accel_z_ga'] = rotated[:,2]
    else:
        print('Warning: no accel columns found to apply gravity TF')

    if src_gyro is not None:
        gyr = imu_df[src_gyro].values.T
        rotatedg = R.dot(gyr).T
        imu_aligned['gyro_x_ga'] = rotatedg[:,0]
        imu_aligned['gyro_y_ga'] = rotatedg[:,1]
        imu_aligned['gyro_z_ga'] = rotatedg[:,2]
    else:
        print('Warning: no gyro columns found to apply gravity TF')

    imu_crop = imu_aligned[(imu_aligned['timestamp'] >= overlap_start) & (imu_aligned['timestamp'] <= overlap_end)].reset_index(drop=True)
    gt_crop = gt_df[(gt_df['timestamp'] >= overlap_start) & (gt_df['timestamp'] <= overlap_end)].reset_index(drop=True)
    
    dt = (1.0 / float(target_rate)) * timestamp_unit  # convert target_rate (Hz) to timestamp units
    common_times = np.arange(overlap_start, overlap_end + dt * 1e-6, dt)

    imu_interp_cols = []
    for c in ['accel_x_ga','accel_y_ga','accel_z_ga','gyro_x_ga','gyro_y_ga','gyro_z_ga']:
        if c in imu_crop.columns:
            imu_interp_cols.append(c)
    if not any(c.endswith('_ga') for c in imu_interp_cols):
        for c in ['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']:
            if c in imu_crop.columns:
                imu_interp_cols.append(c)

    imu_interp_df = pd.DataFrame({'timestamp': common_times})
    for col in imu_interp_cols:
        f = interp1d(imu_crop['timestamp'], imu_crop[col], bounds_error=False, fill_value='extrapolate')
        imu_interp_df[col] = f(common_times)

    gt_interp_cols = [c for c in gt_crop.columns if c not in ['timestamp', 'device_id'] and np.issubdtype(gt_crop[c].dtype, np.number)]
    gt_interp_df = pd.DataFrame({'timestamp': common_times})
    for col in gt_interp_cols:
        f = interp1d(gt_crop['timestamp'], gt_crop[col], bounds_error=False, fill_value='extrapolate')
        gt_interp_df[col] = f(common_times)

    if all(c in gt_interp_df.columns for c in ['qw', 'qx', 'qy', 'qz']):
        quat_data = gt_interp_df[['qx', 'qy', 'qz', 'qw']].values
        for i in range(len(quat_data)):
            q = quat_data[i]
            q_norm = np.linalg.norm(q)
            if q_norm > 1e-8:
                quat_data[i] = q / q_norm
        gt_interp_df[['qx', 'qy', 'qz', 'qw']] = quat_data

    # --- FIX: Convert output timestamps to seconds ---
    print(f"Converting output timestamps from units (1e{np.log10(timestamp_unit):.0f}) to seconds.")
    imu_interp_df['timestamp'] = imu_interp_df['timestamp'] / timestamp_unit
    gt_interp_df['timestamp'] = gt_interp_df['timestamp'] / timestamp_unit
    # --- End of Fix ---

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    imu_csv_out = outp / 'aligned_imu.csv'
    gt_csv_out = outp / 'aligned_gt.csv'
    imu_interp_df.to_csv(imu_csv_out, index=False)
    
    gt_rename_map = {
        'tx': 'x',
        'ty': 'y',
        'tz': 'z'
    }
    gt_interp_df = gt_interp_df.rename(columns=gt_rename_map)
    print(f"Renamed GT columns for output: {list(gt_rename_map.keys())} -> {list(gt_rename_map.values())}")
    
    desired_order = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    available_cols = [col for col in desired_order if col in gt_interp_df.columns]
    gt_interp_df = gt_interp_df[available_cols]
        
    gt_interp_df.to_csv(gt_csv_out, index=False)

    try:
        convert_csv_to_npz(outp, outp)
    except Exception as e:
        print(f"Warning: failed to convert CSV to NPZ: {e}")

    print('Rotation R:')
    np.set_printoptions(precision=6, suppress=True)
    print(R)
    print('\nIMU samples before/after: ', len(imu_df), len(imu_interp_df))
    print('GT samples before/after: ', len(gt_df), len(gt_interp_df))
    print('\nPreview aligned IMU (first 5 rows):')
    print(imu_interp_df.head())
    return imu_interp_df, gt_interp_df, R


def convert_csv_to_npz(sequence_dir, output_dir):
    """
    Convert aligned CSV files to NPZ format with specified structure
    (This function is unchanged)
    """
    sequence_dir = Path(sequence_dir)
    output_dir = Path(output_dir)

    imu_file = sequence_dir / 'aligned_imu.csv'
    gt_file = sequence_dir / 'aligned_gt.csv'

    if not imu_file.exists():
        raise FileNotFoundError(f"IMU file not found: {imu_file}")
    if not gt_file.exists():
        raise FileNotFoundError(f"GT file not found: {gt_file}")

    imu_data = pd.read_csv(imu_file)
    gt_data = pd.read_csv(gt_file)

    ts = imu_data['timestamp'].values

    accel_candidates = [
        ('accel_x_ga','accel_y_ga','accel_z_ga'),
        ('accel_x','accel_y','accel_z'),
        ('ax','ay','az')
    ]
    gyro_candidates = [
        ('gyro_x_ga','gyro_y_ga','gyro_z_ga'),
        ('gyro_x','gyro_y','gyro_z'),
        ('gx','gy','gz')
    ]

    def _find_cols(df, candidates):
        for cand in candidates:
            if all(c in df.columns for c in cand):
                return list(cand)
        return []

    accel_cols = _find_cols(imu_data, accel_candidates)
    gyro_cols = _find_cols(imu_data, gyro_candidates)

    if not accel_cols or not gyro_cols:
        raise ValueError('Required IMU accel/gyro columns not found in aligned_imu.csv')

    imu_accel = imu_data[accel_cols].values
    imu_gyro = imu_data[gyro_cols].values
    imu = np.hstack([imu_accel, imu_gyro])

    if all(c in gt_data.columns for c in ('x','y','z')):
        pos = gt_data[['x','y','z']].values
    else:
        raise ValueError('GT position columns x,y,z not found')

    if all(c in gt_data.columns for c in ('qx','qy','qz','qw')):
        quat = gt_data[['qx','qy','qz','qw']].values
    elif all(c in gt_data.columns for c in ('w','x','y','z')):
        q = gt_data[['w','x','y','z']].values
        quat = q[:,[1,2,3,0]]
    else:
        quat = np.zeros((len(pos),4), dtype=float)
        quat[:,3] = 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_name = sequence_dir.name
    npz_file = output_dir / f"{sequence_name}.npz"

    np.savez_compressed(
        npz_file,
        retargetted_ts=ts,
        retargetted_imu=imu,
        retargetted_pos=pos,
        retargetted_quat=quat
    )

    print(f"Saved NPZ file: {npz_file}")
    print(f"  - Timestamps: {ts.shape}")
    print(f"  - IMU data: {imu.shape} (ax,ay,az,gx,gy,gz)")
    print(f"  - Position: {pos.shape} (x,y,z)")
    print(f"  - Quaternion: {quat.shape} (x,y,z,w)")

    verify_npz_file(npz_file)


def verify_npz_file(npz_file):
    """
    (This function is unchanged)
    """
    try:
        with np.load(npz_file) as data:
            print(f"\nVerifying NPZ file: {npz_file}")
            print(f"Available keys: {list(data.keys())}")

            ts = data['retargetted_ts']
            imu = data['retargetted_imu']
            pos = data['retargetted_pos']
            quat = data['retargetted_quat']

            print(f"  - Timestamps: {ts.shape}, range: {ts[0]:.6f} to {ts[-1]:.6f}")
            print(f"  - IMU: {imu.shape}, range: [{np.nanmin(imu):.6f}, {np.nanmax(imu):.6f}]")
            print(f"  - Position: {pos.shape}, range: [{np.nanmin(pos):.6f}, {np.nanmax(pos):.6f}]")
            print(f"  - Quaternion: {quat.shape}, norm range: [{np.min(np.linalg.norm(quat,axis=1)):.6f}, {np.max(np.linalg.norm(quat,axis=1)):.6f}]")

            quat_norms = np.linalg.norm(quat, axis=1)
            if not np.allclose(quat_norms, 1.0, atol=1e-6):
                print(f"  Warning: Quaternions are not normalized! Norm range: [{quat_norms.min():.6f}, {quat_norms.max():.6f}]")
            else:
                print(f"  Quaternions are properly normalized")
    except Exception as e:
        print(f"Error verifying NPZ file: {e}")