#!/usr/bin/env python3
import pandas as pd
import alignment_library as al
import sys
import os
import shutil
from pathlib import Path
import traceback

# --- Configuration ---

# 1. File Paths
BASE_INPUT_DIR = '/home/hshi/Desktop/AirLab/Lamar/organized_data'
BASE_OUTPUT_DIR = '/home/hshi/Desktop/AirLab/Lamar/lamar-v2-processed-NO-GRAVITY' # All files go here
GRAVITY_TF_CSV = None

# 2. Data Properties
TIMESTAMP_UNIT = 1.0  
IMU_GRAVITY = 9.8105

# 3. Alignment Parameters
TARGET_RATE_HZ = 200.0
GRAVITY_ESTIMATE_SAMPLES = 200

# 4. File Identification
IMU_FILENAME = 'fused_imu.txt'
GT_FILENAME = 'trajectories.txt'

# 5. Column Mapping (Correlation)
# The fused_imu.txt has 14 columns. We must map them carefully so the library
# receives standard IMU naming conventions (gx=gyro, not gravity vector).

# Raw columns as they appear in the file structure spec:
# timestamp, ax, ay, az, rx, ry, rz, mx, my, mz, gx, gy, gz, [heading | qx, qy, qz, qw]

# We define the base names we rely on and add optional columns dynamically
# so the timestamp field never drifts when extra quaternion values are present.
FUSED_IMU_BASE_COLS = [
    'timestamp',
    'ax', 'ay', 'az',            # User Acceleration
    'gx', 'gy', 'gz',            # Rotation Rate (mapped from rx, ry, rz)
    'mx', 'my', 'mz',            # Magnetometer
    'grav_x', 'grav_y', 'grav_z' # Gravity Vector (mapped from gx, gy, gz)
]
FUSED_IMU_HEADING_COL = ['heading']
FUSED_IMU_QUAT_COLS = ['quat_x', 'quat_y', 'quat_z', 'quat_w']

# trajectories.txt columns: timestamp, device_id, qw, qx, qy, qz, tx, ty, tz, [optional covariance...]
TRAJECTORIES_COLS_BASE = ['timestamp', 'device_id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz']

# --- End Configuration ---

def find_gt_file(directory):
    """Find trajectories.txt file in the directory."""
    gt_path = directory / GT_FILENAME
    if gt_path.exists():
        return gt_path
    return None

def load_fused_imu(imu_path):
    print(f"  Loading IMU: {imu_path.name}")
    try:
        # Determine column count so we can assign the correct headers (14 vs 17+)
        first_data_line = None
        with open(imu_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    first_data_line = line
                    break
        if first_data_line is None:
            print("  Error: IMU file contains no data rows.")
            return None

        num_cols = len(first_data_line.split(','))
        col_names = FUSED_IMU_BASE_COLS.copy()

        if num_cols >= len(FUSED_IMU_BASE_COLS) + len(FUSED_IMU_QUAT_COLS):
            col_names.extend(FUSED_IMU_QUAT_COLS)
        elif num_cols >= len(FUSED_IMU_BASE_COLS) + len(FUSED_IMU_HEADING_COL):
            col_names.extend(FUSED_IMU_HEADING_COL)

        # Append placeholder names for any other trailing columns so parsing never misaligns timestamp
        while len(col_names) < num_cols:
            col_names.append(f'extra_{len(col_names) - len(FUSED_IMU_BASE_COLS)}')

        # Load CSV, treating lines starting with # as comments
        # We enforce our custom column names here to handle the correlation
        df = pd.read_csv(
            imu_path, 
            sep=',', 
            comment='#', 
            names=col_names, 
            skipinitialspace=True
        )

        # Convert Microseconds to Seconds
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce') / 1_000_000.0
        df = df.dropna(subset=['timestamp'])
        
        # Combine user acceleration with gravity vector to get total acceleration
        # User acceleration (ax, ay, az) is device acceleration without gravity
        # Gravity vector (grav_x, grav_y, grav_z) is normalized, so scale by IMU_GRAVITY
        if all(col in df.columns for col in ['ax', 'ay', 'az', 'grav_x', 'grav_y', 'grav_z']):
            # Scale normalized gravity vector by gravity magnitude and add to user acceleration
            df['ax'] = df['ax'] + df['grav_x'] * IMU_GRAVITY
            df['ay'] = df['ay'] + df['grav_y'] * IMU_GRAVITY
            df['az'] = df['az'] + df['grav_z'] * IMU_GRAVITY
            print(f"  Combined user acceleration with gravity vector (scaled by {IMU_GRAVITY} m/s²)")
        else:
            print(f"  Warning: Cannot combine acceleration - missing columns. Available: {df.columns.tolist()}")
        
        # Verify required columns exist for alignment
        required = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
        if not all(col in df.columns for col in required):
            print(f"  Error: Missing required columns. Cols: {df.columns}")
            return None

        return df

    except Exception as e:
        print(f"  Error reading IMU file: {e}")
        return None

def load_gt_data(gt_path):
    """
    Load trajectories.txt file.
    Format: timestamp (microseconds), device_id, qw, qx, qy, qz, tx, ty, tz, [optional covariance...]
    """
    print(f"  Loading GT:  {gt_path.name}")
    try:
        # Read CSV, treating lines starting with # as comments
        # We'll read more columns than base to handle optional covariance
        # First, try to determine number of columns
        with open(gt_path, 'r') as f:
            first_data_line = None
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    first_data_line = line
                    break
        
        if first_data_line is None:
            print("  Error: No data lines found in trajectories.txt")
            return None
        
        # Count columns from first data line
        num_cols = len(first_data_line.split(','))
        
        # Create column names: base columns + optional covariance columns
        col_names = TRAJECTORIES_COLS_BASE.copy()
        if num_cols > len(TRAJECTORIES_COLS_BASE):
            # Add covariance columns (36 values for 6x6 matrix)
            for i in range(len(TRAJECTORIES_COLS_BASE), num_cols):
                col_names.append(f'covar_{i}')
        
        # Read the CSV file
        gt_df = pd.read_csv(
            gt_path,
            sep=',',
            comment='#',
            names=col_names,
            skipinitialspace=True
        )
        
        # Convert timestamp from microseconds to seconds (same as IMU)
        gt_df['timestamp'] = gt_df['timestamp'] / 1_000_000.0
        
        # Drop device_id column (ignore it)
        if 'device_id' in gt_df.columns:
            gt_df = gt_df.drop(columns=['device_id'])
        
        # Verify required columns exist
        required = ['timestamp', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz']
        if not all(col in gt_df.columns for col in required):
            print(f"  Error: Missing required columns. Found: {gt_df.columns.tolist()}")
            return None
        
        print(f"  Loaded {len(gt_df)} GT poses")
        return gt_df
        
    except Exception as e:
        print(f"  Error reading GT file: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_directory(input_dir_path, main_output_folder):
    """
    Process a single directory, but output the result to the main flat folder
    with a unique name.
    """
    dir_name = input_dir_path.name
    # Create a unique ID string from the parent path to avoid filename collisions
    # e.g., if path is /data/2023/session1 -> unique_id is "2023_session1"
    unique_id = "_".join(input_dir_path.parts[-2:]) 
    
    print(f"\n--- Processing: {unique_id} ---")

    # 1. Locate Files
    imu_path = input_dir_path / IMU_FILENAME
    gt_path = find_gt_file(input_dir_path)

    if not imu_path.exists() or not gt_path:
        print("  Skipping: Missing IMU or GT files.")
        return False

    # 2. Load Data
    imu_df = load_fused_imu(imu_path)
    gt_df = load_gt_data(gt_path)
    
    if imu_df is None or gt_df is None:
        return False

    # 3. Handle Output Strategy
    # The alignment library likely writes a fixed filename to the 'out_dir'.
    # To support a flat output folder, we write to a temporary folder,
    # rename the file, and move it to the main output.
    temp_work_dir = main_output_folder / f"temp_{unique_id}"
    temp_work_dir.mkdir(parents=True, exist_ok=True)

    print(f"  > Running alignment...")
    try:
        al.align_and_interpolate(
            imu_df=imu_df,
            gt_df=gt_df,
            timestamp_unit=TIMESTAMP_UNIT,
            imu_gravity=IMU_GRAVITY,
            out_dir=str(temp_work_dir),
            gravity_tf_csv=GRAVITY_TF_CSV,
            target_rate=TARGET_RATE_HZ,
            estimate_gravity_samples=GRAVITY_ESTIMATE_SAMPLES
        )
        
        # 4. Move and Rename Result
        # The alignment library creates aligned_imu.csv and aligned_gt.csv (and possibly .npz)
        # Move all output files with unique names
        output_files_moved = False
        
        # Move aligned_imu.csv if it exists
        imu_file = temp_work_dir / 'aligned_imu.csv'
        if imu_file.exists():
            imu_dest = main_output_folder / f"{unique_id}_aligned_imu.csv"
            shutil.move(str(imu_file), str(imu_dest))
            print(f"  > Moved IMU: {imu_dest.name}")
            output_files_moved = True
        
        # Move aligned_gt.csv if it exists
        gt_file = temp_work_dir / 'aligned_gt.csv'
        if gt_file.exists():
            gt_dest = main_output_folder / f"{unique_id}_aligned_gt.csv"
            shutil.move(str(gt_file), str(gt_dest))
            print(f"  > Moved GT: {gt_dest.name}")
            output_files_moved = True
        
        # Move .npz file if it exists
        npz_files = list(temp_work_dir.glob("*.npz"))
        for npz_file in npz_files:
            npz_dest = main_output_folder / f"{unique_id}_{npz_file.name}"
            shutil.move(str(npz_file), str(npz_dest))
            print(f"  > Moved NPZ: {npz_dest.name}")
            output_files_moved = True
        
        # Cleanup temp dir
        if temp_work_dir.exists():
            shutil.rmtree(temp_work_dir)
        
        if output_files_moved:
            print(f"  > Success!")
            return True
        else:
            print("  > Error: Library finished but no output files found.")
            return False
        
    except Exception as e:
        print(f"  > FAILED: {e}")
        traceback.print_exc()
        # Attempt cleanup on fail
        if temp_work_dir.exists():
            shutil.rmtree(temp_work_dir)
        return False

if __name__ == '__main__':
    print("--- Starting Recursive Fused IMU Alignment (Flat Output) ---")
    
    base_input = Path(BASE_INPUT_DIR)
    base_output = Path(BASE_OUTPUT_DIR)
    
    # Ensure main output directory exists
    base_output.mkdir(parents=True, exist_ok=True)
    
    if not base_input.is_dir():
        print(f"Error: Base directory not found: {BASE_INPUT_DIR}")
        sys.exit(1)
            
    processed_count = 0
    failed_count = 0

    # Recursive walk - find directories containing both IMU and GT files
    for root, dirs, files in os.walk(base_input):
        if IMU_FILENAME in files and GT_FILENAME in files:
            current_input_dir = Path(root)
            try:
                # Pass the flat base_output directory directly
                if process_directory(current_input_dir, base_output):
                    processed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"CRITICAL ERROR in {current_input_dir}: {e}")
                traceback.print_exc()
                failed_count += 1

    print("\n--- Complete ---")
    print(f"Success: {processed_count}")
    print(f"Failed:  {failed_count}")
    print(f"All files located in: {base_output}")