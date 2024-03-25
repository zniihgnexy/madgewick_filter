import pandas as pd
import os
import json
import sys

num_data = {}

def sliding_window(data, window_size, overlap):
    windows = []
    step_size = window_size - overlap
    for i in range(0, len(data), step_size):
        window = data[i:i + window_size]
        if len(window) == window_size:
            windows.append(window)
    return windows

def normalize_column(column, min_val, max_val):
    return (column - min_val) / (max_val - min_val)

def split_csv_file(filename, path_ori, path, window_size, overlap, label, column_names):
    df = pd.read_csv(path_ori + filename)
    windowed_data = sliding_window(df, window_size, overlap)
    df.columns = column_names
    for i, window_df in enumerate(windowed_data):
        new_filename = f"{path}/{label}/{label}_part{i+1}.csv"
        os.makedirs(os.path.dirname(new_filename), exist_ok=True)
        window_df.to_csv(new_filename, index=False)
        # print(f"Saved: {new_filename}")
    num_data[label] = len(windowed_data)
    # print(f"Number of {label} data: {num_data}")
    with open('num_data.json', 'w') as f:
        json.dump(num_data, f)

def process_from_json(json_file_path, directory_ori, directory, window_size, overlap, column_names):
    with open(json_file_path, 'r') as f:
        tasks = json.load(f)["tasks"]
        for task in tasks:
            filename = task["filename"]
            label = task["label"]
            split_csv_file(filename, directory_ori, directory, window_size, overlap, label, column_names)

# File and directory settings
# json_file_path = 'file_path/windowing_data.json'
# directory_ori = "../train_data_ori/"
# directory = "E:/master-2/madgewick_filter/video_imu/window_data/"
# window_size = 150
# overlap = 145

json_file_path = sys.argv[1]
directory_ori = sys.argv[2]
directory = sys.argv[3]
window_size = int(sys.argv[4])
overlap = int(sys.argv[5])

column_names = [
    'IMU1_AccX', 'IMU1_AccY', 'IMU1_AccZ', 'IMU1_GyrX', 'IMU1_GyrY', 'IMU1_GyrZ', 'IMU1_MagX', 'IMU1_MagY', 'IMU1_MagZ', 
    'IMU2_AccX', 'IMU2_AccY', 'IMU2_AccZ', 'IMU2_GyrX', 'IMU2_GyrY', 'IMU2_GyrZ', 'IMU2_MagX', 'IMU2_MagY', 'IMU2_MagZ', 
    'IMU3_AccX', 'IMU3_AccY', 'IMU3_AccZ', 'IMU3_GyrX', 'IMU3_GyrY', 'IMU3_GyrZ', 'IMU3_MagX', 'IMU3_MagY', 'IMU3_MagZ'
]

# Start processing
process_from_json(json_file_path, directory_ori, directory, window_size, overlap, column_names)
print(num_data)