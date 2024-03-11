# change the csv file contents to a frequency graph with color

# from turtle import pd
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# from scipy import signal
# from global_parameters import get_imu_max_min

def normalize_imu1(data, max_val, min_val):
    mean = data.mean()
    # print(mean)
    return data - mean

def normalize_imu2(data, range, max_val, min_val):
    mean = (max_val - min_val) / 2
    range_val = range
    return (data - mean) / range_val

def normalize_column(column, min_val, max_val):
    """Normalize a single column of data to range between min_val and max_val."""
    return (column - min_val) / (max_val - min_val)

def get_imu_max_min(prefix, df):
    cols = [col for col in df.columns if col.startswith(prefix)]
    max_vals = df[cols].max().max()
    min_vals = df[cols].min().min()
    return max_vals, min_vals

def process_and_save_image(file_name, output_folder, max_acc, min_acc, max_gyr, min_gyr, max_Mag, min_Mag,
                        range_acc, range_gyr, range_Mag):
    df = pd.read_csv(file_name)
    # df_normalization = pd.read_csv(normal_path)
    max_acc, min_acc, max_gyr, min_gyr, max_Mag, min_Mag = max_acc, min_acc, max_gyr, min_gyr, max_Mag, min_Mag
    range_acc, range_gyr, range_Mag = range_acc, range_gyr, range_Mag
    # max_acc = 1000
    # min_acc = -1000
    # max_gyr = 1000
    # min_gyr = -1000
    
    for col in df.columns:
        if "Acc" in col:
            df[col] = normalize_imu1(df[col], max_acc, min_acc)
        elif "Gyr" in col:
            df[col] = normalize_imu1(df[col], max_gyr, min_gyr)
        elif "Mag" in col:
            df[col] = normalize_imu2(df[col], range_Mag, max_Mag, min_Mag)

    data_normalized = df.to_numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(data_normalized.T, aspect='auto', cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Normalized Value')
    plt.title(f'Normalized IMU Data Visualization for {os.path.basename(file_name)}')
    plt.xlabel('Sample Index')
    plt.ylabel('Sensor Index')
    plt.yticks(np.arange(data_normalized.shape[1]), df.columns)

    plt.savefig(os.path.join(output_folder, os.path.basename(file_name).replace(".csv", ".png")))
    plt.close()
    
def process_files_in_folder(input_folder, train_output_folder_base, test_output_folder_base, train_ratio=0.6):
    for subdir, dirs, files in os.walk(input_folder):
        label = os.path.basename(subdir)
        all_files = [file for file in files if file.endswith(".csv")]
        random.shuffle(all_files)  # Shuffle to ensure random distribution

        # Calculate the split index
        split_index = int(len(all_files) * train_ratio)
        train_files = all_files[:split_index]
        test_files = all_files[split_index:]

        for file_set, output_folder_base in [(train_files, train_output_folder_base), (test_files, test_output_folder_base)]:
            output_folder = os.path.join(output_folder_base, label)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            for file in file_set:
                full_path = os.path.join(subdir, file)
                df = pd.read_csv(full_path)

                temp_max_min_vals = [get_imu_max_min(f'IMU{j}_{sensor}', df) for j in range(1, 4) for sensor in ['Acc', 'Gyr', 'Mag']]
                max_vals, min_vals = zip(*temp_max_min_vals)
                max_acc, min_acc, max_gyr, min_gyr, max_Mag, min_Mag = max(max_vals), min(min_vals), max(max_vals), min(min_vals), max(max_vals), min(min_vals)

                range_acc = max_acc - min_acc
                range_gyr = max_gyr - min_gyr
                range_Mag = max_Mag - min_Mag
                
                process_and_save_image(full_path, output_folder, max_acc, min_acc, max_gyr, min_gyr, max_Mag, min_Mag, range_acc, range_gyr, range_Mag)

input_folder = "../train_data_fv/"
train_output_folder_base = "../train_data_imu_pic/"
test_output_folder_base = "../test_data_imu_pic/"

process_files_in_folder(input_folder, train_output_folder_base, test_output_folder_base)
