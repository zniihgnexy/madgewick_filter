# change the csv file contents to a frequency graph with color

# from turtle import pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# from scipy import signal
# from global_parameters import get_imu_max_min

def normalize_imu1(data, max_val, min_val):
    return (data - data.min()) / 2000

def normalize_column(column, min_val, max_val):
    """Normalize a single column of data to range between min_val and max_val."""
    return (column - min_val) / (max_val - min_val)

def get_imu_max_min(prefix, df):
    cols = [col for col in df.columns if col.startswith(prefix)]
    max_vals = df[cols].max().max()
    min_vals = df[cols].min().min()
    return max_vals, min_vals

def process_and_save_image(file_name, output_folder):
    df = pd.read_csv(file_name)
    # df_normalization = pd.read_csv(normal_path)
    max_acc, min_acc = get_imu_max_min("Acc", df)
    max_gyr, min_gyr = get_imu_max_min("Gyr", df)
    # print(max_acc, min_acc, max_gyr, min_gyr)
    
    # max_acc = 1000
    # min_acc = -1000
    # max_gyr = 1000
    # min_gyr = -1000
    
    for col in df.columns:
        if "Acc" in col:
            df[col] = normalize_column(df[col], -1, 1)
        elif "Gyr" in col:
            df[col] = normalize_imu1(df[col], -1000, 1000)
        elif "Mag" in col:
            df[col] = normalize_column(df[col], 1, -1)

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
    
input_folder = "../train_data/baseline/"
output_folder = "../train_data/normalized_images/baseline/"
root_dir = "../train_data/normalized_images/"
# ori_file = "../train_data_ori/imu_train_data_baseline_sampled.csv"
data_labels = ["baseline"]

output_csv_path = "../train_data/normalized_images/labels.csv"
df_labels = pd.DataFrame(columns=['filename', 'label'])

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read existing labels if the file exists, otherwise create a new DataFrame
if os.path.exists(output_csv_path):
    df_labels = pd.read_csv(output_csv_path)
else:
    df_labels = pd.DataFrame(columns=['filename', 'label'])

for i in range(1, 158):
    file_name = f"baseline_imu_train_data_part{i}.csv"
    full_path = os.path.join(input_folder, file_name)
    # normal_path = os.path.join(ori_file)
    image_filename = process_and_save_image(full_path, output_folder)
    # df_labels = df_labels.append({'filename': image_filename, 'label': 'seat_over'}, ignore_index=True)


# Save the updated labels to CSV, without the index
# df_labels.to_csv(output_csv_path, index=False)