# change the csv file contents to a frequency graph with color

# from turtle import pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# from scipy import signal
# from global_parameters import get_imu_max_min

def normalize_imu(data, max_val, min_val):
    return (data - data.min()) / (data.max() - data.min())

def get_imu_max_min(prefix, df):
    cols = [col for col in df.columns if col.startswith(prefix)]
    max_vals = df[cols].max().max()
    min_vals = df[cols].min().min()
    return max_vals, min_vals

def process_and_save_image(file_name, output_folder):
    df = pd.read_csv(file_name)

    max_acc, min_acc = get_imu_max_min("Acc", df)
    max_gyr, min_gyr = get_imu_max_min("Gyr", df)
    
    for col in df.columns:
        if "Acc" in col:
            df[col] = normalize_imu(df[col], max_acc, min_acc)
        elif "Gyr" in col:
            df[col] = normalize_imu(df[col], max_gyr, min_gyr)

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
    
input_folder = "../train_data/random/"
output_folder = "../train_data/normalized_images/random/"
root_dir = "../train_data/normalized_images/"
data_labels = []

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(1, 101):
    file_name = f"random_imu1_train_data_part{i}.csv"
    full_path = os.path.join(input_folder, file_name)
    process_and_save_image(full_path, output_folder)

output_csv_path = "../train_data/normalized_images/labels.csv"
df_labels = pd.DataFrame(columns=['filename', 'label'])

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".png"):
            file_path = os.path.join(subdir, file)
            label = os.path.basename(subdir)
            data_labels.append((file_path.replace(root_dir, ""), label))

df_labels = pd.DataFrame(data_labels, columns=['filename', 'label'])

df_labels.to_csv(output_csv_path, index=False)