# change the csv file contents to a frequency graph with color

# from turtle import pd
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
        if "IMU1_Acc" in col:
            df[col] = normalize_imu1(df[col],max_acc, min_acc)
        elif "IMU2_Acc" in col:
            df[col] = normalize_imu1(df[col],max_acc, min_acc)
        elif "IMU3_Acc" in col:
            df[col] = normalize_imu1(df[col],max_acc, min_acc)
        elif "Gyr" in col:
            mean_gyr = (max_gyr + min_gyr) / 2
            # df[col] = normalize_imu2(df[col], mean_gyr)
            df[col] = normalize_imu1(df[col], max_gyr, min_gyr)
        elif "IMU1_Mag" in col:
            mean = df[col].mean()
            df[col] = normalize_imu2(df[col], range_Mag, max_Mag, min_Mag)
        elif "IMU2_Mag" in col:
            mean = df[col].mean()
            df[col] = normalize_imu2(df[col], range_Mag, max_Mag, min_Mag)
        elif "IMU3_Mag" in col:
            mean = df[col].mean()
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
    
input_folder = "../train_data/IMU/imu_baseline_2/"
output_folder = "../train_data/normalized_images_1/baseline/"
root_dir = "../train_data/normalized_images_1/"
# ori_file = "../train_data_ori/imu_train_data_baseline_sampled.csv"
data_labels = ["baseline"]

# output_csv_path = "../train_data/normalized_images/labels.csv"
# df_labels = pd.DataFrame(columns=['filename', 'label'])

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# # Read existing labels if the file exists, otherwise create a new DataFrame
# if os.path.exists(output_csv_path):
#     df_labels = pd.read_csv(output_csv_path)
# else:
#     df_labels = pd.DataFrame(columns=['filename', 'label'])

file_numbers = 50

for i in range(1, file_numbers+1):
    file_name = f"imu_baseline_2_imu_train_data_part{i}.csv"
    full_path = os.path.join(input_folder, file_name)
    df = pd.read_csv(full_path)
    temp_acc_max, temp_acc_min = 0, 0
    temp_gyr_max, temp_gyr_min = 0, 0
    temp_Mag_max, temp_Mag_min = 0, 0
    for j in range(1, 4):
        max_acc, min_acc = get_imu_max_min(f'IMU{j}_Acc', df)
        max_gyr, min_gyr = get_imu_max_min(f'IMU{j}_Gyr', df)
        max_Mag, min_Mag = get_imu_max_min(f'IMU{j}_Mag', df)
        if max_acc > temp_acc_max:
            temp_acc_max = max_acc
        if min_acc < temp_acc_min:
            temp_acc_min = min_acc
        if max_gyr > temp_gyr_max:
            temp_gyr_max = max_gyr
        if min_gyr < temp_gyr_min:
            temp_gyr_min = min_gyr
        if max_Mag > temp_Mag_max:
            temp_Mag_max = max_Mag
        if min_Mag < temp_Mag_min:
            temp_Mag_min = min_Mag
            
print(temp_acc_max, temp_acc_min, temp_gyr_max, temp_gyr_min, temp_Mag_max, temp_Mag_min)
range_acc = temp_acc_max - temp_acc_min
range_gyr = temp_gyr_max - temp_gyr_min
range_Mag = temp_Mag_max - temp_Mag_min
print("range", range_acc, range_gyr, range_Mag)
    
for i in range(1, file_numbers+1):
    file_name = f"imu_baseline_2_imu_train_data_part{i}.csv"
    full_path = os.path.join(input_folder, file_name)
    # normal_path = os.path.join(ori_file)
    image_filename = process_and_save_image(full_path, output_folder, temp_acc_max, temp_acc_min, temp_gyr_max, temp_gyr_min, temp_Mag_max, temp_Mag_min,
                                            range_acc, range_gyr, range_Mag)
    # df_labels = df_labels.append({'filename': image_filename, 'label': 'seat_over'}, ignore_index=True)


# Save the updated labels to CSV, without the index
# df_labels.to_csv(output_csv_path, index=False)