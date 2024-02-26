import pandas as pd

def sliding_window(data, window_size, overlap):
    windows = []
    step_size = window_size - overlap
    for i in range(0, len(data), step_size):
        window = data[i:i + window_size]
        if len(window) == window_size:
            windows.append(window)
    return windows

def normalize_column(column, min_val, max_val):
    # Normalize the values in the column based on min and max values
    return (column - min_val) / (max_val - min_val)

def split_csv_file(filename, path_ori, path, window_size, overlap, label, column_names):
    # Load the original CSV file
    df = pd.read_csv(path_ori + filename)
    
    # Set the sliding window parameters
    window_size = 150  # Number of rows per window
    overlap = 100      # Number of rows to overlap
    
    # Normalize the data
    for col in df.columns:
        if 'Acc' in col:
            df[col] = normalize_column(df[col], -1000, 1000)
        elif 'Gyr' in col:
            df[col] = normalize_column(df[col], -1000, 1000)
    
    # Apply sliding window to the dataframe
    windowed_data = sliding_window(df, window_size, overlap)
    
    # Update column names
    df.columns = column_names
    
    # Calculate the number of files to be generated based on sliding window output
    num_files = len(windowed_data)
    
    # Iterate over each window and save as a new CSV file
    for i, window_df in enumerate(windowed_data):
        new_filename = f"{path}/{label}/{label}_{filename[:14]}_part{i+1}.csv"
        window_df.to_csv(new_filename, index=False)
        print(f"Saved: {new_filename}")

# Directory and file settings
directory_ori = "../train_data_ori/"
directory = "../train_data/"
filename = "imu_train_data_reach_under_sampled.csv"
label = "reach_under"
column_names = ['IMU1_AccX', 'IMU1_AccY', 'IMU1_AccZ', 'IMU1_GyrX', 'IMU1_GyrY', 'IMU1_GyrZ', 'IMU1_MagX', 'IMU1_MagY', 'IMU1_MagZ', 
                'IMU2_AccX', 'IMU2_AccY', 'IMU2_AccZ', 'IMU2_GyrX', 'IMU2_GyrY', 'IMU2_GyrZ', 'IMU2_MagX', 'IMU2_MagY', 'IMU2_MagZ', 
                'IMU3_AccX', 'IMU3_AccY', 'IMU3_AccZ', 'IMU3_GyrX', 'IMU3_GyrY', 'IMU3_GyrZ', 'IMU3_MagX', 'IMU3_MagY', 'IMU3_MagZ']

# Execute function
split_csv_file(filename, directory_ori, directory, 150, 100, label, column_names)
