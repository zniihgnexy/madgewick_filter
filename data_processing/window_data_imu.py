import pandas as pd

def split_csv_file(filename,path_ori, path, rows_per_file, label, column_names):
    df = pd.read_csv(directory_ori + filename)
    
    df.columns = column_names
    
    num_rows = df.shape[0]
    num_files = num_rows // rows_per_file + (1 if num_rows % rows_per_file else 0)
    
    for i in range(num_files):
        start_row = i * rows_per_file
        end_row = start_row + rows_per_file
        df_subset = df.iloc[start_row:end_row]
        
        new_filename = f"{path}/{label}/{label}_{filename[:-4]}_part{i+1}.csv"
        
        df_subset.to_csv(new_filename, index=False)
        
        print(f"Saved: {new_filename}")


directory_ori = "../train_data_ori/"
directory = "../train_data/"
filename = "imu_train_data.csv"
rows_per_file = 20
label = "random"
column_names = ['IMU1_AccX', 'IMU1_AccY', 'IMU1_AccZ', 'IMU1_GyrX', 'IMU1_GyrY', 'IMU1_GyrZ', 'IMU1_MagX', 'IMU1_MagY', 'IMU1_MagZ', 
                'IMU2_AccX', 'IMU2_AccY', 'IMU2_AccZ', 'IMU2_GyrX', 'IMU2_GyrY', 'IMU2_GyrZ', 'IMU2_MagX', 'IMU2_MagY', 'IMU2_MagZ', 
                'IMU3_AccX', 'IMU3_AccY', 'IMU3_AccZ', 'IMU3_GyrX', 'IMU3_GyrY', 'IMU3_GyrZ', 'IMU3_MagX', 'IMU3_MagY', 'IMU3_MagZ']
split_csv_file(filename, directory_ori, directory, rows_per_file, label, column_names)
