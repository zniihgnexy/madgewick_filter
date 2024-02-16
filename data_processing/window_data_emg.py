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

data_temp = pd.read_csv('../train_data_ori/emg_train_data.csv')
data_temp = data_temp.drop(columns=['Timestamp (uS)'])
data_temp.to_csv('../train_data_ori/emg_train_data1.csv', index=False)

directory_ori = "../train_data_ori/"
directory = "../train_data/emg/"
filename = "emg_train_data1.csv"
rows_per_file = 400
label = "random"
column_names = ['CH1', 'CH2', 'CH3']
split_csv_file(filename, directory_ori, directory, rows_per_file, label, column_names)
