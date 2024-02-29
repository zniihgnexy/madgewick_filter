import os
import pandas as pd

data_root = 'E:/master-2/madgewick_filter/train_data/train_reach'
data_root2 = 'E:/master-2/madgewick_filter/train_data/EMG'
data_info = []
data_info2 = []

for label in os.listdir(data_root):
    folder_path = os.path.join(data_root, label)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            data_info.append([file_path, label])

df_data_info = pd.DataFrame(data_info, columns=['FilePath', 'Label'])

output_csv_path = 'E:/master-2/madgewick_filter/train_data/train_reach/output_imu_labels.csv'
df_data_info.to_csv(output_csv_path, index=False)

# for label in os.listdir(data_root2):
#     folder_path2 = os.path.join(data_root2, label)
#     if os.path.isdir(folder_path2):
#         for file in os.listdir(folder_path2):
#             file_path2 = os.path.join(folder_path2, file)
#             data_info2.append([file_path2, label])

# df_data_info2 = pd.DataFrame(data_info2, columns=['FilePath', 'Label'])

# output_csv_path2 = 'E:/master-2/madgewick_filter/train_data/output_emg_data_info.csv'
# df_data_info2.to_csv(output_csv_path2, index=False)

print(f'Data and labels info saved to {output_csv_path}')
