import os
import pandas as pd

data_root = 'E:/master-2/madgewick_filter/test_data/test_reach/IMU'
data_info = []

for label in os.listdir(data_root):
    folder_path = os.path.join(data_root, label)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            data_info.append([file_path, label])

df_data_info = pd.DataFrame(data_info, columns=['FilePath', 'Label'])

output_csv_path = 'E:/master-2/madgewick_filter/test_data/test_reach/IMU/output_imu_labels.csv'
df_data_info.to_csv(output_csv_path, index=False)

print(f'Data and labels info saved to {output_csv_path}')
