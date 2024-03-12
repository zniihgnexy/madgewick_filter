import os
import pandas as pd

data_roots = {
    'E:/master-2/madgewick_filter/train_data_imu_pic': 'train_imu_labels.csv',
    'E:/master-2/madgewick_filter/test_data_imu_pic': 'test_imu_labels.csv'
}

# data_roots = {
#     'E:/master-2/madgewick_filter/SplitEMG_train_data_20240312': 'train_emg_labels.csv',
#     'E:/master-2/madgewick_filter/SplitEMG_test_data_20240312': 'test_emg_labels.csv'
# }

for data_root, label_file_name in data_roots.items():
    data_info = []
    for label in os.listdir(data_root):
        folder_path = os.path.join(data_root, label)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                data_info.append([file_path, label])

    df_data_info = pd.DataFrame(data_info, columns=['FilePath', 'Label'])

    output_csv_path = os.path.join(data_root, label_file_name)
    df_data_info.to_csv(output_csv_path, index=False)

    print(f'Data and labels info saved to {output_csv_path}')
