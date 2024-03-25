import subprocess

#imu pre processing paths
raw_data_seat_over = 'E:/master-2/madgewick_filter/data/Rec16.csv'
save_data_seat_over = 'E:/master-2/madgewick_filter/train_data_ori/sample/imu_seat_over.csv'
save_sampled_seat_over = 'E:/master-2/madgewick_filter/detection_data/sample/imu_seat_over_sampled.csv'

raw_data_baseline = 'E:/master-2/madgewick_filter/data/Rec1.csv'
save_data_baseline = 'E:/master-2/madgewick_filter/train_data_ori/sample/imu_baseline.csv'
save_sampled_baseline = 'E:/master-2/madgewick_filter/detection_data/sample/imu_baseline_sampled.csv'

raw_data_reach_under = 'E:/master-2/madgewick_filter/data/Rec3.csv'
save_data_reach_under = 'E:/master-2/madgewick_filter/train_data_ori/sample/imu_reach_under.csv'
save_sampled_reach_under = 'E:/master-2/madgewick_filter/detection_data/sample/imu_reach_under_sampled.csv'

raw_data_reach_over = 'E:/master-2/madgewick_filter/data/Rec4.csv'
save_data_reach_over = 'E:/master-2/madgewick_filter/train_data_ori/sample/imu_reach_over.csv'
save_sampled_reach_over = 'E:/master-2/madgewick_filter/detection_data/sample/imu_reach_over_sampled.csv'

raw_data_drop_over = 'E:/master-2/madgewick_filter/data/Rec6.csv'
save_data_drop_over = 'E:/master-2/madgewick_filter/train_data_ori/sample/imu_drop_over.csv'
save_sampled_drop_over = 'E:/master-2/madgewick_filter/detection_data/sample/imu_drop_over_sampled.csv'

raw_data_drop_under = 'E:/master-2/madgewick_filter/data/Rec13.csv'
save_data_drop_under = 'E:/master-2/madgewick_filter/train_data_ori/sample/imu_drop_under.csv'
save_sampled_drop_under = 'E:/master-2/madgewick_filter/detection_data/sample/imu_drop_under_sampled.csv'

raw_data_seat_under = 'E:/master-2/madgewick_filter/data/Rec15.csv'
save_data_seat_under = 'E:/master-2/madgewick_filter/train_data_ori/sample/imu_seat_under.csv'
save_sampled_seat_under = 'E:/master-2/madgewick_filter/detection_data/sample/imu_seat_under_sampled.csv'

# Set parameters for windowing data
json_file_path = "E:/master-2/madgewick_filter/data_processing/file_path/windowing_data_test.json"
directory_ori = "E:/master-2/madgewick_filter/detection_data/raw_data/"
window_data_directory = "E:/master-2/madgewick_filter/detection_data/sampled_data/"
window_size = "150"
overlap = "50"

# Set parameters for generating graphs
graph_output_directory_train = "E:/master-2/madgewick_filter/detection_data/imu_pic/"
graph_output_directory_test = "E:/master-2/madgewick_filter/detection_data/imu_pic/"

# generate labels
data_roots_1 = "E:/master-2/madgewick_filter/detection_data/imu_pic"
data_roots_name_1 = "test_imu_labels.csv"
data_roots_2 = "E:/master-2/madgewick_filter/SplitEMG_test_data_20240312"
data_roots_name_2 = "test_emg_labels.csv"

# Set parameters for testing
test_imu_data_dir = "E:/master-2/madgewick_filter/detection_data/imu_pic/"
test_emg_data_dir = "E:/master-2/madgewick_filter/SplitEMG_test_data_20240312/"
imu_labels_csv_path = "E:/master-2/madgewick_filter/detection_data/imu_pic/test_imu_labels.csv"
emg_labels_csv_path = "E:/master-2/madgewick_filter/SplitEMG_test_data_20240312/test_emg_labels.csv"

# Paths to Python scripts
sample_raw_data_path = "E:/master-2/madgewick_filter/data_processing/imu_pre_processing.py"
windowing_script_path = "E:/master-2/madgewick_filter/data_processing/window_data_imu.py"
graph_script_path = "E:/master-2/madgewick_filter/data_processing/graph_imu.py"
test_script_path = "E:/master-2/madgewick_filter/testing_model/posture_detection.py"
get_labels_script_path = "E:/master-2/madgewick_filter/data_processing/get_labels.py"

# start to sample the raw data
print("Starting sampling raw data process...")
subprocess.run(["python", sample_raw_data_path, raw_data_seat_over, save_data_seat_over, save_sampled_seat_over], check=True)
subprocess.run(["python", sample_raw_data_path, raw_data_baseline, save_data_baseline, save_sampled_baseline], check=True)
subprocess.run(["python", sample_raw_data_path, raw_data_reach_under, save_data_reach_under, save_sampled_reach_under], check=True)
subprocess.run(["python", sample_raw_data_path, raw_data_reach_over, save_data_reach_over, save_sampled_reach_over], check=True)
subprocess.run(["python", sample_raw_data_path, raw_data_drop_over, save_data_drop_over, save_sampled_drop_over], check=True)
subprocess.run(["python", sample_raw_data_path, raw_data_drop_under, save_data_drop_under, save_sampled_drop_under], check=True)
subprocess.run(["python", sample_raw_data_path, raw_data_seat_under, save_data_seat_under, save_sampled_seat_under], check=True)

# Start windowing process
print("Starting windowing process...")
subprocess.run(["python", windowing_script_path, json_file_path, directory_ori, window_data_directory, window_size, overlap], check=True)

# Generate graphs from windowed data
print("Generating graphs from windowed data...")
subprocess.run(["python", graph_script_path, window_data_directory, graph_output_directory_train, graph_output_directory_test], check=True)

# print("generating labels...")
subprocess.run(["python", get_labels_script_path, data_roots_1, data_roots_name_1, data_roots_2, data_roots_name_2], check=True)

# Uncomment the following lines if you want to include the testing process
print("Running tests...")
subprocess.run(["python", test_script_path], check=True)

print("All processes completed successfully.")
