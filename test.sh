# for windowing data
JSON_FILE_PATH="E:/master-2/madgewick_filter/data_processing/file_path/windowing_data.json"
DIRECTORY_ORI="E:/master-2/madgewick_filter/detection_data/raw_data/"
WINDOW_DATA_DIRECTORY="E:/master-2/madgewick_filter/detection_data/"
WINDOW_SIZE=150
OVERLAP=75

# for generating graphs
GRAPH_OUTPUT_DIRECTORY_TRAIN="E:/master-2/madgewick_filter/detection_data/"
GRAPH_OUTPUT_DIRECTORY_TEST="E:/master-2/madgewick_filter/detection_data/"

# for testing
TEST_IMU_DATA_DIR="E:/master-2/madgewick_filter/detection_data/imu_data/"
TEST_EMG_DATA_DIR="E:/master-2/madgewick_filter/SplitEMG_test_data_20240312/"


# Execute Python scripts
echo "Starting windowing process..."
python E:\\master-2\\madgewick_filter\\testing_model\\windowing_data_imu.py $JSON_FILE_PATH $DIRECTORY_ORI $WINDOW_DATA_DIRECTORY $WINDOW_SIZE $OVERLAP

echo "Generating graphs from windowed data..."
python E:/master-2/madgewick_filter/testing_model/graph_imu.py $WINDOW_DATA_DIRECTORY $GRAPH_OUTPUT_DIRECTORY_TRAIN $GRAPH_OUTPUT_DIRECTORY_TEST

# echo "Running tests..."
# python E:/master-2/madgewick_filter/testing_model/test_combine.py $TEST_IMU_DATA_DIR $TEST_EMG_DATA_DIR

echo "All processes completed successfully."
