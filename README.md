# AML lab

### environment

tensorflow uninstall older version

```
pip uninstall tensorflow
pip uninstall tensorflow-intel
pip uninstall tensorflow-gpu 
pip install tensorflow=2.10.1
```

cuda 11.8

numpy > 3.0

### data processing

```
python pre_processing.py
python visulization.py
python window_data.py
```

#### *pre_processing.py*

file_path: original data file

butterworth_filtered folder: all the data after filter, 9 files in total

angles_data folder: after madgwick filter data, roll pitvch yaw angles

train_data_ori folder: acc and gyr data from one imu to one file

position folder: visulization 3D position of 3 imus, one imu data in one file

#### *visulization.py*

read from position folder, output graph called IMU_position.png

#### *window_data.py*

directory_ori: read from this folder

directory: output to this folder

filename: file name for the data

rows_per_file: how many lines want to keep in one file, the file number is depend on this

label: what is this data, use this as the name of output files

column_names: output column names

### training_model

```
python training_prepare.py
```

input_shape: flattern ont file and the element number is the input number

directory: the data of certain lable, folder name is the label name

### *model.py*

change the model here
