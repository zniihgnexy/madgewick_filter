import numpy as np
import pandas as pd
from imu_filter import imu_filter
from numpy import arctan2, row_stack
import pandas as pd
from comlplimentary import Complimentary
import numpy as np
import pandas as pd
from butterworth import butter_lowpass_filter
import matplotlib.pyplot as plt
from madgwick_filter import MadgwickFilter

def complimentary_filter(compl_filter_imu, compl_data_imu, roll, pitch, yaw, gx, gy, gz, DELTA_T):
    measuredRoll = np.degrees(np.arctan2(ay, az))
    measuredPitch = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2)))
    measuredYaw = 0
    
    compl_filter_imu.updateRollPitchYaw(measuredRoll, measuredPitch, measuredYaw, gx, gy, gz, DELTA_T)
    roll, pitch, yaw = compl_filter_imu.roll, compl_filter_imu.pitch, compl_filter_imu.yaw
    compl_data_imu.append([roll, pitch, yaw])
    return compl_filter_imu.roll, compl_filter_imu.pitch, compl_filter_imu.yaw

def return_array_filter(array, cutoff, fs, order):
    array = np.array(array)
    array_filtered = butter_lowpass_filter(array, cutoff, fs, order)
    
    return array_filtered

def combine3_save_to_csv(timestamps, data_x, data_y, data_z, file_name):
    df = pd.DataFrame({'timestamps': timestamps, 'x': data_x, 'y': data_y, 'z': data_z})
    df.to_csv(file_name, index=False)

def save_to_csv(timestamps, data, file_name, columns):
    df = pd.DataFrame(data, columns=columns)
    df = pd.concat([timestamps, df], axis=1)
    df.to_csv(file_name, index=False)

def all_data_filter(x_1, y_1, z_1, x_2, y_2, z_2, x_3, y_3, z_3, cutoff, fs, order, name):
    x_1 = return_array_filter(x_1, cutoff, fs, order)
    y_1 = return_array_filter(y_1, cutoff, fs, order)
    z_1 = return_array_filter(z_1, cutoff, fs, order)
    
    x_2 = return_array_filter(x_2, cutoff, fs, order)
    y_2 = return_array_filter(y_2, cutoff, fs, order)
    z_2 = return_array_filter(z_2, cutoff, fs, order)
    
    x_3 = return_array_filter(x_3, cutoff, fs, order)
    y_3 = return_array_filter(y_3, cutoff, fs, order)
    z_3 = return_array_filter(z_3, cutoff, fs, order)
    
    combine3_save_to_csv(timestamps, x_1, y_1, z_1, '../butterworth_filtered/' + name + '_imu1.csv')
    combine3_save_to_csv(timestamps, x_2, y_2, z_2, '../butterworth_filtered/' + name + '_imu2.csv')
    combine3_save_to_csv(timestamps, x_3, y_3, z_3, '../butterworth_filtered/' + name + '_imu3.csv')
    
    return x_1, y_1, z_1, x_2, y_2, z_2, x_3, y_3, z_3
    

def imu_position_update(ax, ay, az, ax_prev, ay_prev, az_prev, imu_positions, dt, original_position):
    # 将加速度从mg转换为m/s^2，注意1g = 9.81m/s^2
    ax_m_s2 = ax * 9.81 / 1000
    ay_m_s2 = ay * 9.81 / 1000
    az_m_s2 = az * 9.81 / 1000

    if not imu_positions:
        new_position = [0, 0, 0]
    else:
        last_position = imu_positions[-1]
        
        delta_x = 0.5 * (ax_m_s2 + ax_prev * 9.81 / 1000) * dt**2
        delta_y = 0.5 * (ay_m_s2 + ay_prev * 9.81 / 1000) * dt**2
        delta_z = 0.5 * (az_m_s2 + az_prev * 9.81 / 1000) * dt**2

        new_position = [
            last_position[0] + delta_x,
            last_position[1] + delta_y,
            last_position[2] + delta_z
        ]

    return new_position

def speed_update(ax, ay, az, ax_prev, ay_prev, az_prev, imu_speeds, dt, original_speed):
    # 将加速度从mg转换为m/s^2，注意1g = 9.81m/s^2
    ax_m_s2 = ax * 9.81 / 1000
    ay_m_s2 = ay * 9.81 / 1000
    az_m_s2 = az * 9.81 / 1000

    if not imu_speeds:
        new_speed = [0, 0, 0]
    else:
        last_speed = imu_speeds[-1]
        
        delta_x = 0.5 * (ax_m_s2 + ax_prev * 9.81 / 1000) * dt ** 2
        delta_y = 0.5 * (ay_m_s2 + ay_prev * 9.81 / 1000) * dt ** 2
        delta_z = 0.5 * (az_m_s2 + az_prev * 9.81 / 1000) * dt ** 2

        new_speed = [
            last_speed[0] + delta_x,
            last_speed[1] + delta_y,
            last_speed[2] + delta_z
        ]

    return new_speed

if __name__ == "__main__":
    file_path = '../data/Rec28.csv'
    
    # import pdb; pdb.set_trace()
    
    data_all = pd.read_csv(file_path, skiprows=1, usecols=lambda column: column not in ['CH1', 'CH2', 'CH3'])
    data = data_all.drop(columns=['Timestamp (uS)'])
    # print(data.columns)
    # print(data.head())
    data = data.drop(columns=['Unnamed: 31'])
    # print(data.columns)

    imu_columns = [
        'AccX(mg)', 'AccY(mg)', 'AccZ(mg)', 
        'GyrX(DPS)', 'GyrY(DPS)', 'GyrZ(DPS)', 
        'MagX(uT)', 'MagY(uT)', 'MagZ(uT)',
        'AccX(mg).1', 'AccY(mg).1', 'AccZ(mg).1', 
        'GyrX(DPS).1', 'GyrY(DPS).1', 'GyrZ(DPS).1', 
        'MagX(uT).1', 'MagY(uT).1', 'MagZ(uT).1',
        'AccX(mg).2', 'AccY(mg).2', 'AccZ(mg).2', 
        'GyrX(DPS).2', 'GyrY(DPS).2', 'GyrZ(DPS).2', 
        'MagX(uT).2', 'MagY(uT).2', 'MagZ(uT).2'
    ]
    
    # print("read specific data",data.columns)
    timestamps = data_all['Timestamp (uS)']
    # print("read timestamps",timestamps.head())

    imu1 = data.iloc[:, :9]
    imu1.columns = imu_columns[:9]
    # print("read imu1 data",imu1.columns)
    
    imu2 = data.iloc[:, 9:18]
    imu2.columns = imu_columns[9:18]
    # print("read imu2 data",imu2.columns)
    
    imu3 = data.iloc[:, 18:27]
    imu3.columns = imu_columns[18:27]
    # print("read imu3 data",imu3.columns)
    
    # print(imu1.columns)
    # print(imu2.columns)
    # print(imu3.columns)
    
    # set original position of three IMUs to a certain 3D point
    imu1_initial_position = [0, 0, 0]
    imu2_initial_position = [0.5, 0.5, 0.5]
    imu3_initial_position = [1, 1, 1]
    imu1_positions = [imu1_initial_position]
    imu2_positions = [imu2_initial_position]
    imu3_positions = [imu3_initial_position]
    
    filter_imu1 = MadgwickFilter()
    filter_imu2 = MadgwickFilter()
    filter_imu3 = MadgwickFilter()
    
    compl_filter_imu1 = Complimentary(gain=0.50) # adjust gain value
    compl_filter_imu2 = Complimentary(gain=0.50)
    compl_filter_imu3 = Complimentary(gain=0.50)
    compl_filter_imu = Complimentary(gain=0.50)

    DELTA_T = 1/100.0 # 100Hz, change to the correct one
    PI = np.pi
    cutoff = 20
    fs = 100
    order = 6
    
    ax_list1, ay_list1, az_list1 = [], [], []
    gx_list1, gy_list1, gz_list1 = [], [], []
    mx_list1, my_list1, mz_list1 = [], [], []
    speedx_list1, speedy_list1, speedz_list1 = [], [], []
    
    acc_raw_data_imu1 = []
    gyr_raw_data_imu1 = []
    mag_raw_data_imu1 = []
    angles_data_imu1 = []
    compl_data_imu1 = []
    imu1_train_data = []
    
    ax_list2, ay_list2, az_list2 = [], [], []
    gx_list2, gy_list2, gz_list2 = [], [], []
    mx_list2, my_list2, mz_list2 = [], [], []
    speedx_list2, speedy_list2, speedz_list2 = [], [], []

    acc_raw_data_imu2 = []
    gyr_raw_data_imu2 = []
    mag_raw_data_imu2 = []
    angles_data_imu2 = []
    compl_data_imu2 = []
    imu2_train_data = []
    
    ax_list3, ay_list3, az_list3 = [], [], []
    gx_list3, gy_list3, gz_list3 = [], [], []
    mx_list3, my_list3, mz_list3 = [], [], []
    speedx_list3, speedy_list3, speedz_list3 = [], [], []

    acc_raw_data_imu3 = []
    gyr_raw_data_imu3 = []
    mag_raw_data_imu3 = []
    angles_data_imu3 = []
    compl_data_imu3 = []
    imu3_train_data = []

    print("\nIMU 1")
    for index, row in imu1.iterrows():
        ax_list1.append(row['AccX(mg)'])
        ay_list1.append(row['AccY(mg)'])
        az_list1.append(row['AccZ(mg)'])
        gx_list1.append(row['GyrX(DPS)'])
        gy_list1.append(row['GyrY(DPS)'])
        gz_list1.append(row['GyrZ(DPS)'])
        mx_list1.append(row['MagX(uT)'])
        my_list1.append(row['MagY(uT)'])
        mz_list1.append(row['MagZ(uT)'])
        
        # previous time setp's data
        ax_prev = ax_list1[-1] if len(ax_list1) > 1 else 0
        ay_prev = ay_list1[-1] if len(ay_list1) > 1 else 0
        az_prev = az_list1[-1] if len(az_list1) > 1 else 0
        
        speedx_prev = speedx_list1[-1] if len(speedx_list1) > 1 else 0
        speedy_prev = speedy_list1[-1] if len(speedy_list1) > 1 else 0
        speedz_prev = speedz_list1[-1] if len(speedz_list1) > 1 else 0
        
        speedx_list1.append(speed_update(row['AccX(mg)'], row['AccY(mg)'], row['AccZ(mg)'], ax_list1[-1], ay_list1[-1], az_list1[-1], imu1_positions, DELTA_T, imu1_initial_position))
        speedy_list1.append(speed_update(row['AccY(mg)'], row['AccX(mg)'], row['AccZ(mg)'], ay_list1[-1], ax_list1[-1], az_list1[-1], imu1_positions, DELTA_T, imu1_initial_position))
        speedz_list1.append(speed_update(row['AccZ(mg)'], row['AccY(mg)'], row['AccX(mg)'], az_list1[-1], ay_list1[-1], ax_list1[-1], imu1_positions, DELTA_T, imu1_initial_position))
        
        # import pdb; pdb.set_trace()
        imu_data = filter_imu1.imu_filter(
            ax=row['AccX(mg)'], ay=row['AccY(mg)'], az=row['AccZ(mg)'],
            gx=row['GyrX(DPS)'], gy=row['GyrY(DPS)'], gz=row['GyrZ(DPS)'],
            mx=row['MagX(uT)'], my=row['MagY(uT)'], mz=row['MagZ(uT)'],
        )
        ax, ay, az = row['AccX(mg)'], row['AccY(mg)'], row['AccZ(mg)']
        gx, gy, gz = np.radians([row['GyrX(DPS)'], row['GyrY(DPS)'], row['GyrZ(DPS)']])
        mx, my, mz = row['MagX(uT)'], row['MagY(uT)'], row['MagZ(uT)']
        
        acc_raw_data_imu1.append([ax, ay, az])
        gyr_raw_data_imu1.append([gx, gy, gz])
        mag_raw_data_imu1.append([mx, my, mz])
        
        roll, pitch, yaw = filter_imu1.eulerAngles()
        angles_data_imu1.append([roll, pitch, yaw])
        # print(f"IMU 1 - Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        compl_filter_imu1.roll, compl_filter_imu1.pitch, compl_filter_imu1.yaw = complimentary_filter(compl_filter_imu1, compl_data_imu1, roll, pitch, yaw, gx, gy, gz, DELTA_T)
        
        imu1_position = imu_position_update(ax, ay, az, ax_prev, ay_prev, az_prev, imu1_positions, DELTA_T, imu1_initial_position)
        imu1_positions.append(imu1_position)
        # print(f"IMU 1 - Position: {imu1_positions}")
        
        # put acc and gyr data into a list
        imu1_train_data.append([ax, ay, az, gx, gy, gz])

    print("\nIMU 2")
    for index, row in imu2.iterrows():
        ax_list2.append(row['AccX(mg).1'])
        ay_list2.append(row['AccY(mg).1'])
        az_list2.append(row['AccZ(mg).1'])
        gx_list2.append(row['GyrX(DPS).1'])
        gy_list2.append(row['GyrY(DPS).1'])
        gz_list2.append(row['GyrZ(DPS).1'])
        mx_list2.append(row['MagX(uT).1'])
        my_list2.append(row['MagY(uT).1'])
        mz_list2.append(row['MagZ(uT).1'])
        # previous time setp's data
        ax_prev = ax_list2[-1] if len(ax_list2) > 1 else 0
        ay_prev = ay_list2[-1] if len(ay_list2) > 1 else 0
        az_prev = az_list2[-1] if len(az_list2) > 1 else 0
        
        speedx_prev = speedx_list2[-1] if len(speedx_list2) > 1 else 0
        speedy_prev = speedy_list2[-1] if len(speedy_list2) > 1 else 0
        speedz_prev = speedz_list2[-1] if len(speedz_list2) > 1 else 0
        
        speedx_list2 = speed_update(row['AccX(mg).1'], row['AccY(mg).1'], row['AccZ(mg).1'], ax_list2[-1], ay_list2[-1], az_list2[-1], imu2_positions, DELTA_T, imu2_initial_position)
        speedy_list2 = speed_update(row['AccY(mg).1'], row['AccX(mg).1'], row['AccZ(mg).1'], ay_list2[-1], ax_list2[-1], az_list2[-1], imu2_positions, DELTA_T, imu2_initial_position)
        speedz_list2 = speed_update(row['AccZ(mg).1'], row['AccY(mg).1'], row['AccX(mg).1'], az_list2[-1], ay_list2[-1], ax_list2[-1], imu2_positions, DELTA_T, imu2_initial_position)
        
        imu_data = filter_imu2.imu_filter(
            ax=row['AccX(mg).1'], ay=row['AccY(mg).1'], az=row['AccZ(mg).1'],
            gx=row['GyrX(DPS).1'], gy=row['GyrY(DPS).1'], gz=row['GyrZ(DPS).1'],
            mx=row['MagX(uT).1'], my=row['MagY(uT).1'], mz=row['MagZ(uT).1'],
        )
        ax, ay, az = row['AccX(mg).1'], row['AccY(mg).1'], row['AccZ(mg).1']
        gx, gy, gz = np.radians([row['GyrX(DPS).1'], row['GyrY(DPS).1'], row['GyrZ(DPS).1']])
        mx, my, mz = row['MagX(uT).1'], row['MagY(uT).1'], row['MagZ(uT).1']
        
        acc_raw_data_imu2.append([ax, ay, az])
        gyr_raw_data_imu2.append([gx, gy, gz])
        mag_raw_data_imu2.append([mx, my, mz])
        
        roll, pitch, yaw = filter_imu2.eulerAngles()
        angles_data_imu2.append([roll, pitch, yaw])
        # print(f"IMU 2 - Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")

        compl_filter_imu2.roll, compl_filter_imu2.pitch, compl_filter_imu2.yaw = complimentary_filter(compl_filter_imu2, compl_data_imu2, roll, pitch, yaw, gx, gy, gz, DELTA_T)
        
        # import pdb; pdb.set_trace()
        imu2_position = imu_position_update(ax, ay, az, ax_prev, ay_prev, az_prev, imu2_positions, DELTA_T, imu2_initial_position)
        imu2_positions.append(imu2_position)
        # print(f"IMU 2 - Position: {imu2_positions}")
        
        imu2_train_data.append([ax, ay, az, gx, gy, gz])

    print("\nIMU 3")
    for index, row in imu3.iterrows():
        ax_list3.append(row['AccX(mg).2'])
        ay_list3.append(row['AccY(mg).2'])
        az_list3.append(row['AccZ(mg).2'])
        gx_list3.append(row['GyrX(DPS).2'])
        gy_list3.append(row['GyrY(DPS).2'])
        gz_list3.append(row['GyrZ(DPS).2'])
        mx_list3.append(row['MagX(uT).2'])
        my_list3.append(row['MagY(uT).2'])
        mz_list3.append(row['MagZ(uT).2'])
        # previous time setp's data
        ax_prev = ax_list3[-1] if len(ax_list3) > 1 else 0
        ay_prev = ay_list3[-1] if len(ay_list3) > 1 else 0
        az_prev = az_list3[-1] if len(az_list3) > 1 else 0
        
        speedx_prev = speedx_list3[-1] if len(speedx_list3) > 1 else 0
        speedy_prev = speedy_list3[-1] if len(speedy_list3) > 1 else 0
        speedz_prev = speedz_list3[-1] if len(speedz_list3) > 1 else 0
        
        speedx_list3 = speed_update(row['AccX(mg).2'], row['AccY(mg).2'], row['AccZ(mg).2'], ax_list3[-1], ay_list3[-1], az_list3[-1], imu3_positions, DELTA_T, imu3_initial_position)
        speedy_list3 = speed_update(row['AccY(mg).2'], row['AccX(mg).2'], row['AccZ(mg).2'], ay_list3[-1], ax_list3[-1], az_list3[-1], imu3_positions, DELTA_T, imu3_initial_position)
        speedz_list3 = speed_update(row['AccZ(mg).2'], row['AccY(mg).2'], row['AccX(mg).2'], az_list3[-1], ay_list3[-1], ax_list3[-1], imu3_positions, DELTA_T, imu3_initial_position)
        
        imu_data = filter_imu3.imu_filter(
            ax=row['AccX(mg).2'], ay=row['AccY(mg).2'], az=row['AccZ(mg).2'],
            gx=row['GyrX(DPS).2'], gy=row['GyrY(DPS).2'], gz=row['GyrZ(DPS).2'],
            mx=row['MagX(uT).2'], my=row['MagY(uT).2'], mz=row['MagZ(uT).2'],
        )
        ax, ay, az = row['AccX(mg).2'], row['AccY(mg).2'], row['AccZ(mg).2']
        gx, gy, gz = np.radians([row['GyrX(DPS).2'], row['GyrY(DPS).2'], row['GyrZ(DPS).2']])
        mx, my, mz = row['MagX(uT).2'], row['MagY(uT).2'], row['MagZ(uT).2']
        
        acc_raw_data_imu3.append([ax, ay, az])
        gyr_raw_data_imu3.append([gx, gy, gz])
        mag_raw_data_imu3.append([mx, my, mz])
        
        roll, pitch, yaw = filter_imu3.eulerAngles()
        angles_data_imu3.append([roll, pitch, yaw])
        # print(f"IMU 3 - Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        
        compl_filter_imu3.roll, compl_filter_imu3.pitch, compl_filter_imu3.yaw = complimentary_filter(compl_filter_imu3, compl_data_imu3, roll, pitch, yaw, gx, gy, gz, DELTA_T)
        
        imu3_position = imu_position_update(ax, ay, az, ax_prev, ay_prev, az_prev, imu3_positions, DELTA_T, imu3_initial_position)
        imu3_positions.append(imu3_position)
        # print(f"IMU 3 - Position: {imu3_positions}")
        
        imu3_train_data.append([ax, ay, az, gx, gy, gz])
    
    
    ###########################################################################################
    # madgewick filter data processing
    # original
    df_imu1 = pd.DataFrame(acc_raw_data_imu1, columns=['IMU1_AccX', 'IMU1_AccY', 'IMU1_AccZ'])
    df_imu2 = pd.DataFrame(acc_raw_data_imu2, columns=['IMU2_AccX', 'IMU2_AccY', 'IMU2_AccZ'])
    df_imu3 = pd.DataFrame(acc_raw_data_imu3, columns=['IMU3_AccX', 'IMU3_AccY', 'IMU3_AccZ'])

    # angles from the filter
    angle_imu1 = pd.DataFrame(angles_data_imu1, columns=['Roll1', 'Pitch1', 'Yaw1'])
    angle_imu2 = pd.DataFrame(angles_data_imu2, columns=['Roll2', 'Pitch2', 'Yaw2'])
    angle_imu3 = pd.DataFrame(angles_data_imu3, columns=['Roll3', 'Pitch3', 'Yaw3'])
    
    # import pdb; pdb.set_trace()
    # angles from the complimentary filter
    # compl_imu1 = pd.DataFrame(compl_data_imu1, columns=['Roll1', 'Pitch1', 'Yaw1'])
    # compl_imu2 = pd.DataFrame(compl_data_imu2, columns=['Roll2', 'Pitch2', 'Yaw2'])
    # compl_imu3 = pd.DataFrame(compl_data_imu3, columns=['Roll3', 'Pitch3', 'Yaw3'])
    
    # data for train
    train_imu1 = pd.DataFrame(imu1_train_data, columns=['IMU1_AccX', 'IMU1_AccY', 'IMU1_AccZ', 'IMU1_GyrX', 'IMU1_GyrY', 'IMU1_GyrZ'])
    train_imu2 = pd.DataFrame(imu2_train_data, columns=['IMU2_AccX', 'IMU2_AccY', 'IMU2_AccZ', 'IMU2_GyrX', 'IMU2_GyrY', 'IMU2_GyrZ'])
    train_imu3 = pd.DataFrame(imu3_train_data, columns=['IMU3_AccX', 'IMU3_AccY', 'IMU3_AccZ', 'IMU3_GyrX', 'IMU3_GyrY', 'IMU3_GyrZ'])
    
    # 将三个DataFrame按列合并
    Acc_final_df = pd.concat([timestamps, df_imu1, df_imu2, df_imu3], axis=1)
    Angle_final_df = pd.concat([timestamps, angle_imu1, angle_imu2, angle_imu3], axis=1)
    # compl_final_df = pd.concat([timestamps, compl_imu1, compl_imu2, compl_imu3], axis=1)
    Train_imu1_final_df = pd.concat([timestamps, train_imu1], axis=1)
    Train_imu2_final_df = pd.concat([timestamps, train_imu2], axis=1)
    Train_imu3_final_df = pd.concat([timestamps, train_imu3], axis=1)
    
    # save to csv file
    Acc_final_df.to_csv('../raw_data/imu_Acc_data.csv', index=False)
    Angle_final_df.to_csv('../angles_data/angle_state_imu_angle_data.csv', index=False)
    # compl_final_df.to_csv('compl_imu_Compl_angle_data.csv', index=False)
    
    Train_imu1_final_df.to_csv('../train_data/imu1_train_data.csv', index=False)
    Train_imu2_final_df.to_csv('../train_data/imu2_train_data.csv', index=False)
    Train_imu3_final_df.to_csv('../train_data/imu3_train_data.csv', index=False)
    ###########################################################################################
    # butter worth filter part
    all_data_filter(ax_list1, ay_list1, az_list1, ax_list2, ay_list2, az_list2, ax_list3, ay_list3, az_list3, cutoff, fs, order, name='acc')
    all_data_filter(gx_list1, gy_list1, gz_list1, gx_list2, gy_list2, gz_list2, gx_list3, gy_list3, gz_list3, cutoff, fs, order, name='gyr')
    all_data_filter(mx_list1, my_list1, mz_list1, mx_list2, my_list2, mz_list2, mx_list3, my_list3, mz_list3, cutoff, fs, order, name='mag')
    ################################################################################################
    # position update
    save_to_csv(timestamps, imu1_positions, '../position/IMU1_position.csv', columns=['IMU1_X', 'IMU1_Y', 'IMU1_Z'])
    save_to_csv(timestamps, imu2_positions, '../position/IMU2_position.csv', columns=['IMU2_X', 'IMU2_Y', 'IMU2_Z'])
    save_to_csv(timestamps, imu3_positions, '../position/IMU3_position.csv', columns=['IMU3_X', 'IMU3_Y', 'IMU3_Z'])

    