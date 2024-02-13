import numpy as np
import pandas as pd
from imu_filter import imu_filter
from numpy import arctan2, row_stack
import pandas as pd
from comlplimentary import Complimentary
import numpy as np
import pandas as pd
from butterworth import butter_lowpass_filter

class MadgwickFilter:
    def __init__(self, beta=0.1):
        # Initial quaternion estimate
        self.beta = beta
        self.q_est = np.array([1.0, 0.0, 0.0, 0.0])
    
    def imu_filter(self, ax, ay, az, gx, gy, gz, mx, my, mz):
        gx, gy, gz = np.radians([gx, gy, gz])
        self.q_est = imu_filter(ax, ay, az, gx, gy, gz)
        return ax, ay, az, gx, gy, gz, self.q_est
        
    def eulerAngles(self):
        q1, q2, q3, q4 = self.q_est
        PI = np.pi
        
        yaw = np.arctan2(2*q2*q3 - 2*q1*q4, 2*q1**2 + 2*q2**2 - 1)
        pitch = -np.arcsin(2*q2*q4 + 2*q1*q3)
        roll = np.arctan2(2*q3*q4 - 2*q1*q2, 2*q1**2 + 2*q4**2 - 1)

        yaw *= (180.0 / PI)
        pitch *= (180.0 / PI)
        roll *= (180.0 / PI)

        return roll, pitch, yaw


def eulerAngles(self):
    # 使用self.q_est计算欧拉角
    q1, q2, q3, q4 = self.q_est
    PI = np.pi

    yaw = np.arctan2(2*q2*q3 - 2*q1*q4, 2*q1**2 + 2*q2**2 - 1)
    pitch = -np.arcsin(2*q2*q4 + 2*q1*q3)
    roll = np.arctan2(2*q3*q4 - 2*q1*q2, 2*q1**2 + 2*q4**2 - 1)

    yaw *= (180.0 / PI)
    pitch *= (180.0 / PI)
    roll *= (180.0 / PI)

    return roll, pitch, yaw

if __name__ == "__main__":
    file_path = 'Rec17.csv'
    
    # import pdb; pdb.set_trace()
    
    data_all = pd.read_csv(file_path, skiprows=1, usecols=lambda column: column not in ['CH1', 'CH2', 'CH3'])
    data = data_all.drop(columns=['Timestamp (uS)'])
    print("read data 1")
    print(data.columns)
    print(data.head())
    data = data.drop(columns=['Unnamed: 31'])
    print(data.columns)

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
    
    print("read specific data",data.columns)
    
    timestamps = data_all['Timestamp (uS)']
    print("read timestamps",timestamps.head())

    # read imu1 data from the first 9 columns
    imu1 = data.iloc[:, :9]
    imu1.columns = imu_columns[:9]
    # print("read imu1 data",imu1.columns)
    
    # read imu2 data from the next 9 columns
    imu2 = data.iloc[:, 9:18]
    imu2.columns = imu_columns[9:18]
    # print("read imu2 data",imu2.columns)
    
    # read imu3 data from the last 9 columns
    imu3 = data.iloc[:, 18:27]
    imu3.columns = imu_columns[18:27]
    # print("read imu3 data",imu3.columns)
    
    print(imu1.columns)
    print(imu2.columns)
    print(imu3.columns)
    
    filter_imu1 = MadgwickFilter()
    filter_imu2 = MadgwickFilter()
    filter_imu3 = MadgwickFilter()
    
    compl_filter_imu1 = Complimentary(gain=0.50)
    compl_filter_imu2 = Complimentary(gain=0.50)
    compl_filter_imu3 = Complimentary(gain=0.50)

    DELTA_T = 1/1000.0
    PI = np.pi
    cutoff = 5
    fs = 100
    order = 6
    
    processed_data_imu1 = []
    filtered_data_imu1 = []
    compl_data_imu1 = []

    processed_data_imu2 = []
    filtered_data_imu2 = []
    compl_data_imu2 = []

    processed_data_imu3 = []
    filtered_data_imu3 = []
    compl_data_imu3 = []

    print("\nIMU 1")
    for index, row in imu1.iterrows():
        # import pdb; pdb.set_trace()
        imu_data = filter_imu1.imu_filter(
            ax=row['AccX(mg)'], ay=row['AccY(mg)'], az=row['AccZ(mg)'],
            gx=row['GyrX(DPS)'], gy=row['GyrY(DPS)'], gz=row['GyrZ(DPS)'],
            mx=row['MagX(uT)'], my=row['MagY(uT)'], mz=row['MagZ(uT)'],
        )
        processed_data_imu1.append([row['AccX(mg)'], row['AccY(mg)'], row['AccZ(mg)']])
        roll, pitch, yaw = filter_imu1.eulerAngles()
        filtered_data_imu1.append([roll, pitch, yaw])
        # print(f"IMU 1 - Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        processed_df = pd.DataFrame(processed_data_imu1, columns=['Processed_AccX', 'Processed_AccY', 'Processed_AccZ'])
        filtered_df = pd.DataFrame(filtered_data_imu1, columns=['Roll', 'Pitch', 'Yaw'])
        
        ax, ay, az = row['AccX(mg)'], row['AccY(mg)'], row['AccZ(mg)']
        gx, gy, gz = np.radians([row['GyrX(DPS)'], row['GyrY(DPS)'], row['GyrZ(DPS)']])
        
        ax_filtered = butter_lowpass_filter(ax, cutoff, fs, order)
        ay_filtered = butter_lowpass_filter(ay, cutoff, fs, order)
        az_filtered = butter_lowpass_filter(az, cutoff, fs, order)
        
        measuredRoll = np.degrees(np.arctan2(ay, az))
        measuredPitch = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2)))
        measuredYaw = 0
        
        compl_filter_imu1.updateRollPitchYaw(measuredRoll, measuredPitch, measuredYaw, gx, gy, gz, DELTA_T)
        roll, pitch, yaw = compl_filter_imu1.roll, compl_filter_imu1.pitch, compl_filter_imu1.yaw
        compl_data_imu1.append([roll, pitch, yaw])

    # 将补偿滤波器的结果保存或用于进一步处理
    filtered_data_imu1.append([roll, pitch, yaw])

    print("\nIMU 2")
    for index, row in imu2.iterrows():
        imu_data = filter_imu2.imu_filter(
            ax=row['AccX(mg).1'], ay=row['AccY(mg).1'], az=row['AccZ(mg).1'],
            gx=row['GyrX(DPS).1'], gy=row['GyrY(DPS).1'], gz=row['GyrZ(DPS).1'],
            mx=row['MagX(uT).1'], my=row['MagY(uT).1'], mz=row['MagZ(uT).1'],
        )
        # print("imu data z", ax, ay, az, gx, gy, gz)
        processed_data_imu2.append([row['AccX(mg).1'], row['AccY(mg).1'], row['AccZ(mg).1']])
        roll, pitch, yaw = filter_imu2.eulerAngles()
        filtered_data_imu2.append([roll, pitch, yaw])
        # print(f"IMU 2 - Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        processed_df = pd.DataFrame(processed_data_imu2, columns=['Processed_AccX1', 'Processed_AccY1', 'Processed_AccZ1'])
        filtered_df = pd.DataFrame(filtered_data_imu2, columns=['Roll1', 'Pitch1', 'Yaw1'])
        
        ax, ay, az = row['AccX(mg).1'], row['AccY(mg).1'], row['AccZ(mg).1']
        gx, gy, gz = np.radians([row['GyrX(DPS).1'], row['GyrY(DPS).1'], row['GyrZ(DPS).1']])
        
        measuredRoll = np.degrees(np.arctan2(ay, az))
        measuredPitch = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2)))
        measuredYaw = 0
        
        compl_filter_imu2.updateRollPitchYaw(measuredRoll, measuredPitch, measuredYaw, gx, gy, gz, DELTA_T)
        roll, pitch, yaw = compl_filter_imu2.roll, compl_filter_imu2.pitch, compl_filter_imu2.yaw
        compl_data_imu2.append([roll, pitch, yaw])
        

    print("\nIMU 3")
    for index, row in imu3.iterrows():
        imu_data = filter_imu3.imu_filter(
            ax=row['AccX(mg).2'], ay=row['AccY(mg).2'], az=row['AccZ(mg).2'],
            gx=row['GyrX(DPS).2'], gy=row['GyrY(DPS).2'], gz=row['GyrZ(DPS).2'],
            mx=row['MagX(uT).2'], my=row['MagY(uT).2'], mz=row['MagZ(uT).2'],
        )
        processed_data_imu3.append([row['AccX(mg).2'], row['AccY(mg).2'], row['AccZ(mg).2']])
        roll, pitch, yaw = filter_imu3.eulerAngles()
        filtered_data_imu3.append([roll, pitch, yaw])
        # print(f"IMU 3 - Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        processed_df = pd.DataFrame(processed_data_imu3, columns=['Processed_AccX2', 'Processed_AccY2', 'Processed_AccZ2'])
        filtered_df = pd.DataFrame(filtered_data_imu3, columns=['Roll2', 'Pitch2', 'Yaw2'])
        
        ax, ay, az = row['AccX(mg).2'], row['AccY(mg).2'], row['AccZ(mg).2']
        gx, gy, gz = np.radians([row['GyrX(DPS).2'], row['GyrY(DPS).2'], row['GyrZ(DPS).2']])
        
        measuredRoll = np.degrees(np.arctan2(ay, az))
        measuredPitch = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2)))
        measuredYaw = 0
        
        compl_filter_imu3.updateRollPitchYaw(measuredRoll, measuredPitch, measuredYaw, gx, gy, gz, DELTA_T)
        roll, pitch, yaw = compl_filter_imu3.roll, compl_filter_imu3.pitch, compl_filter_imu3.yaw
        compl_data_imu3.append([roll, pitch, yaw])
    
    # 创建DataFrame来存储每个IMU的处理后数据
    df_imu1 = pd.DataFrame(processed_data_imu1, columns=['IMU1_AccX', 'IMU1_AccY', 'IMU1_AccZ'])
    df_imu2 = pd.DataFrame(processed_data_imu2, columns=['IMU2_AccX', 'IMU2_AccY', 'IMU2_AccZ'])
    df_imu3 = pd.DataFrame(processed_data_imu3, columns=['IMU3_AccX', 'IMU3_AccY', 'IMU3_AccZ'])
    # 创建DataFrame来存储每个IMU的处理后数据
    angle_imu1 = pd.DataFrame(filtered_data_imu1, columns=['Roll1', 'Pitch1', 'Yaw1'])
    angle_imu2 = pd.DataFrame(filtered_data_imu2, columns=['Roll2', 'Pitch2', 'Yaw2'])
    angle_imu3 = pd.DataFrame(filtered_data_imu3, columns=['Roll3', 'Pitch3', 'Yaw3'])
    
    compl_imu1 = pd.DataFrame(compl_data_imu1, columns=['Roll1', 'Pitch1', 'Yaw1'])
    compl_imu2 = pd.DataFrame(compl_data_imu2, columns=['Roll2', 'Pitch2', 'Yaw2'])
    compl_imu3 = pd.DataFrame(compl_data_imu3, columns=['Roll3', 'Pitch3', 'Yaw3'])
    # 将三个DataFrame按列合并
    Acc_final_df = pd.concat([df_imu1, df_imu2, df_imu3], axis=1)
    Angle_final_df = pd.concat([angle_imu1, angle_imu2, angle_imu3], axis=1)
    compl_final_df = pd.concat([compl_imu1, compl_imu2, compl_imu3], axis=1)
    
    # Save the DataFrame to a CSV file
    Angle_final_df.to_csv('angle_state_imu_data.csv', index=False)
    Acc_final_df.to_csv('processed_imu_Acc_data.csv', index=False)
    compl_final_df.to_csv('compl_imu_Acc_data.csv', index=False)