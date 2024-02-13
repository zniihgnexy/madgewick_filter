import numpy as np
import pandas as pd
from imu_filter import imu_filter
from numpy import arctan2, row_stack
import pandas as pd

import numpy as np
import pandas as pd

# Assuming imu_filter is your filtering logic
class MadgwickFilter:
    def __init__(self):
        # Initial quaternion estimate
        self.q_est = np.array([1.0, 0.0, 0.0, 0.0])
    
    def imu_filter(self, ax, ay, az, gx, gy, gz):
        gx, gy, gz = np.radians([gx, gy, gz])
        self.q_est = imu_filter(ax, ay, az, gx, gy, gz)
        
    def eulerAngles(self):
        # Calculate Euler angles from self.q_est
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
    
    data = pd.read_csv(file_path, skiprows=1, usecols=lambda column: column not in ['Timestamp (uS)', 'CH1', 'CH2', 'CH3'])
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

    # read imu1 data from the first 9 columns
    imu1 = data.iloc[:, :6]
    imu1.columns = imu_columns[:6]
    print("read imu1 data",imu1.columns)
    
    # read imu2 data from the next 9 columns
    imu2 = data.iloc[:, 9:15]
    imu2.columns = imu_columns[9:15]
    print("read imu2 data",imu2.columns)
    
    # read imu3 data from the last 9 columns
    imu3 = data.iloc[:, 18:24]
    imu3.columns = imu_columns[18:24]
    print("read imu3 data",imu3.columns)
    
    print(imu1.columns)
    print(imu2.columns)
    print(imu3.columns)
    
    filter_imu1 = MadgwickFilter()
    filter_imu2 = MadgwickFilter()
    filter_imu3 = MadgwickFilter()

    DELTA_T = 1/1000.0
    PI = np.pi
    
    filtered_data = []
    processed_data = []

    print("\n\n Upright to solve gradient descent problem")
    for index, row in imu1.iterrows():
        # import pdb; pdb.set_trace()
        imu_data = filter_imu1.imu_filter(
            ax=row['AccX(mg)'], ay=row['AccY(mg)'], az=row['AccZ(mg)'],
            gx=row['GyrX(DPS)'], gy=row['GyrY(DPS)'], gz=row['GyrZ(DPS)'],
            # mx=row['MagX(uT)'], my=row['MagY(uT)'], mz=row['MagZ(uT)'],
        )
        processed_data.append([row['AccX(mg)'], row['AccY(mg)'], row['AccZ(mg)']])
        roll, pitch, yaw = filter_imu1.eulerAngles()
        filtered_data.append([roll, pitch, yaw])
        print(f"IMU 1 - Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")

    print("\n\nYAW 100 Degrees per Second")
    for index, row in imu2.iterrows():
        ax, ay, az, gx, gy, gz = filter_imu2.imu_filter(
            ax=row['AccX(mg).1'], ay=row['AccY(mg).1'], az=row['AccZ(mg).1'],
            gx=row['GyrX(DPS).1'], gy=row['GyrY(DPS).1'], gz=row['GyrZ(DPS).1'],
            # mx=row['MagX(uT).1'], my=row['MagY(uT).1'], mz=row['MagZ(uT).1'],
        )
        # print("imu data z", ax, ay, az, gx, gy, gz)
        processed_data.append([row['AccX(mg)'], row['AccY(mg)'], row['AccZ(mg)']])
        roll, pitch, yaw = filter_imu2.eulerAngles()
        filtered_data.append([roll, pitch, yaw])
        print(f"IMU 2 - Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")

    print("\n\nYAW -20 Degrees per Second")
    for index, row in imu3.iterrows():
        ax, ay, az, gx, gy, gz = filter_imu3.imu_filter(
            ax=row['AccX(mg).2'], ay=row['AccY(mg).2'], az=row['AccZ(mg).2'],
            gx=row['GyrX(DPS).2'], gy=row['GyrY(DPS).2'], gz=row['GyrZ(DPS).2'],
            # mx=row['MagX(uT).2'], my=row['MagY(uT).2'], xz=row['MagZ(uT).2'],
        )
        processed_data.append([row['AccX(mg)'], row['AccY(mg)'], row['AccZ(mg)']])
        roll, pitch, yaw = filter_imu3.eulerAngles()
        filtered_data.append([roll, pitch, yaw])
        print(f"IMU 3 - Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
    
    filtered_df = pd.DataFrame(filtered_data, columns=['Roll', 'Pitch', 'Yaw'])
    processed_df = pd.DataFrame(processed_data, columns=['Processed_AccX', 'Processed_AccY', 'Processed_AccZ'])

    # Save the DataFrame to a CSV file
    filtered_df.to_csv('filtered_imu_data.csv', index=False)
    processed_df.to_csv('processed_imu_data.csv', index=False)