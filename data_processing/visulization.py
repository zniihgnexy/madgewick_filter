import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df1 = pd.read_csv('../position/IMU1_position.csv')
df2 = pd.read_csv('../position/IMU2_position.csv')
df3 = pd.read_csv('../position/IMU3_position.csv')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df1['IMU1_X'], df1['IMU1_Y'], df1['IMU1_Z'], c='r', marker='o')
ax.scatter(df2['IMU2_X'], df2['IMU2_Y'], df2['IMU2_Z'], c='g', marker='^')
ax.scatter(df3['IMU3_X'], df3['IMU3_Y'], df3['IMU3_Z'], c='b', marker='s')

for i in range(len(df1)):
    ax.plot([df1['IMU1_X'][i], df2['IMU2_X'][i], df3['IMU3_X'][i]],
            [df1['IMU1_Y'][i], df2['IMU2_Y'][i], df3['IMU3_Y'][i]],
            [df1['IMU1_Z'][i], df2['IMU2_Z'][i], df3['IMU3_Z'][i]], 'k-')

# 设置轴标签
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# 烦人的比例尺问题，走的上限和下限
max_positive_range = np.array([df1[['IMU1_X', 'IMU1_Y', 'IMU1_Z']].max(),
                        df1[['IMU1_X', 'IMU1_Y', 'IMU1_Z']].min(),
                        df2[['IMU2_X', 'IMU2_Y', 'IMU2_Z']].max(),
                        df2[['IMU2_X', 'IMU2_Y', 'IMU2_Z']].min(),
                        df3[['IMU3_X', 'IMU3_Y', 'IMU3_Z']].max(),
                        df3[['IMU3_X', 'IMU3_Y', 'IMU3_Z']].min()]).max()

max_negative_range = np.array([df1[['IMU1_X', 'IMU1_Y', 'IMU1_Z']].max(),
                        df1[['IMU1_X', 'IMU1_Y', 'IMU1_Z']].min(),
                        df2[['IMU2_X', 'IMU2_Y', 'IMU2_Z']].max(),
                        df2[['IMU2_X', 'IMU2_Y', 'IMU2_Z']].min(),
                        df3[['IMU3_X', 'IMU3_Y', 'IMU3_Z']].max(),
                        df3[['IMU3_X', 'IMU3_Y', 'IMU3_Z']].min()]).min()

# 统一三个imu的坐标范围，并进行绘图
max_position_range = max(max_positive_range, abs(max_negative_range))
max_range = max_position_range
min_position_range = min(max_positive_range, abs(max_negative_range))
min_range = min_position_range

ax.set_xlim(min_range, max_range)
ax.set_ylim(min_range, max_range)
ax.set_zlim(min_range, max_range)

plt.show()
fig.savefig('../position/IMU_position.png')
