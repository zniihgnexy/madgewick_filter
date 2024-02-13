from imu_filter import imu_filter
import numpy as np

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