import numpy as np

# 假定的常量
BETA = 0.1  # 你需要根据实际情况调整这个值
DELTA_T = 1/1000.0  # 假设的时间间隔，根据实际情况调整

# 四元数操作函数
def quat_mult(q, r):
    return np.array([
        q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3],
        q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2],
        q[0]*r[2] - q[1]*r[3] + q[2]*r[0] + q[3]*r[1],
        q[0]*r[3] + q[1]*r[2] - q[2]*r[1] + q[3]*r[0]
    ])

def quat_scalar(q, scalar):
    return q * scalar

def quat_add(q, r):
    return q + r

def quat_sub(q, r):
    return q - r

def quat_normalization(q):
    return q / np.linalg.norm(q)

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def imu_filter(ax, ay, az, gx, gy, gz, mx, my, mz, q_est_prev, beta):
    q_est_prev = np.array([1.0, 0.0, 0.0, 0.0])
    q_a = np.array([0.0, ax, ay, az])
    q_w = np.array([0.0, gx, gy, gz])
    q_m = np.array([0.0, mx, my, mz])
    
    # 陀螺仪数据预处理
    q_w = quat_scalar(q_w, 0.5)
    q_w = quat_mult(q_est_prev, q_w)
    
    # 加速度计和磁力计数据标准化
    q_a = quat_normalization(q_a)
    q_m = quat_normalization(q_m)
    
    # 融合加速度计和磁力计数据
    # 加速度计目标函数和雅可比矩阵与之前相同
    # 磁力计目标函数和雅可比矩阵的计算
    h = quat_mult(quat_mult(q_est_prev, q_m), quat_conjugate(q_est_prev))
    b = np.array([0, np.sqrt(h[1]**2 + h[2]**2), 0, h[3]])
    
    F_m = np.array([
        2*(q_est_prev[1]*q_est_prev[3] - q_est_prev[0]*q_est_prev[2]) - b[1],
        2*(q_est_prev[0]*q_est_prev[1] + q_est_prev[2]*q_est_prev[3]) - b[2],
        2*(0.5 - q_est_prev[1]**2 - q_est_prev[2]**2) - b[3]
    ])
    
    J_m = np.array([
        [-2*q_est_prev[2],  2*q_est_prev[3], -2*q_est_prev[0], 2*q_est_prev[1]],
        [2*q_est_prev[1], 2*q_est_prev[0], 2*q_est_prev[3], 2*q_est_prev[2]],
        [0, -4*q_est_prev[1], -4*q_est_prev[2], 0]
    ])
    
    F_g = np.array([
        2*(q_est_prev[1]*q_est_prev[3] - q_est_prev[0]*q_est_prev[2]) - q_a[1],
        2*(q_est_prev[0]*q_est_prev[1] + q_est_prev[2]*q_est_prev[3]) - q_a[2],
        2*(0.5 - q_est_prev[1]**2 - q_est_prev[2]**2) - q_a[3]
    ])
    
    J_g = np.array([
        [-2*q_est_prev[2],  2*q_est_prev[3], -2*q_est_prev[0], 2*q_est_prev[1]],
        [2*q_est_prev[1], 2*q_est_prev[0], 2*q_est_prev[3], 2*q_est_prev[2]],
        [0, -4*q_est_prev[1], -4*q_est_prev[2], 0]
    ])
    
    # 融合加速度计和磁力计的目标函数和雅可比矩阵
    F_gm = np.concatenate((F_g, F_m))
    J_gm = np.concatenate((J_g, J_m), axis=0)
    
    # 计算梯度
    gradient = np.dot(J_gm.T, F_gm)
    gradient = quat_normalization(gradient)
    
    # 应用Madgwick滤波器更新方程
    gradient = quat_scalar(gradient, BETA)
    q_est_dot = quat_sub(q_w, gradient)
    q_est_dot = quat_scalar(q_est_dot, DELTA_T)
    q_est = quat_add(q_est_prev, q_est_dot)
    q_est = quat_normalization(q_est)
    
    return q_est


# # get jacobi matrix
# def getMagJacobi(q):
#     return np.array([
#         [-2*q[2],  2*q[3], -2*q[0], 2*q[1]],
#         [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
#         [0, -4*q[1], -4*q[2], 0]
#     ])

# def getMagFunction(q, mag):
#     q = quat_normalization(q)
#     q_mag = np.array([0.0, mag[0], mag[1], mag[2]])
#     q_mag = quat_normalization(q_mag)
    
#     q_mag = quat_mult(q, q_mag)
#     q_mag = quat_mult(q_mag, quat_conjugate(q))
    
#     return np.array([
#         2*(q_mag[1]*q_mag[3] - q_mag[0]*q_mag[2]),
#         2*(q_mag[0]*q_mag[1] + q_mag[2]*q_mag[3]),
#         2*(0.5 - q_mag[1]**2 - q_mag[2]**2)
#     ])