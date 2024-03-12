import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
# data = pd.read_csv('./train_data/EMG_reach_under/abs/DFT/1_EMG_DFT_30.csv')
data = pd.read_csv('./train_data/EMG_reach_over/abs/DFT/1_EMG_DFT_20.csv')

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(data.iloc[:, 0], data.iloc[:, 1])
plt.title('DFT Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()