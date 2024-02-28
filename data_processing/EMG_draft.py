import os
import numpy as np
from matplotlib.figure import SubplotParams
import pyemgpipeline as pep
import numpy as np
import pandas as pd
from numpy import arctan2, row_stack
import pandas as pd
import numpy as np
import pandas as pd
import sys 
from butterworth import butter_lowpass_filter
import matplotlib.pyplot as plt
from madgwick_filter import MadgwickFilter

import os
import numpy as np
from matplotlib.figure import SubplotParams
import pyemgpipeline as pep

from EMG_STFT import window_nonzero, create_overlapping_segments, stft, istft
from EMG_STFT import istft
import numpy as np
from scipy.signal.windows import boxcar, hann, tukey

import scipy.signal as signal
from scipy.signal import resample

import matplotlib.pyplot as plt

import pandas as pd

def plot_stft(f, t, Zxx, i,name,label,number):
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude')
    plt.savefig(f'./train_data/EMG_{label}/{name}/STFT/images/{number}_EMG_stft_{i}.png')  # 保存图像
    plt.close()

def get_csv_data(file_path, imu_columns):
    data_all = pd.read_csv(file_path, usecols=imu_columns,skiprows=1)
    data_all = data_all.dropna()
    EMG_array = np.array(data_all)

    return EMG_array

# def downsampled_data(EMG_array, downsample_factor):
#     downsampled_data = signal.decimate(EMG_array[:,0], downsample_factor)
#     return downsampled_data

def downsample_data(data, downsample_factor):
    num_samples = len(data) // downsample_factor
    return resample(data, num_samples)

def normalize_data(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)

def plot_comparison(original_data, downsampled_data):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(original_data)
    plt.title('Original Data')

    plt.subplot(2, 1, 2)
    plt.plot(downsampled_data)
    plt.title('Downsampled Data')

    plt.tight_layout()
    plt.show()

def plot_LocalDFT(f, LocalT, Zxx):
    import pdb; pdb.set_trace()
    Z = Zxx[:,LocalT]

    plt.figure(figsize=(12, 6))
    plt.plot(f, np.abs(Z))
    plt.title('DFT Magnitude')
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency [Hz]')
    plt.show()

def file_setting(file_path):
    if file_path == './data/Rec0.csv':
        name = 'trap'
        number = "1"
        label = "baseline"
    elif file_path == './data/Rec1.csv':
        name = 'back'
        number = "1"
        label = "baseline"
    elif file_path == './data/Rec2.csv':
        name = 'abs'
        number = "1"
        label = "baseline"
    elif file_path == './data/Rec3.csv':
        name = 'abs'
        number = "1"
        label = "reach_under"
    elif file_path == './data/Rec4.csv':
        name = 'abs'
        number = "1"
        label = "reach_over"
    elif file_path == './data/Rec5.csv':
        name = 'back'
        number = "1_2"
        label = "baseline"
    elif file_path == './data/Rec6.csv':
        name = 'trap'
        number = "1"
        label = "drop_over"
    elif file_path == './data/Rec7.csv':
        name = 'trap'
        number = "1"
        label = "seat_over"
    elif file_path == './data/Rec8.csv':
        name = 'trap'
        number = "2"
        label = "baseline"
    elif file_path == './data/Rec9.csv':
        name = 'back'
        number = "2"
        label = "baseline"
    elif file_path == './data/Rec10.csv':
        name = 'abs'
        number = "2"
        label = "baseline"
    elif file_path == './data/Rec11.csv':
        name = 'abs'
        number = "2"
        label = "reach_under"
    elif file_path == './data/Rec12.csv':
        name = 'abs'
        number = "2"
        label = "reach_over"
    elif file_path == './data/Rec13.csv':
        name = 'back'
        number = "2"
        label = "drop_under"
    elif file_path == './data/Rec14.csv':
        name = 'back'
        number = "2"
        label = "drop_over"
    elif file_path == './data/Rec15.csv':
        name = 'trap'
        number = "2"
        label = "seat_under"
    elif file_path == './data/Rec16.csv':
        name = 'trap'
        number = "2"
        label = "seat_over"
    return name, number, label

if __name__ == "__main__":
    ############################################################
    # Prepare data
    ############################################################

    file_paths = [os.path.join('./data', file).replace('\\', '/') for file in os.listdir('./data') if file.endswith('.csv')]

    # 对每个文件调用file_setting函数
    for file_path in file_paths:
        name, number, label = file_setting(file_path)
        imu_columns = ['CH2']
        downsample_factor = 4
        sample_rate = 4000
        EMG_array = get_csv_data(file_path, imu_columns)

        emg_plot_params = pep.plots.EMGPlotParams(
            n_rows=len(imu_columns),
            fig_kwargs={
                'figsize': (8, 6),
                'dpi': 70,
                'subplotpars': SubplotParams(wspace=0, hspace=0.6),
            },
            line2d_kwargs={
                'color': 'red',
            }
        )

        ############################################################
        # Normalize Data
        ############################################################

        max_value = float('-inf')
        min_value = float('inf')

        for filename in os.listdir('./data'):
            if filename.endswith('.csv'):
                # 读取csv文件
                df = pd.read_csv(os.path.join('./data', filename), usecols=['CH2'],skiprows=1)
                
                # 移除极端值
                upper_threshold = df['CH2'].quantile(0.99)
                lower_threshold = df['CH2'].quantile(0.01)
                df = df[(df['CH2'] < upper_threshold) & (df['CH2'] > lower_threshold)]
                
                # 更新最大值
                max_value = max(max_value, df['CH2'].max())
                
                # 更新最小值，只有当新的最小值不为零时，才更新最小值
                min_value_temp = df[df['CH2'] != 0]['CH2'].min()
                if min_value_temp > 0:
                    min_value = min(min_value, min_value_temp)

        print('Max value:', max_value)
        print('Min value:', min_value)

        normalized_EMG = normalize_data(EMG_array, min_value, max_value)

        ############################################################
        # Down Sampling from 4K Hz to 1K Hz
        ############################################################

        m = pep.wrappers.EMGMeasurement(normalized_EMG, hz=sample_rate, trial_name="trial_name",
                                        channel_names=imu_columns, emg_plot_params=emg_plot_params)
        m.apply_dc_offset_remover()
        m.apply_bandpass_filter(bf_order=4, bf_cutoff_fq_lo=10, bf_cutoff_fq_hi=499)

        downsampled_EMG = downsample_data(m.data[:], downsample_factor) #1K HZ sampling rate

        # plot_comparison(EMG_array, downsampled_EMG)
        ############################################################
        # Sampling Data with 1.5K window and 50% overlap
        ############################################################

        window_length = 1500  # 窗口长度
        overlap = 0.5  # 重叠
        sample_rate = 1000
        step = int(window_length * (1 - overlap))  # 步长
        stft_results_f = []
        stft_results_Zxx = []
        j = 0


        for i in range(0, len(downsampled_EMG) - window_length, step):
            print("i:",i)
            print("i+window_length:",i+window_length)
            print("j:",j)
            window_data = downsampled_EMG[i:i+window_length]

            df = pd.DataFrame(window_data)
            df.to_csv(f'./train_data/EMG_{label}/{name}/raw_window/{number}_EMG_raw_window_data_{j}.csv', index=False)

            f, t, Zxx = signal.stft(window_data[:,0], fs=sample_rate,window='hann', nperseg=100, noverlap=50, nfft=100, padded=True)
            plot_stft(f, t, Zxx, j, name, label,number)

            df_stft = pd.DataFrame(Zxx.real, index=f, columns=t)
            df_stft.to_csv(f'./train_data/EMG_{label}/{name}/STFT/CSV/{number}_EMG_stft_{j}.csv')


            Zxx = np.fft.fft(m.data[:])
            f = np.fft.fftfreq(len(m.data[:]))

            # 只取一半的数据（正频率部分）
            half_point = len(Zxx) // 2
            Zxx = Zxx[:half_point]
            f = f[:half_point] * sample_rate

            df_DFT = pd.DataFrame(Zxx.real, index=f)
            df_DFT.to_csv(f'./train_data/EMG_{label}/{name}/DFT/{number}_EMG_DFT_{j}.csv')

            # # 绘制结果
            # plt.figure(figsize=(12, 6))
            # plt.plot(f, Zxx.real)
            # plt.title('DFT Magnitude')
            # plt.ylabel('Magnitude')
            # plt.xlabel('Frequency [Hz]')
            # plt.show()
            j = j + 1




            # # f, t, Zxx = signal.stft(m.data[:], fs=sample_rate)
            # f, t, Zxx = signal.stft(m.data[:], fs=sample_rate,window='hann', nperseg=50, noverlap=25, nfft=50, padded=True)
            # stft_results_f.append(f)
            # stft_results_Zxx.append(Zxx)
            # plot_stft(f, t, Zxx)
            # import pdb; pdb.set_trace()





   
