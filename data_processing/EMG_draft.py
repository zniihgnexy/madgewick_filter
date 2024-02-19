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

if __name__ == "__main__":
    file_path = './train_data_ori/emg_train_data1.csv'
    
    data_all = pd.read_csv(file_path, usecols=lambda column: ['CH1', 'CH2', 'CH3'], nrows=30000)
    data_all = data_all.dropna()

    imu_columns = ['CH1', 'CH2', 'CH3']

    EMG_array = np.array(data_all)

    DELTA_T = 1/100.0 # 100Hz, change to the correct one
    PI = np.pi
    cutoff = 20
    fs = 100
    order = 6


    sample_rate = 1000
 
    emg_plot_params = pep.plots.EMGPlotParams(
        n_rows=3,
        fig_kwargs={
            'figsize': (8, 6),
            'dpi': 70,
            'subplotpars': SubplotParams(wspace=0, hspace=0.6),
        },
        line2d_kwargs={
            'color': 'red',
        }
    )
    
    m = pep.wrappers.EMGMeasurement(EMG_array, hz=sample_rate, trial_name="trial_name",
                                    channel_names=imu_columns, emg_plot_params=emg_plot_params)
    # m.plot()
 
    
    m.apply_dc_offset_remover()
    # m.plot()
    m.apply_bandpass_filter(bf_order=4, bf_cutoff_fq_lo=10, bf_cutoff_fq_hi=499)
    # m.plot()
    m.apply_full_wave_rectifier()
    m.apply_linear_envelope(le_order=4, le_cutoff_fq=12)
    m.plot()
    
    # max_amplitude = [0.043, 0.069, 0.364, 0.068]  # assume the MVC is known
    # m.apply_amplitude_normalizer(max_amplitude)
    # m.plot()

    data = m.data
    segment_length = 2048
    shift_length = 216
    segment_length_padded = 2048
    window_function = boxcar
    # take stft
    x_stft, start_list, stop_list = stft(data, segment_length,
        segment_length_padded, shift_length, window_function)
    
    magnitude = np.abs(x_stft)  # 计算幅度
    plt.figure(figsize=(10, 6))
    plt.imshow(20*np.log10(magnitude), aspect='auto', origin='lower',
            extent=[0, 76, 0, 513])
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time Segment')
    plt.ylabel('Frequency Bin')
    plt.title('Spectrogram')
    plt.show()



   
