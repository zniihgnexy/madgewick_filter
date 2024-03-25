1K signal processing
- window: 1.5s (1500 rows)
- raw processed data (after normalisation, band-pass filter and down-sampling) 
    - stored in the "raw_window"
- STFT results 
	- images
	- CSV 
		- Zxx
            1st row:frequencys
            1st column: Time
- DFT results
	- CSV
		- Zxx
            1st column: frequency
            2nd column: amplitude

Example:
    from EMG_baseline/abs/DFT
        - 1_EMG_DFT_0.csv
            - 1 : 1st participant
            - EMG_DFT: EMG_DFT
            - 0: window 0
        - 2_EMG_DFT_0.csv
            - 2 : 2nd participant     