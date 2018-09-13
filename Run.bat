python New_Ver.py -ht LSTM -hu 512 -tt 10 -me 10  -em M -et Agnes -idx 5
rem python New_Ver.py -ht LSTM -hu 512 -tt 1000 -se 3000 -me 8000  -em M -et Agnes  -idx 0 -mf ./IDX_0/IM_Spectrogram.HM_LSTM.OM_SRV.PM_Normal.Size_10000.H_512.WL_10.NR.Trimmed.EM_M.ET_Agnes.IDX_0/Result/Metadata.pickle
rem python New_Ver.py -ht LSTM -hu 512 -tt 1000 -se 8000 -me 10000 -em M -et Agnes -ei -idx 0 -mf D:/Deep_Listener_Results/IDX_0/IM_Spectrogram.HM_LSTM.OM_SRV.PM_Normal.Size_10000.H_512.WL_10.NR.Trimmed.EM_M.ET_Agnes.IDX_0/Result/Metadata.pickle
python Result_Analysis.py -f ./IDX_0/IM_Spectrogram.HM_LSTM.OM_SRV.PM_Normal.Size_10000.H_512.WL_10.NR.Trimmed.EM_M.ET_Agnes.IDX_0
