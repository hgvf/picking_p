import numpy as np
import json
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from scipy import integrate
from scipy.signal import sosfilt, iirfilter, zpk2sos

def highpass(data, freq, df, corners=4, zerophase=False):
    fe = 0.5 * df
    f = freq / fe

    z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)

def lowpass(data, freq, df, corners=4, zerophase=False):
    fe = 0.5 * df
    f = freq / fe
    
    z, p, k = iirfilter(corners, f, btype='lowpass', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)

# 速度微分 -> 加速度
# 100.0: sampling_rate
def v_to_a(z, n, e, sample_rate):
    d_z = np.gradient(z, 1.0/sample_rate)
    d_n = np.gradient(n, 1.0/sample_rate)
    d_e = np.gradient(e, 1.0/sample_rate)

    return d_z, d_n, d_e

# 加速度積分 -> 速度
# 100.0: sampling_rate
def a_to_v(z, n, e, sample_rate):
    i_z = integrate.cumtrapz(z)/sample_rate
    i_n = integrate.cumtrapz(n)/sample_rate
    i_e = integrate.cumtrapz(e)/sample_rate

    return i_z, i_n, i_e

def calc_pga(z, n, e, waveType, sample_rate):
    # 檢查波形是 速度 or 加速度
    if waveType == 'Velocity':
        z, n, e = v_to_a(z, n, e, sample_rate)

    # 10Hz low pass filter
    z = lowpass(z, 10, sample_rate)
    n = lowpass(n, 10, sample_rate)
    e = lowpass(e, 10, sample_rate)

    # 合成震波
    acc = np.sqrt(z**2+n**2+e**2)

    # Max=PGA
    pga = max(acc)
    pga_z = max(z)
    pga_n = max(n)
    pga_e = max(e)

    return pga, pga_z, pga_n, pga_e

def pga_to_intensity(pga):
    if pga < 0.8:
        return 0
    elif pga >= 0.8 and pga < 2.5:
        return 1
    elif pga >= 2.5 and pga < 8.0:
        return 2
    elif pga >= 8.0 and pga < 25.0:
        return 3
    elif pga >= 25.0 and pga < 80.0:
        return 4
    else:
        return 5

def calc_pgv(z, n, e, waveType, sample_rate):
    # 檢查波形是 速度 or 加速度
    if waveType == 'Acceleration':
        z, n, e = a_to_v(z, n, e, sample_rate)
 
    # 0.075Hz high pass filter
    z = highpass(z, 0.075, sample_rate)
    n = highpass(n, 0.075, sample_rate)
    e = highpass(e, 0.075, sample_rate)
        
    # 合成震波
    acc = np.sqrt(z**2+n**2+e**2)
    
    # Max=PGV
    pgv = max(acc)
    pgv_z = max(z)
    pgv_n = max(n)
    pgv_e = max(e)
    
    return pgv, pgv_z, pgv_n, pgv_e

def pgv_to_intensity(pgv):
    if pgv >= 15 and pgv < 30:
        return "5 weak"
    elif pgv >= 30 and pgv < 50:
        return "5 strong"
    elif pgv >= 50 and pgv < 80:
        return "6 weak"
    elif pgv >= 80 and pgv < 140:
        return "6 strong"
    elif pgv >= 140:
        return '7'
    else:
        return '4'

def calc_intensity(z, n, e, waveType, sample_rate):

    # 物理量修正 (去平均)
    z = z - np.mean(z)
    n = n - np.mean(n)
    e = e - np.mean(e)

    # 先計算 PGA 檢查震度是否 > 5
    pga, pga_z, pga_n, pga_e = calc_pga(z, n, e, waveType, sample_rate)
    #print(f"pgaZ:{pga_z}, pgaN:{pga_n}, pgaE: {pga_e}")
    
    # 依照 pga 推測震度
    intensity = pga_to_intensity(pga)
    
    pgv, pgv_z, pgv_n, pgv_e = calc_pgv(z, n, e, waveType, sample_rate)
    #print(f"pgvZ:{pgv_z}, pgvN:{pgv_n}, pgvE: {pgv_e}")
    
    # 震度五級以上，則用 PGV 決定震度
    if intensity == 5:    
        # 依照 pgv 推測震度
        intensity = pgv_to_intensity(pgv)
    
    intensity = str(intensity)
    return intensity, pga, pgv
