import torch
import torch.nn as nn
import numpy as np
import math
import opensmile
import scipy
import scipy.signal as signal
from matplotlib import mlab
from scipy.fft import fftshift
from scipy.signal import butter, lfilter, freqz

# ================= waveform preprocess ================= #
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def zscore(wave):
    for i in range(3):
        tmp_wave = wave[i] - torch.mean(wave[i])
        tmp_wave /= torch.std(wave[i])

        if torch.any(torch.isinf(tmp_wave)):
            tmp_wave[torch.isinf(tmp_wave)] = 0
        
        if torch.any(torch.isnan(tmp_wave)):
            tmp_wave[torch.isnan(tmp_wave)] = 0
    
        wave[i] = tmp_wave

    return wave

# ================= Spectrogram ================= #
def _nearest_pow_2(x):
    """
    Find power of two nearest to x

    >>> _nearest_pow_2(3)
    2.0
    >>> _nearest_pow_2(15)
    16.0

    :type x: float
    :param x: Number
    :rtype: int
    :return: Nearest power of 2 to x
    """
    a = math.pow(2, math.ceil(np.log2(x)))
    b = math.pow(2, math.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b

def obspy_spectrogram(data):
    samp_rate = 100.0

    npts = data.shape[0]
    wlen = samp_rate / 100.
    per_lap = 0.9
    mult = 8.0
    clip = [0.0, 1.0]
    
    # nfft needs to be an integer, otherwise a deprecation will be raised
    # XXX add condition for too many windows => calculation takes for ever
    nfft = int(_nearest_pow_2(wlen * samp_rate))
    if nfft > npts:
        nfft = int(_nearest_pow_2(npts / 8.0))

    mult = int(_nearest_pow_2(mult))
    mult = mult * nfft
    nlap = int(nfft * float(per_lap))

    data = data - data.mean()
    end = npts / samp_rate
    
    # Here we call not plt.specgram as this already produces a plot
    # matplotlib.mlab.specgram should be faster as it computes only the
    # arrays
    # XXX mlab.specgram uses fft, would be better and faster use rfft
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft, noverlap=nlap)
   
    # db scale and remove zero/offset for amplitude
    specgram = np.sqrt(specgram[1:, :])
    freq = freq[1:]

    # this method is much much faster!
    specgram = np.flipud(specgram)
   
    return specgram, time, freq

def calc_spectrogram(wave):
    # 10Hz low pass filter
    wave[0] = torch.FloatTensor(butter_lowpass_filter(wave[0].numpy(), 10, 100.0, 6))
    wave[1] = torch.FloatTensor(butter_lowpass_filter(wave[1].numpy(), 10, 100.0, 6))
    wave[2] = torch.FloatTensor(butter_lowpass_filter(wave[2].numpy(), 10, 100.0, 6))

    # input: torch.FloatTensor -> wave: (4, 3000) (z, n, e, gt)
    specgram_z, t_z, f_z = obspy_spectrogram(wave[0].numpy())
    specgram_n, t_n, f_n = obspy_spectrogram(wave[1].numpy())
    specgram_e, t_e, f_e = obspy_spectrogram(wave[2].numpy())

    feat_z = np.empty((wave.shape[1], 14))
    feat_n = np.empty((wave.shape[1], 14))
    feat_e = np.empty((wave.shape[1], 14))
    t_z = (t_z*100).astype(int)
    t_n = (t_n*100).astype(int)
    t_e = (t_e*100).astype(int)

    for i in range(t_z.shape[0]):
        if i == 0:
            feat_z[:t_z[i], :] = specgram_z[:14, i]
            feat_n[:t_n[i], :] = specgram_n[:14, i]
            feat_e[:t_e[i], :] = specgram_e[:14, i]
        elif i == t_z.shape[0]-1:
            feat_z[t_z[i]:, :] = specgram_z[:14, i]
            feat_n[t_n[i]:, :] = specgram_n[:14, i]
            feat_e[t_e[i]:, :] = specgram_e[:14, i]
        else:
            feat_z[t_z[i-1]:t_z[i], :] = specgram_z[:14, i]
            feat_n[t_n[i-1]:t_n[i], :] = specgram_n[:14, i]
            feat_e[t_e[i-1]:t_e[i], :] = specgram_e[:14, i]
            
    feat = np.concatenate((feat_z, feat_n, feat_e), axis=-1)
    
    big = np.max(feat, axis=0)
    small = np.min(feat, axis=0)
    
    tmp1 = np.subtract(feat, small)
    if np.any(np.isnan(tmp1)):
        tmp1[np.isnan(tmp1)==1] = 0.001

    tmp2 = np.subtract(big, small)
    if np.any(np.isnan(tmp2)):
        tmp2[np.isnan(tmp2)==1] = 0.001

    tmp2[tmp2==0.0] = 0.001

    feat = np.divide(tmp1, tmp2)

    return torch.FloatTensor(feat)

# =================== STFT ====================== #
def calc_stft(wave):
    # 10Hz low pass filter
    wave[0] = torch.FloatTensor(butter_lowpass_filter(wave[0].numpy(), 10, 100.0, 6))
    wave[1] = torch.FloatTensor(butter_lowpass_filter(wave[1].numpy(), 10, 100.0, 6))
    wave[2] = torch.FloatTensor(butter_lowpass_filter(wave[2].numpy(), 10, 100.0, 6))

    FS = 100
    NPERSEG = 50
    NFFT = 128

    f_z, t_z, tmp_signal_z = scipy.signal.stft(wave[0].numpy(), fs=FS, nperseg=NPERSEG, nfft=NFFT, boundary='zeros')
    tmp_signal_z.real[np.isnan(tmp_signal_z.real)] = 0
    tmp_signal_z.imag[np.isinf(tmp_signal_z.imag)] = 0

    f_n, t_n, tmp_signal_n = scipy.signal.stft(wave[1].numpy(), fs=FS, nperseg=NPERSEG, nfft=NFFT, boundary='zeros')
    tmp_signal_n.real[np.isnan(tmp_signal_n.real)] = 0
    tmp_signal_n.imag[np.isinf(tmp_signal_n.imag)] = 0

    f_e, t_e, tmp_signal_e = scipy.signal.stft(wave[2].numpy(), fs=FS, nperseg=NPERSEG, nfft=NFFT, boundary='zeros')
    tmp_signal_e.real[np.isnan(tmp_signal_e.real)] = 0
    tmp_signal_e.imag[np.isinf(tmp_signal_e.imag)] = 0

    feat_z = np.empty((wave.shape[1], 14))
    feat_n = np.empty((wave.shape[1], 14))
    feat_e = np.empty((wave.shape[1], 14))
    t_z = (t_z*100).astype(int)
    t_n = (t_n*100).astype(int)
    t_e = (t_e*100).astype(int)

    for i in range(t_z.shape[0]):
        if i == 0:
            feat_z[:t_z[i], :] = np.abs(tmp_signal_z[:14, i])
            feat_n[:t_n[i], :] = np.abs(tmp_signal_n[:14, i])
            feat_e[:t_e[i], :] = np.abs(tmp_signal_e[:14, i])
        elif i == t_z.shape[0]-1:
            feat_z[t_z[i]:, :] = np.abs(tmp_signal_z[:14, i])
            feat_n[t_n[i]:, :] = np.abs(tmp_signal_n[:14, i])
            feat_e[t_e[i]:, :] = np.abs(tmp_signal_e[:14, i])
        else:
            feat_z[t_z[i-1]:t_z[i], :] = np.abs(tmp_signal_z[:14, i])
            feat_n[t_n[i-1]:t_n[i], :] = np.abs(tmp_signal_n[:14, i])
            feat_e[t_e[i-1]:t_e[i], :] = np.abs(tmp_signal_e[:14, i])

    feat = np.concatenate((feat_z, feat_n, feat_e), axis=-1)

    return torch.FloatTensor(feat)

# # =================== OpenSmile ====================== #
def extract_features(wave, smile):
    for j in range(3):
        out = torch.FloatTensor(smile.process_signal(wave[j].numpy(), 100).to_numpy())
        
        if j == 0:
            feat_z = out
        elif j == 1:
            feat_n = out
        else:
            feat_e = out
    
    feat_z, feat_n, feat_e = feat_z.unsqueeze(0), feat_n.unsqueeze(0), feat_e.unsqueeze(0)
    feat = torch.cat((feat_z, feat_n, feat_e), dim=0)

    return feat

# ==================== Output ===================== #
def calc_frequency_features(wave):
    # [42, 3000]
    spectrogram = calc_spectrogram(wave).permute(1,0)
    stft = calc_stft(wave).permute(1,0)
    
    # zscore normalization
    s_mean, s_std = torch.mean(spectrogram, dim=1), torch.std(spectrogram, dim=1)
    stft_mean, stft_std = torch.mean(stft, dim=1), torch.std(stft, dim=1)

    for i in range(stft.shape[0]):
        spectrogram[i] -= s_mean[i]
        spectrogram[i] /= s_std[i]

        stft[i] -= stft_mean[i]
        stft[i] /= stft_std[i]

        if torch.any(torch.isnan(spectrogram[i])):
            spectrogram[i][torch.isnan(spectrogram[i])] = 0
        if torch.any(torch.isnan(stft[i])):
            stft[i][torch.isnan(stft[i])] = 0
        if torch.any(torch.isinf(spectrogram[i])):
            spectrogram[i][torch.isinf(spectrogram[i])] = 0
        if torch.any(torch.isinf(stft[i])):
            stft[i][torch.isinf(stft[i])] = 0

    return spectrogram, stft

def calc_opensmile_features(wave, egemaps, emobase, compare):
    # zscore
    wave = zscore(wave)

    # egemaps: slope0-500_sma3 (3), slope500-1500_sma3 (4), spectralFlux_sma3 (5): [9, 2996]
    feat_ege = extract_features(wave, egemaps)[:, :, [3,4,5]]
    feat_ege = torch.FloatTensor(feat_ege).permute(0, 2, 1)
    feat_ege = feat_ege.reshape(9, -1)

    # compare: pcm_RMSenergy_sma (8), pcm_fftMag_fband250-650_sma (36), pcm_fftMag_spectralSlope_sma (48): [9, 2996]
    feat_comp = extract_features(wave, compare)[:, :, [8,36,48]]
    feat_comp = torch.FloatTensor(feat_comp).permute(0, 2, 1)
    feat_comp = feat_comp.reshape(9, -1)

    # emobase: pcm_intensity_sma (0), pcm_loudness_sma (1): [6, 2998]
    feat_emo = extract_features(wave, emobase)[:, :, [0,1]]
    feat_emo = torch.FloatTensor(feat_emo).permute(0, 2, 1)
    feat_emo = feat_emo.reshape(6, -1)

    # zscore normalization
    e_mean, e_std = torch.mean(feat_ege, dim=1), torch.std(feat_ege, dim=1)
    c_mean, c_std = torch.mean(feat_comp, dim=1), torch.std(feat_comp, dim=1)
    emo_mean, emo_std = torch.mean(feat_emo, dim=1), torch.std(feat_emo, dim=1)

    for i in range(feat_ege.shape[0]):
        feat_ege[i] -= e_mean[i]
        feat_ege[i] /= e_std[i]

        feat_comp[i] -= c_mean[i]
        feat_comp[i] /= c_std[i]

        if i < 6:
            feat_emo[i] -= emo_mean[i]
            feat_emo[i] /= emo_std[i]

            if torch.any(torch.isnan(feat_emo[i])):
                feat_emo[i][torch.isnan(feat_emo[i])] = 0
            if torch.any(torch.isinf(feat_emo[i])):
                feat_emo[i][torch.isinf(feat_emo[i])] = 0

        if torch.any(torch.isnan(feat_ege[i])):
            feat_ege[i][torch.isnan(feat_ege[i])] = 0
        if torch.any(torch.isnan(feat_comp[i])):
            feat_comp[i][torch.isnan(feat_comp[i])] = 0
        if torch.any(torch.isinf(feat_ege[i])):
            feat_ege[i][torch.isinf(feat_ege[i])] = 0
        if torch.any(torch.isinf(feat_comp[i])):
            feat_comp[i][torch.isinf(feat_comp[i])] = 0

    # print(feat_ege.shape, feat_comp.shape, feat_emo.shape)
    return feat_ege, feat_comp, feat_emo

# ==================== Feature Combination network ====================== #
class FrequencyCNN(nn.Module):
    def __init__(self, spectrogram=True, stft=True):
        super(FrequencyCNN, self).__init__()
        
        self.spectrogram = spectrogram
        self.stft = stft

        if spectrogram:
            self.spectrogram_conv = nn.Sequential(nn.Conv1d(42, 32, 5, padding='same'),
                                                nn.BatchNorm1d(32),
                                                nn.ReLU(),
                                                nn.Conv1d(32, 16, 5, padding='same'),
                                                nn.BatchNorm1d(16),
                                                nn.ReLU())
        
        if stft:
            self.stft_conv = nn.Sequential(nn.Conv1d(42, 32, 5, padding='same'),
                                            nn.BatchNorm1d(32),
                                            nn.ReLU(),
                                            nn.Conv1d(32, 16, 5, padding='same'),
                                            nn.BatchNorm1d(16),
                                            nn.ReLU())

        if spectrogram and stft:
            self.out = nn.Sequential(nn.Linear(32, 96),
                                    nn.ReLU(),
                                    nn.Linear(96, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 8))
        else:
            self.out = nn.Sequential(nn.Linear(16, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 16),
                                    nn.ReLU(),
                                    nn.Linear(16, 8))
        self.trash = nn.Linear(1,1)

    def forward(self, freq_feat):
        spectrogram, stft = freq_feat
                    
        if self.spectrogram and self.stft:
            spectrogram = spectrogram.to(self.trash.weight.device)
            out_spectrogram = self.spectrogram_conv(spectrogram)

            stft = stft.to(self.trash.weight.device)
            out_stft = self.stft_conv(stft)

            feat = torch.cat((out_spectrogram, out_stft), dim=1).permute(0,2,1)
        elif self.spectrogram:
            spectrogram = spectrogram.to(self.trash.weight.device)
            out_spectrogram = self.spectrogram_conv(spectrogram)

            feat = out_spectrogram.permute(0,2,1)
        elif self.stft:
            stft = stft.to(self.trash.weight.device)
            out_stft = self.stft_conv(stft)

            feat = out_stft.permute(0,2,1)

        out_feat = self.out(feat)
        
        return out_feat

class OpenSmileCNN(nn.Module):
    def __init__(self, emobase=True, compare=True, egemaps=True):
        super(OpenSmileCNN, self).__init__()
        
        self.emobase = emobase
        self.compare = compare
        self.egemaps = egemaps

        if emobase:
            self.emobase_conv = nn.Sequential(nn.Upsample(3000),
                                        nn.Conv1d(6, 6, 5, padding='same'),
                                        nn.ReLU(),
                                        nn.Conv1d(6, 16, 5, padding='same'))
        
        if compare:
            self.compare_conv = nn.Sequential(nn.Upsample(3000),
                                            nn.Conv1d(9, 9, 5, padding='same'),
                                            nn.ReLU(),
                                            nn.Conv1d(9, 16, 5, padding='same'),
                                            nn.ReLU())

        if egemaps:
            self.egemaps_conv = nn.Sequential(nn.Upsample(3000),
                                            nn.Conv1d(9, 9, 5, padding='same'),
                                            nn.ReLU(),
                                            nn.Conv1d(9, 16, 5, padding='same'),
                                            nn.ReLU())

        total = emobase + compare + egemaps
        if total == 3:
            self.out = nn.Sequential(nn.Linear(48, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 48),
                                    nn.ReLU(),
                                    nn.Linear(48, 8))
        elif total == 2:
            self.out = nn.Sequential(nn.Linear(32, 96),
                                    nn.ReLU(),
                                    nn.Linear(96, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 8))
        else:
            self.out = nn.Sequential(nn.Linear(16, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 16),
                                    nn.ReLU(),
                                    nn.Linear(16, 8))
        self.trash = nn.Linear(1,1)

    def forward(self, acu_feat):
        egemaps, compare, emobase = acu_feat

        batch_size, dim = egemaps.size(0), egemaps.size(1)
        # print(egemaps.shape, compare.shape, emobase.shape)

        if self.emobase:
            emobase = emobase.to(self.trash.weight.device)
            out_emobase = self.emobase_conv(emobase)
        if self.compare:
            compare = compare.to(self.trash.weight.device)
            out_compare = self.compare_conv(compare)
        if self.egemaps:
            egemaps = egemaps.to(self.trash.weight.device)
            out_egemaps = self.egemaps_conv(egemaps)

        feat = torch.cat((out_emobase, out_compare, out_egemaps), dim=1)        
        # print('feat: ', feat.shape)
        out_feat = self.out(feat.permute(0,2,1))
        
        return out_feat
