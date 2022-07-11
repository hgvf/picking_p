import os
import numpy as np
import matplotlib.pyplot as plt
import opensmile
import scipy.signal as signal

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from calc_frequency_feat import calc_frequency_features, calc_opensmile_features

class earthquake(Dataset):
    def __init__(self, stage, sliding_window=False, window_size=None, low_inten=False, upsample=False, triangular=False, 
                    plot=False, z_score_normalize=False, wiener=False, frequency=False, acoustic=False, gaussian=False,
                    aug=True, fixed_trigger=False, fixed_trigger_point=300):

        self.stage = stage
        self.aug = aug
        self.fixed_trigger = fixed_trigger
        self.fixed_trigger_point = fixed_trigger_point
        self.triangular = triangular
        self.gaussian = gaussian
        self.z_score = z_score_normalize
        self.wiener = wiener
        self.frequency = frequency
        self.acoustic = acoustic
        self.sliding_window = sliding_window
        self.window_size = window_size
        
        if acoustic:
            # openSmile
            self.egemaps = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv01a,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
            self.emobase = opensmile.Smile(
                feature_set=opensmile.FeatureSet.emobase,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
            self.compare = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

        if plot:
            print('start plotting')
            self.stage = 'train'

        if fixed_trigger:
            print(f'Fixed trigger point at {fixed_trigger_point}-th sample.')

        if self.triangular:
            print(f"{self.stage} on triangular label")

        # file directory
        self.palert_path = "/mnt/nas3/earthquake_dataset_large/Palert"
        self.tsmip_path = "/mnt/nas3/earthquake_dataset_large/TSMIP/"
        self.cwbsn_path = "/mnt/nas3/earthquake_dataset_large/CWBSN/"
        self.stead_noise_path = "/mnt/nas3/earthquake_dataset_large/STEAD/chunk1/"
        
        # get files
        self.palert_files = os.listdir(self.palert_path)               
        self.tsmip_files = os.listdir(self.tsmip_path)
        self.cwbsn_files = os.listdir(self.cwbsn_path)                 
        # self.stead_noise_files = os.listdir(self.stead_noise_path)[:25000]
        self.stead_noise_files = os.listdir(self.stead_noise_path)     

        # 不能同時只 train 在 low intensity 與 upsample
        assert not(low_inten == True and upsample == True), 'Low intensity and upsample cannot both be True'

        # 只訓練在震度低的資料上
        if low_inten:
            print(f"only {self.stage} on low intensity")
            self.palert_files = self.low_intensity(self.palert_files)
            self.tsmip_files = self.low_intensity(self.tsmip_files)
            self.cwbsn_files = self.low_intensity(self.cwbsn_files)

        # 針對高震度資料作 upsample
        if upsample:
            self.palert_files = self.upsample(self.palert_files)
            self.tsmip_files = self.upsample(self.tsmip_files)
            self.cwbsn_files = self.upsample(self.cwbsn_files)

        self.p_alert = len(self.palert_files)
        self.tsmip = len(self.tsmip_files)
        self.cwbsn = len(self.cwbsn_files)
        self.stead_noise = len(self.stead_noise_files)
        self.n_files = self.p_alert + self.tsmip + self.cwbsn + self.stead_noise
        
        # 建立 index & shuffle
        self.indices = np.arange(self.n_files)
        np.random.shuffle(self.indices)

        if stage == 'train':
            self.stage_indices = self.indices[:int(self.n_files*0.8)]
        elif stage == 'valid':
            self.stage_indices = self.indices[int(self.n_files*0.8) : int(self.n_files*0.9)]
        else:
            self.stage_indices = self.indices[int(self.n_files*0.9):]
        
    def __len__(self):
        return self.stage_indices.shape[0]
    
    def __getitem__(self, idx):
        idx = self.stage_indices[idx]
        
        wave, isParrival, isEarthquake = self._load_wave(idx)

        # Z-score normalization
        if self.z_score:
            wave = self.z_score_standardize(wave)

        # gaussian noise on earthquake waveform
        if idx < self.p_alert + self.tsmip + self.cwbsn and self.stage != 'test' and self.aug:
            wave = self._add_noise(wave)
            # wave = self._shift_to_end(wave)

        # adding outliers & dropout channel & gaps on both waveform
        if self.stage != 'test' and self.aug:
            wave = self._add_outliers(wave)
            wave = self._add_gaps(wave)
            wave = self._drop_channel(wave)

        if self.triangular:
            wave[-1] = self._triangular(wave[-1])

        if self.gaussian:
            if isParrival:
                wave[-1] = self._gaussian(wave.shape[1], 50, wave[-1])
            else:
                wave[-1] = torch.zeros(self.window_size)

        if self.frequency and self.acoustic:
            freq_feat = calc_frequency_features(wave)
            acu_feat = calc_opensmile_features(wave, self.egemaps, self.emobase, self.compare)
            wave = wave.permute(1,0)
            return wave[:, :-1], wave[:, -1], freq_feat, acu_feat, isEarthquake

        elif self.frequency:
            freq_feat = calc_frequency_features(wave)
            wave = wave.permute(1,0)
            return wave[:, :-1], wave[:, -1], freq_feat, -1, isEarthquake

        elif self.acoustic:
            acu_feat = calc_opensmile_features(wave, self.egemaps, self.emobase, self.compare)
            wave = wave.permute(1,0)
            return wave[:, :-1], wave[:, -1], -1, acu_feat, isEarthquake

        else:
            wave = wave.permute(1,0)

            return wave[:, :-1], wave[:, -1], -1, -1, isEarthquake
        
    def _load_wave(self, idx):
        isParrival=True
        isEarthquake=True

        while True:
            # p_alert
            if idx < self.p_alert:
                flag, wave = self._open_p_alert(idx)

                if self.sliding_window:
                    wave, isParrival = self._slice_waveform(wave)

            # tsmip
            elif idx >= self.p_alert and idx < self.p_alert + self.tsmip:
                flag, wave = self._open_tsmip(idx-self.p_alert)

                if self.sliding_window:
                    wave, isParrival = self._slice_waveform(wave)

            # cwbsn
            elif idx >= self.p_alert + self.tsmip and idx < self.p_alert + self.tsmip + self.cwbsn:
                flag, wave = self._open_cwbsn(idx-(self.p_alert+self.tsmip))
            
                if self.sliding_window:
                    wave, isParrival = self._slice_waveform(wave)

            # stead noise
            else:
                flag, wave = self._open_stead_noise(idx-(self.p_alert+self.tsmip+self.cwbsn))
                isEarthquake = False

                if self.sliding_window:
                    wave, isParrival = self._slice_waveform(wave, True)

            # exception during loading waveform, generating new index to load waveform
            if not flag or (self.sliding_window and self.window_size != wave.shape[1]):
                new_idx = np.random.randint(0, self.stage_indices.shape[0])
                idx = self.stage_indices[new_idx]
                # print('restart waveform')
            else:
                break

        if self.fixed_trigger:
            wave = self._fixed_trigger(wave)

        return wave, isParrival, isEarthquake

    def _fixed_trigger(self, data):
        # 將 p arrival time 固定在某 sample 點
        wave = data.clone()

        tri = torch.where(wave[-1]==1)[0]
            
        if len(tri) > 0:
            trigger = tri[0]

            if self.sliding_window:
                total_len = self.window_size
            else:
                total_len = 3000

            front_noise_len = self.fixed_trigger_point
            left_edge = (trigger-front_noise_len) if (trigger-front_noise_len)>=0 else 0
            remaining = total_len-(trigger-left_edge)
            right_edge = trigger+remaining

            if right_edge >= total_len:
                padding_len = right_edge-total_len
                
                mean = torch.mean(wave[:3, :trigger-200], dim=1) if trigger-200 >= 0 else torch.mean(wave[:3, :trigger], dim=1)
                
                x = wave[:, left_edge:right_edge]
                
                wave[0] = torch.cat((x[0], mean[0].repeat(padding_len)))
                wave[1] = torch.cat((x[1], mean[0].repeat(padding_len)))
                wave[2] = torch.cat((x[2], mean[0].repeat(padding_len)))
                wave[-1] = torch.cat((x[-1], torch.zeros(padding_len)))
        
        return wave

    def _slice_waveform(self, data, isNoise=False):
        wave = data.clone()
        
        if isNoise:
            start = np.random.randint(0, 3000-self.window_size-1)

            return wave[:, start:start+self.window_size], False

        # 有 p_arrival_time 在前面的機率 60 %
        if np.random.uniform(0, 1) < 0.90:
            tri = torch.where(wave[-1]==1)[0]
            
            if len(tri) > 0:
                start = np.random.randint(max(tri[0]-350, 0), max(tri[0]-45, 1))
                # print(f"tri: {tri[0]}, start: {start}, case p_arrival")
            else:
                start = np.random.randint(0, 3000-self.window_size-1)
                # print(f"start: {start}, noise with high prob")
            
            return wave[:, start:start+self.window_size], True
        else:
            tri = torch.where(wave[-1]==1)[0]

            if len(tri) > 0:
                if tri[0] >= 3000-self.window_size-1:
                    if tri[0] == 3000-self.window_size-1:
                        return wave[:, tri[0]+1:], False
                        
                    data = torch.empty(4, self.window_size)
                    tri = tri[0]
                    remaining_wave = self.window_size-(3000-tri)+10
                    mean = torch.mean(wave[:3, :tri-400], dim=1) if tri-400 >= 0 else torch.mean(wave[:3, :tri], dim=1)
                
                    if torch.any(torch.isnan(mean)):
                        mean = torch.zeros(3)
                    
                    data[0] = torch.cat((wave[0, tri+10:], mean[0].repeat(remaining_wave)))
                    data[1] = torch.cat((wave[1, tri+10:], mean[1].repeat(remaining_wave)))
                    data[2] = torch.cat((wave[2, tri+10:], mean[2].repeat(remaining_wave)))
                    data[3] = torch.cat((wave[3, tri+10:], torch.zeros(remaining_wave)))
                    # print(f'tri: {tri}, padding with mean')
                    return data, False
                else:
                    if tri[0]+100 >= 3000-self.window_size-1:
                        start = np.random.randint(low=tri[0], high=3000-self.window_size-1)
                    else:
                        start = np.random.randint(low=tri[0]+100, high=3000-self.window_size-1)
                    # print(f"start: {start}, trigger: {tri[0]}, randomly choose start position")
            else:
                start = np.random.randint(low=0, high=3000-self.window_size-1)
                # print(f"start: {start}, noise")

            return wave[:, start:start+self.window_size] , False

    def _open_p_alert(self, idx):
        data_path = os.path.join(self.palert_path, self.palert_files[idx])
        
        wave = torch.load(data_path)

        if self.wiener:
            wave[:-1] = torch.FloatTensor(signal.wiener(wave[:-1].numpy()))

        if wave.shape[1] != 3000:
            return False, torch.rand(4, 3000)

        return True, wave
    
    def _open_tsmip(self, idx):
        data_path = os.path.join(self.tsmip_path, self.tsmip_files[idx])
        wave = torch.load(data_path)

        if self.wiener:
            wave[:-1] = torch.FloatTensor(signal.wiener(wave[:-1].numpy()))

        # print(data_path)
        # swap the row: (e, n, z, gt) -> (z, n, e, gt)
        #new_idx = torch.LongTensor([2, 1, 0, 3])
        #wave = wave[new_idx]
        
        if wave.shape[1] != 3000:
            return False, torch.rand(4, 3000)

        return True, wave
    
    def _open_cwbsn(self, idx):
        data_path = os.path.join(self.cwbsn_path, self.cwbsn_files[idx])
        # print(data_path)
        wave = torch.load(data_path)
        
        if self.wiener:
            wave[:-1] = torch.FloatTensor(signal.wiener(wave[:-1].numpy()))

        if wave.shape[1] != 3000:
            return False, torch.rand(4, 3000)

        return True, wave
    
    def _open_stead_noise(self, idx):
        data_path = os.path.join(self.stead_noise_path, self.stead_noise_files[idx])
        # print(data_path)
        wave = torch.load(data_path)
        
        if self.wiener:
            wave[:-1] = torch.FloatTensor(signal.wiener(wave[:-1].numpy()))

        if wave.shape[1] != 3000:
            return False, torch.rand(4, 3000)

        return True, wave
    
    def _drop_channel(self, data):
        'Randomly replace values of one or two components to zeros in earthquake data'

        data = data.clone()

        try:
            if np.random.uniform(0, 1) < 0.2: 
                c1 = np.random.choice([0, 1])
                c2 = np.random.choice([0, 1])
                c3 = np.random.choice([0, 1])

                if c1 + c2 + c3 > 0:
                    data[np.array([c1, c2, c3, 1]) == 0, :] = 0    

        except Exception as e:
            print(e)

        return data
    
    def _add_noise(self, data):
        'Randomly add Gaussian noie with a random SNR into waveforms'
        
        data_noisy = np.empty((data.shape))

        try:
            if np.random.uniform(0, 1) < 0.5: 
                data_noisy = np.empty((data.shape))
                data_noisy[0, :] = data[0, :] + np.random.normal(0, np.abs(np.random.uniform(0.01, 0.15)*max(data[0, :])), data.shape[1])
                data_noisy[1, :] = data[1, :] + np.random.normal(0, np.abs(np.random.uniform(0.01, 0.15)*max(data[1, :])), data.shape[1])
                data_noisy[2, :] = data[2, :] + np.random.normal(0, np.abs(np.random.uniform(0.01, 0.15)*max(data[2, :])), data.shape[1])    
                data_noisy[3] = data[3]
            else:
                data_noisy = data
        except Exception as e:
            print(e)
            data_noisy = data
        
        return torch.FloatTensor(data_noisy)
    
    def _add_gaps(self, data): 
        'Randomly add gaps (zeros) of different sizes into waveforms'
        
        data = data.clone()
        
        try:
            gap_start = np.random.randint(100, data.shape[1]-150)
            gap_end = np.random.randint(gap_start, data.shape[1]-50)
            if np.random.uniform(0, 1) < 0.2: 
                data[:3, gap_start:gap_end] = 0           

        except Exception as e:
            print(e)

        return data  
    
    def _add_outliers(self, data):
        data = data.clone()

        try:
            outlier_start = np.random.randint(100, data.shape[1]-100)
            outlier_length = np.random.randint(1, 100)
            if np.random.uniform(0, 1) < 0.1:
                c1 = np.random.choice([0, 1])
                c2 = np.random.choice([0, 1])
                c3 = np.random.choice([0, 1])

                if c1 == 1:
                    a1 = np.random.randint(torch.max(data[0]), torch.max(data[0])+30)
                    data[0, outlier_start:outlier_start+outlier_length] = a1
                if c2 == 1:
                    a2 = np.random.randint(torch.max(data[1]), torch.max(data[1])+30)
                    data[1, outlier_start:outlier_start+outlier_length] = a2
                if c3 == 1:
                    a3 = np.random.randint(torch.max(data[2]), torch.max(data[2])+30)
                    data[2, outlier_start:outlier_start+outlier_length] = a3
        except Exception as e:
            print(e)

        return data

    def _shift_to_end(self, wave): 
        data = wave.clone()

        try:
            if np.random.uniform(0, 1) < 0.35:
                tri = torch.where(wave[-1]==1)[0]

                if len(tri) > 0:
                    tri = tri[0].item()
                else:
                    return wave

                mean = torch.mean(wave[:3, :tri-200], dim=1) if tri-200 >= 0 else torch.mean(wave[:3, :tri], dim=1)
                
                if torch.any(torch.isnan(mean)):
                    mean = torch.zeros(3)

                new_tri = np.random.randint(data.shape[1]-200, data.shape[1]-50)
                remaining_wave = data.shape[1]-new_tri
                
                wave[0] = torch.cat((mean[0].repeat(data.shape[1]-remaining_wave), wave[0, tri:tri+remaining_wave]))
                wave[1] = torch.cat((mean[1].repeat(data.shape[1]-remaining_wave), wave[1, tri:tri+remaining_wave]))
                wave[2] = torch.cat((mean[2].repeat(data.shape[1]-remaining_wave), wave[2, tri:tri+remaining_wave]))
                wave[3] = torch.cat((torch.zeros(data.shape[1]-remaining_wave), wave[3, tri:tri+remaining_wave]))
        except Exception as e:
            print(e)  
            return data  
        
        return wave

    def z_score_standardize(self, data):
        new_wave = torch.empty((data.shape))

        try:
            for i in range(3):
                tmp_wave = data[i] - torch.mean(data[i])
                tmp_wave /= torch.std(data[i])

                if torch.any(torch.isinf(tmp_wave)):
                    tmp_wave[torch.isinf(tmp_wave)] = 0
                
                if torch.any(torch.isnan(tmp_wave)):
                    tmp_wave[torch.isnan(tmp_wave)] = 0

                new_wave[i] = tmp_wave

            new_wave[3] = data[3]
        except Exception as e:
            print(e)
            return data

        return new_wave

    def _triangular(self, y):
        if y.any():
            start = torch.where(y)[0][0]
        else:
            return y

        for i in range(1, 20):
            y[start+i] = y[start-i] = (1/20)*(20-i)

        y[:start-20] = y[start+20:] = 0

        return y

    def low_intensity(self, files):
        # 挑震度小於 3 的地震波型

        low = []
        for f in files:
            if int(f[-4]) <= 3:
                low.append(f)

        return low

    def upsample(self, files):
        # upsample: 震度超過 3 的就大幅提升該事件數量

        print('original: ', len(files))
        append_list = []
        for f in files:
            if int(f[-4]) >= 4:
                tmp = [f]
                append_list += np.repeat(tmp, int(1.5**(int(f[-4])))).tolist()

        files += append_list
        print('after upsampling: ', len(files))

        return files

    def _gaussian(self, data_length, mask_window, y):
        '''
        data_length: target function length
        point: point of phase arrival
        mask_window: length of mask, must be even number
                    (mask_window//2+1+mask_window//2)
        '''
        # find trigger point
        if y.any():
            point = torch.where(y)[0][0]
        else:
            return y

        target = np.zeros(data_length)
        half_win = mask_window//2
        gaus = np.exp(-(
            np.arange(-half_win, half_win+1))**2 / (2*(half_win//2)**2))
        #print(gaus.std())
        gaus_first_half = gaus[:mask_window//2]
        gaus_second_half = gaus[mask_window//2+1:]
        target[point] = gaus.max()
        #print(gaus.max())
        if point < half_win:
            reduce_pts = half_win-point
            start_pt = 0
            gaus_first_half = gaus_first_half[reduce_pts:]
        else:
            start_pt = point-half_win
        target[start_pt:point] = gaus_first_half
        target[point+1:point+half_win+1] = \
            gaus_second_half[:len(target[point+1:point+half_win+1])]

        return torch.FloatTensor(target)





