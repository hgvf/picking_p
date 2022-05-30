
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class earthquake(Dataset):
    def __init__(self, stage, low_inten=False, upsample=False, triangular=False, plot=False, z_score_normalize=False):
        self.stage = stage
        self.triangular = triangular
        self.z_score = z_score_normalize
        if plot:
            print('start plotting')
            self.stage = 'train'

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
        
        # p_alert
        if idx < self.p_alert:
            wave = self._open_p_alert(idx)
        
        # tsmip
        elif idx >= self.p_alert and idx < self.p_alert + self.tsmip:
            wave = self._open_tsmip(idx-self.p_alert)
        
        # cwbsn
        elif idx >= self.p_alert + self.tsmip and idx < self.p_alert + self.tsmip + self.cwbsn:
            wave = self._open_cwbsn(idx-(self.p_alert+self.tsmip))
        
        # stead noise
        else:
            wave = self._open_stead_noise(idx-(self.p_alert+self.tsmip+self.cwbsn))

        # Z-score normalization
        if self.z_score:
            wave = self.z_score_standardize(wave)

        # gaussian noise on earthquake waveform
        if idx < self.p_alert + self.tsmip + self.cwbsn and self.stage != 'test':
            wave = self._add_noise(wave)

        # adding gaps on noise waveform
        if idx >= self.p_alert + self.tsmip + self.cwbsn and self.stage != 'test':
            wave = self._add_gaps(wave)

        # adding outliers & dropout channel on both waveform
        if self.stage != 'test':
            wave = self._add_outliers(wave)
            wave = self._drop_channel(wave)

        if self.triangular:
            wave[-1] = self._triangular(wave[-1])

        wave = wave.permute(1,0)
        #draw(wave[0], wave[1], wave[2], wave[3])
        
        return wave[:, :-1], wave[:, -1]
        
    def _open_p_alert(self, idx):
        while True:
            data_path = os.path.join(self.palert_path, self.palert_files[idx])
            
            wave = torch.load(data_path)
            tmp = torch.load(os.path.join(self.palert_path, self.palert_files[idx+1]))
            if wave.shape[1] == 3000:
                break
            else:
                idx += 1
                continue
        wave = torch.cat((wave, tmp), dim=1)
        return wave
    
    def _open_tsmip(self, idx):
        while True:
            data_path = os.path.join(self.tsmip_path, self.tsmip_files[idx])
            wave = torch.load(data_path)
            tmp = torch.load(os.path.join(self.tsmip_path, self.tsmip_files[idx+1]))

            # swap the row: (e, n, z, gt) -> (z, n, e, gt)
            #new_idx = torch.LongTensor([2, 1, 0, 3])
            #wave = wave[new_idx]
            
            if wave.shape[1] == 3000:
                
                break
            else:
                idx += 1
                continue
        wave = torch.cat((wave, tmp), dim=1)
        return wave
    
    def _open_cwbsn(self, idx):
        while True:
            data_path = os.path.join(self.cwbsn_path, self.cwbsn_files[idx])

            wave = torch.load(data_path)
            tmp = torch.load(os.path.join(self.cwbsn_path, self.cwbsn_files[idx+1]))
            if wave.shape[1] == 3000:
                break
            else:
                idx += 1
                continue
        wave = torch.cat((wave, tmp), dim=1)
        return wave
    
    def _open_stead_noise(self, idx):
        while True:
            data_path = os.path.join(self.stead_noise_path, self.stead_noise_files[idx])

            wave = torch.load(data_path)
            tmp = torch.load(os.path.join(self.stead_noise_path, self.stead_noise_files[idx+1]))
            if wave.shape[1] == 3000:
                break
            else:
                idx += 1
                continue
        wave = torch.cat((wave, tmp), dim=1) 
        return wave
    
    def _drop_channel(self, data):
        'Randomly replace values of one or two components to zeros in earthquake data'

        data = data.clone()
        if np.random.uniform(0, 1) < 0.2: 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])

        return data
    
    def _add_noise(self, data):
        'Randomly add Gaussian noie with a random SNR into waveforms'
        
        data_noisy = np.empty((data.shape))
        if np.random.uniform(0, 1) < 0.5: 
            data_noisy = np.empty((data.shape))
            data_noisy[0, :] = data[0, :] + np.random.normal(0, np.abs(np.random.uniform(0.01, 0.15)*max(data[0, :])), data.shape[1])
            data_noisy[1, :] = data[1, :] + np.random.normal(0, np.abs(np.random.uniform(0.01, 0.15)*max(data[1, :])), data.shape[1])
            data_noisy[2, :] = data[2, :] + np.random.normal(0, np.abs(np.random.uniform(0.01, 0.15)*max(data[2, :])), data.shape[1])    
            data_noisy[3] = data[3]
        else:
            data_noisy = data
            
        
        return torch.FloatTensor(data_noisy)
    
    def _add_gaps(self, data): 
        'Randomly add gaps (zeros) of different sizes into waveforms'
        
        data = data.clone()
        gap_start = np.random.randint(300, 2100)
        gap_end = np.random.randint(gap_start, 2900)
        if np.random.uniform(0, 1) < 0.2: 
            data[:3, gap_start:gap_end] = 0           
        return data  
    
    def _add_outliers(self, data):
        data = data.clone()
        outlier_start = np.random.randint(200, 2500)
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
                
        return data

    def z_score_standardize(self, data):
        new_wave = data.clone()

        new_wave[0] = (data[0]-torch.mean(data[0]))/torch.std(data[0])
        new_wave[1] = (data[1]-torch.mean(data[1]))/torch.std(data[1])
        new_wave[2] = (data[2]-torch.mean(data[2]))/torch.std(data[2])

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
                append_list += np.repeat(tmp, int(2**int(f[-4]))).tolist()

        files += append_list
        print('after upsampling: ', len(files))

        return files


