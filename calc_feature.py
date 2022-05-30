import numpy as np
import torch
import torch.nn as nn

class characteristic(nn.Module):
    def __init__(self):
        super(characteristic, self).__init__()
        
        self.CharFuncFilt = torch.tensor([3])
        self.rawDataFilt = torch.tensor([0.939])
        self.small_float = torch.tensor([1.0e-10])
        
    def forward(self, rdata):
        self.CharFuncFilt = self.CharFuncFilt.to(rdata.device)
        self.rawDataFilt = self.rawDataFilt.to(rdata.device)
        self.small_float = self.small_float.to(rdata.device)

        # filter
        result = torch.empty((rdata.shape[0], rdata.shape[1], rdata.shape[2])).to(rdata.device)
        data = torch.zeros((rdata.shape[0], 3)).to(rdata.device)
        
        for i in range(rdata.shape[1]):
            if i == 0:
                data = data*self.rawDataFilt+rdata[:, i, :]+self.small_float
            else:
                data = data*self.rawDataFilt+(rdata[:, i, :]-rdata[:, i-1, :])+self.small_float

            result[:, i, :] = data
        
        wave_square = torch.square(result)
        
        # characteristic_diff
        diff = torch.empty((result.shape[0], result.shape[1], result.shape[2])).to(result.device)

        for i in range(result.shape[1]):
            if i == 0:
                diff[:, i, :] = result[:, 0, :]
            else:
                diff[:, i, :] = result[:, i, :]-result[:, i-1, :]

        diff_square = torch.square(diff)
        
        wave_characteristic = torch.add(wave_square,torch.multiply(diff_square,self.CharFuncFilt))
        
        return wave_characteristic

class sta(nn.Module):
    def __init__(self):
        super(sta, self).__init__()
        
        self.STA_W = torch.tensor([0.6])
        
    def forward(self, wave):
        self.STA_W = self.STA_W.to(wave.device)

        sta = torch.zeros(wave.shape[0], 3).to(wave.device)
        wave_sta = torch.empty((wave.shape[0], wave.shape[1], wave.shape[2])).to(wave.device)

        # Compute esta, the short-term average of edat 
        for i in range(wave.shape[1]):
            sta += self.STA_W*(wave[:, i, :]-sta)
            wave_sta[:, i, :] = sta

        return wave_sta
        
class lta(nn.Module):
    def __init__(self):
        super(lta, self).__init__()
        
        self.LTA_W = torch.tensor([0.015])
        
    def forward(self, wave):
        self.LTA_W = self.LTA_W.to(wave.device)

        lta = torch.zeros(wave.shape[0], 3).to(wave.device)
        wave_lta = torch.empty((wave.shape[0], wave.shape[1], wave.shape[2])).to(wave.device)
        
        # Compute esta, the short-term average of edat 
        for i in range(wave.shape[1]):
            lta += self.LTA_W*(wave[:, i, :]-lta)
            wave_lta[:, i, :] = lta

        return wave_lta
    
class compute_feature(nn.Module):
    def __init__(self):
        super(compute_feature, self).__init__()
        
        self.char = characteristic()
        self.sta = sta()
        self.lta = lta()
    def forward(self, wave):
        #e, n, z = e.permute(1,0), n.permute(1,0), z.permute(1,0)
   
        #wave = torch.hstack((e, n, z)).unsqueeze(0)

        char = self.char(wave)
        s = self.sta(char)
        l = self.lta(char)
        
        feature = torch.cat((char, s, l, wave), axis=-1)
        
        return feature

