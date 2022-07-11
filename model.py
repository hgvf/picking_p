import torch
import torch.nn as nn
import math

from calc_feature import *
from calc_frequency_feat import FrequencyCNN, OpenSmileCNN
from conformer import *

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size*num_windows, window_size, input_dim)
        x = x[:] + self.pe.squeeze()
        
        return self.dropout(x)

class SingleP_transformer_window(nn.Module):
    def __init__(self, d_ffn, n_head, enc_layers, dropout, window_size, frequency=False, acoustic=False):
        super(SingleP_transformer_window, self).__init__()

        self.window_size = window_size
        self.frequency = frequency
        self.acoustic = acoustic

        self.compute_feature = compute_feature()

        if frequency:
            self.freq_feat_network = FrequencyCNN()
        if acoustic:
            self.acou_feat_network = OpenSmileCNN()

        self.transformer = nn.TransformerEncoderLayer(d_model=20, nhead=n_head, dim_feedforward=d_ffn, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformer, num_layers=enc_layers)
        
        self.fc = nn.Linear(20, 1)
        self.score = nn.Sigmoid()

    def forward(self, input_x, freq_feat, acu_feat):
        # compute 12-dim features
        input_x = self.compute_feature(input_x)
        
        if self.frequency and self.acoustic:
            freq_feat = self.freq_feat_network(freq_feat)
            acu_feat = self.acou_feat_network(acu_feat)

            input_x = torch.cat((input_x, freq_feat, acu_feat), dim=-1)
        elif self.frequency:
            freq_feat = self.freq_feat_network(freq_feat)
            # print('========================================')
            # print('freq_feat: ', freq_feat)
            # print('isnan: ', torch.any(torch.isnan(freq_feat)))
            # print('========================================')
            input_x = torch.cat((input_x, freq_feat), dim=-1)
        elif self.acoustic:
            acu_feat = self.acou_feat_network(acu_feat)
            input_x = torch.cat((input_x, acu_feat), dim=-1)
            
        # input_x: [batch_size, seg_len, input_dim]
        batch_size, seg_len, input_dim = input_x.size(0), input_x.size(1), input_x.size(2)

        # windows 總數 = batch_size * (一段 waveform 共有多少時間窗 ex. n_windows = 3000-20 = 2980 個時間窗)
        # ouput_scores: [batch_size, windows 總數 / batch_size]
        output_scores = torch.empty((batch_size, seg_len-self.window_size), dtype=torch.float32)
        
        # sliding windows 處理完後 shape: [windows 總數, windows_size, input_size]
        list_of_windows = [input_x[:, start:start+self.window_size] for start in range(seg_len-self.window_size)]
        x = torch.stack((list_of_windows))
        x = torch.reshape(x, (-1, self.window_size, input_dim))

        x = self.encoder(x)
        x = self.fc(x).squeeze()
        scores = self.score(x[:, -1])
    
        # score input
        for b in range(batch_size):
            output_scores[b] = scores[b::batch_size]
        
        return output_scores

class SingleP_transformer_window_short(nn.Module):
    def __init__(self, d_ffn, n_head, enc_layers, window_size, dropout=0.1):
        super(SingleP_transformer_window_short, self).__init__()

        self.compute_feature = compute_feature()
        self.pos_emb = PositionalEncoding(12, 0.1, window_size)

        self.transformer = nn.TransformerEncoderLayer(d_model=12, nhead=n_head, dim_feedforward=d_ffn, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformer, num_layers=enc_layers)
        
        self.fc = nn.Linear(12, 1)
        self.score = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        nn.init.xavier_uniform_(self.fc.weight)
        self._init_transformer_weight()
        
    def _init_transformer_weight(self):
        nn.init.xavier_uniform_(self.transformer.linear1.weight)
        nn.init.xavier_uniform_(self.transformer.linear2.weight)        
        
    def forward(self, input_x, freq_feat, acu_feat):
        # compute 12-dim features
        x = self.compute_feature(input_x)
        x = self.pos_emb(x)
        
        x = self.encoder(x)
        
        scores = self.score(self.fc(x).squeeze())
        
        return scores

class SingleP_CNNTransformer(nn.Module):
    def __init__(self, d_ffn, n_head, n_layers, frequency=True, acoustic=True):
        super(SingleP_CNNTransformer, self).__init__()

        self.frequency = frequency
        self.acoustic = acoustic
        self.compute_feature = compute_feature()

        if self.frequency and self.acoustic:
            in_dim = 28
        elif not self.frequency and not self.acoustic:
            in_dim = 12
        else:
            in_dim = 20

        self.down1 = nn.Sequential(nn.Conv1d(in_dim, in_dim, 11),
                                 nn.ReLU(),
                                 nn.AvgPool1d(2),
                                 nn.Conv1d(in_dim, 32, 9),
                                 nn.ReLU(),
                                 nn.AvgPool1d(2))
        
        self.down2 = nn.Sequential(nn.Conv1d(32, 32, 7),
                                  nn.ReLU(),
                                  nn.AvgPool1d(2),
                                  nn.Conv1d(32, 64, 7),
                                  nn.ReLU(),
                                  nn.AvgPool1d(2))
        
        self.down3 = nn.Sequential(nn.Conv1d(64, 64, 5),
                                  nn.ReLU(),
                                  nn.AvgPool1d(2),
                                  nn.Conv1d(64, 128, 5),
                                  nn.ReLU(),
                                  nn.AvgPool1d(2),
                                  nn.Conv1d(128, 128, 3),
                                  nn.ReLU(),
                                  nn.AvgPool1d(2))

        self.up1 = nn.Sequential(nn.Upsample(40),
                                nn.Conv1d(128, 96, 3),
                                nn.ReLU(),
                                nn.Upsample(76),
                                nn.Conv1d(96, 96, 5),
                                nn.ReLU(),
                                nn.Upsample(144),
                                nn.Conv1d(96, 64, 5),
                                nn.ReLU(),)
        
        self.up2 = nn.Sequential(nn.Upsample(280),
                                nn.Conv1d(64, 64, 7),
                                nn.ReLU(),
                                nn.Upsample(548),
                                nn.Conv1d(64, 32, 7),
                                nn.ReLU(),
                                nn.Upsample(1084),
                                nn.Conv1d(32, 32, 9),
                                nn.ReLU(),
                                nn.Upsample(2168),
                                nn.Conv1d(32, 16, 11),
                                nn.ReLU(),
                                nn.Upsample(3010),
                                nn.Conv1d(16, 8, 11),
                                nn.ReLU(),)

        self.fc = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

        if frequency:
            self.freq_feat_network = FrequencyCNN()
        if acoustic:
            self.acou_feat_network = OpenSmileCNN()

        # self.transformer_front = nn.TransformerEncoderLayer(d_model=128, nhead=n_head, dim_feedforward=d_ffn, batch_first=True)
        # self.encoder_front = nn.TransformerEncoder(self.transformer_front, num_layers=n_layers)

        self.transformer_mid = nn.TransformerEncoderLayer(d_model=128, nhead=n_head, dim_feedforward=d_ffn, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformer_mid, num_layers=n_layers)

        # self.transformer_back = nn.TransformerEncoderLayer(d_model=32, nhead=n_head, dim_feedforward=d_ffn, batch_first=True)
        # self.encoder_back = nn.TransformerEncoder(self.transformer_back, num_layers=n_layers)

    def forward(self, x, freq_feat, acu_feat):
        # compute 12-dim features
        x = self.compute_feature(x)

        if self.frequency and self.acoustic:
            freq_feat = self.freq_feat_network(freq_feat)
            acu_feat = self.acou_feat_network(acu_feat)

            x = torch.cat((x, freq_feat, acu_feat), dim=-1)
        elif self.frequency:
            freq_feat = self.freq_feat_network(freq_feat)
            # print('========================================')
            # print('freq_feat: ', freq_feat)
            # print('isnan: ', torch.any(torch.isnan(freq_feat)))
            # print('========================================')
            x = torch.cat((x, freq_feat), dim=-1)
        elif self.acoustic:
            acu_feat = self.acou_feat_network(acu_feat)
            x = torch.cat((x, acu_feat), dim=-1)

        feat = self.down1(x.permute(0,2,1))
        feat = self.down2(feat)
        feat = self.down3(feat)

        enc = self.encoder(feat.permute(0,2,1)).permute(0,2,1) + feat

        feat = self.up1(enc)
        feat = self.up2(feat)

        out = self.sigmoid(self.fc(feat.permute(0,2,1)))

        return out

class SingleP(nn.Module):
    def __init__(self, hidden_dim, n_layers, frequency=True, acoustic=True):
        super(SingleP, self).__init__()

        self.frequency = frequency
        self.acoustic = acoustic

        self.compute_feature = compute_feature()

        if frequency:
            self.freq_feat_network = FrequencyCNN()
        if acoustic:
            self.acou_feat_network = OpenSmileCNN()

        self.lstm = nn.LSTM(20, hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.1)

        self.fc = nn.Linear(hidden_dim, 1)
        self.score = nn.Sigmoid()

    def forward(self, x, freq_feat, acu_feat):
        x = self.compute_feature(x)

        if self.frequency and self.acoustic:
            freq_feat = self.freq_feat_network(freq_feat)
            acu_feat = self.acou_feat_network(acu_feat)

            x = torch.cat((x, freq_feat, acu_feat), dim=-1)
        elif self.frequency:
            freq_feat = self.freq_feat_network(freq_feat)
            # print('========================================')
            # print('freq_feat: ', freq_feat)
            # print('isnan: ', torch.any(torch.isnan(freq_feat)))
            # print('========================================')
            x = torch.cat((x, freq_feat), dim=-1)
        elif self.acoustic:
            acu_feat = self.acou_feat_network(acu_feat)
            x = torch.cat((x, acu_feat), dim=-1)

        out, _ = self.lstm(x)
        # print('out: ', out)
        # print('isnan: ', torch.any(torch.isnan(out)))
        # print('========================================')
        out = self.score(self.fc(out))

        # print('fc: ', out)
        # print('isnan: ', torch.any(torch.isnan(out)))
        # print('========================================')

        return out

class SingleP_Conformer(nn.Module):
    def __init__(self, conformer_class, d_ffn, n_head, enc_layers, frequency=False, acoustic=False):
        super(SingleP_Conformer, self).__init__()

        self.compute_feature = compute_feature()
        self.conformer = Conformer(num_classes=conformer_class, input_dim=12, encoder_dim=d_ffn, num_attention_heads=n_head, num_encoder_layers=enc_layers)
        
        self.upconv = nn.Sequential(nn.Upsample(1500),
                                    nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                    nn.ReLU(),
                                    nn.Upsample(2000),
                                    nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                    nn.ReLU(),
                                    nn.Upsample(2500),
                                    nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                    nn.ReLU(),
                                    nn.Upsample(3000),
                                    nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                    nn.ReLU())        
        self.fc = nn.Linear(conformer_class, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, wave, freq_feat, acu_feat, input_lengths=3000):
        feature = self.compute_feature(wave)
      
        out, _ = self.conformer(feature, input_lengths)
     
        out = self.upconv(out.permute(0,2,1))
      
        out = self.sigmoid(self.fc(out.permute(0,2,1)))

        return out


