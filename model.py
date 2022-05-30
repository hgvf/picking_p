import torch
import torch.nn as nn

from calc_feature import *

class SingleP_transformer_window(nn.Module):
    def __init__(self, d_ffn, n_head, enc_layers, dec_layers, dropout, window_size):
        super(SingleP_transformer_window, self).__init__()

        self.window_size = window_size
        self.compute_feature = compute_feature()

        self.transformer = nn.TransformerEncoderLayer(d_model=12, nhead=n_head, dim_feedforward=d_ffn, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformer, num_layers=enc_layers)
        
        #self.rnnDecoder = nn.LSTM(12, 24, batch_first=True, dropout=0.1, num_layers=2)
        #self.fc = nn.Linear(24, 1)
        self.fc = nn.Linear(12, 1)
        self.score = nn.Sigmoid()

    def forward(self, input_x):
        # compute 12-dim features
        input_x = self.compute_feature(input_x)
        print(input_x.shape)
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

class SingleP(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super(SingleP, self).__init__()

        self.compute_feature = compute_feature()
        self.lstm = nn.LSTM(12, hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.1)

        self.fc = nn.Linear(hidden_dim, 1)
        self.score = nn.Sigmoid()

    def forward(self, x):
        x = self.compute_feature(x)

        out, _ = self.lstm(x)
        out = self.score(self.fc(out))

        return out