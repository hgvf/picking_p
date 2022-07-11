import torch
import torch.nn as nn
import seisbench.models as sbm
import numpy as np

from model import *

# Weighted bce loss function
def loss_function_bce(out, target, triangular, gaussian, isEarthquake, earthquake_weight, device='cpu'):
    
    with torch.autograd.set_detect_anomaly(True):
        if triangular or gaussian:
            weights = torch.add(torch.mul(target, 20), 1)
        else:
            weights = torch.add(torch.mul(target, 2), 1)

        earthquake_weight = isEarthquake * earthquake_weight
        weights = weights * earthquake_weight[:, None].to(device)

        # if torch.any(torch.isnan(out)):
        #     print('nan')
        # if torch.any(torch.isinf(out)):
        #     print('inf')
        # out[torch.isnan(out)==1] = 0
        # out[torch.isinf(out)==1] = 1
    return nn.BCELoss(weight=weights)(out, target)

def load_model(opt, device, frequency, acoustic):
    #model = sbm.EQTransformer(in_samples=3000, classes=1, phases='P', lstm_blocks=opt.lstm_blocks).to(device)
    # model = SingleP_transformer_window(opt.d_ffn, opt.nhead, opt.enc_layers, opt.dropout, opt.window_size, frequency, acoustic).to(device)
    # model = SingleP_CNNTransformer(opt.d_ffn, opt.nhead, opt.enc_layers, frequency, acoustic).to(device)
    # model = SingleP_transformer_window_short(opt.d_ffn, opt.nhead, opt.enc_layers, opt.window_size).to(device)
    # model = SingleP(opt.hidden_dim, opt.n_layers, frequency, acoustic).to(device)
    model = SingleP_Conformer(opt.conformer_class, opt.d_ffn, opt.nhead, opt.enc_layers, frequency, acoustic).to(device)

    return model
