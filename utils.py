import torch
import torch.nn as nn
import seisbench.models as sbm

from model import *

# Weighted bce loss function
def loss_function_bce(out, target, triangular, device = 'cpu', window_size=None):
    if window_size is not None:
        if target.shape[0] > 1000:
            seq_len = target.shape[0]
            target = target[window_size-seq_len:].to(device)
        else:
            seq_len = target.shape[1]
            target = target[:, window_size-seq_len:].to(device)

    if triangular:
        weights = torch.add(torch.mul(target, 75), 1).to(device)
    else:
        weights = torch.add(torch.mul(target, 2), 1).to(device)

    return nn.BCELoss(weight=weights)(out, target)
    # return nn.BCELoss()(out, target)

def load_model(opt):
    #model = sbm.EQTransformer(in_samples=3000, classes=1, phases='P', lstm_blocks=opt.lstm_blocks)
    model = SingleP_transformer_window(opt.d_ffn, opt.nhead, opt.enc_layers, opt.dec_layers, opt.dropout, opt.window_size)
    #model = SingleP(opt.hidden_dim, opt.n_layers)

    return model
