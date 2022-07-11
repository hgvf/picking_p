import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

import seisbench.models as sbm
from large_dataset import earthquake
from snr import snr_p
from calc import *
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", type=str, default='tmp')
    parser.add_argument('--plot', type=bool, default=False)

    # EQT model hyperparameters
    parser.add_argument('--lstm_blocks', type=int, default=3)

    # For Transformer sliding window
    parser.add_argument('--sliding_window', type=bool, default=False)
    parser.add_argument('--d_ffn', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--dec_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--window_size', type=int, default=30)

    # For LSTM
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=2)

    # For Conformer
    parser.add_argument('--conformer_class', type=int, default=8)

    parser.add_argument('--earthquake_weight', type=int, default=5)

    # dataset hyperparameters
    parser.add_argument('--low_intensity', type=bool, default=False)
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--fixed_trigger', type=bool, default=False)
    parser.add_argument('--fixed_trigger_point', type=int, default=300)
    parser.add_argument('--triangular', type=bool, default=False)
    parser.add_argument('--gaussian', type=bool, default=False)
    parser.add_argument('--upsample', type=bool, default=False)
    parser.add_argument('--zscore', type=bool, default=False)
    parser.add_argument('--wiener', type=bool, default=False)
    parser.add_argument('--frequency', type=bool, default=False)
    parser.add_argument('--acoustic', type=bool, default=False)

    # threshold
    parser.add_argument('--threshold_trigger', type=int, default=40)
    parser.add_argument('--sample_tolerant', type=int, default=50)
    parser.add_argument('--threshold_prob', type=float, default=0.55)
    parser.add_argument('--threshold_type', type=str, default='avg')

    opt = parser.parse_args()

    return opt

def plot(d, out, target, step, res, pred_trigger, gt_trigger, snr_cur, inten, plot_path):
    d = d.detach().numpy()

    plt.figure(figsize=(18, 25))
    plt.rcParams.update({'font.size': 18})
    plt.subplot(5,1,1)
    plt.plot(d[0, :, 0])
    plt.axvline(x=pred_trigger, color='r')
    plt.title('Z')

    plt.subplot(5,1,2)
    plt.plot(d[0, :, 1])
    plt.axvline(x=pred_trigger, color='r')
    plt.title('N')

    plt.subplot(5,1,3)
    plt.plot(d[0, :, 2])
    plt.axvline(x=pred_trigger, color='r')
    plt.title('E')

    plt.subplot(5,1,4)
    plt.plot(out)
    plt.ylim([-0.05, 1.05])
    plt.axvline(x=pred_trigger, color='r')
    pred_title = 'pred (' + str(pred_trigger) + ')'
    plt.title(pred_title)

    plt.subplot(5,1,5)
    plt.plot(target)
    plt.axvline(x=pred_trigger, color='r')
    gt_title = 'ground truth (' + str(gt_trigger) + ')'
    plt.title(gt_title)

    if snr_cur == 'noise':
        filename = res + '_' + str(step) + '_' + str(snr_cur) + '_' + str(inten) + '_' + str(pred_trigger) + '_' + str(gt_trigger) + '.png'
    else:
        filename = res + '_' + str(step) + '_' + str(round(snr_cur, 4)) + '_' + str(inten) + '_' + str(pred_trigger) + '_' + str(gt_trigger) + '.png'

    png_path = os.path.join(plot_path, filename)
    plt.savefig(png_path, dpi=300)
    plt.clf()
    plt.close()

def evaluation(pred, gt, threshold_prob, threshold_trigger, sample_tolerant, threshold_type):
    tp=fp=tn=fn=0
    pred_isTrigger = False
    gt_isTrigger = False
    diff = []
    abs_diff = []

    gt_trigger = 0
    if gt.any():
        c = np.where(gt == 1)
        gt_trigger = c[0][0]
        gt_isTrigger = True

    if threshold_type == 'avg':
        a = pd.Series(pred)    
        win_avg = a.rolling(window=threshold_trigger).mean().to_numpy()

        c = np.where(win_avg >= threshold_prob, 1, 0)

        pred_trigger = 0
        if c.any():
            tri = np.where(c==1)
            pred_trigger = tri[0][0]-threshold_trigger+1
            pred_isTrigger = True

    elif threshold_type == 'continue':
        pred = np.where(pred >= threshold_prob, 1, 0)
        
        a = pd.Series(pred)    
        data = a.groupby(a.eq(0).cumsum()).cumsum().tolist()

        if threshold_trigger in data:
            pred_trigger = data.index(threshold_trigger)-threshold_trigger+1
            pred_isTrigger = True
        else:
            pred_trigger = 0

    left_edge = (gt_trigger - sample_tolerant) if (gt_trigger - sample_tolerant) >= 0 else 0
    right_edge = (gt_trigger + sample_tolerant) if (gt_trigger + sample_tolerant) <= 3000 else 3000
    
    # case positive 
    if (pred_trigger >= left_edge) and (pred_trigger <= right_edge) and (pred_isTrigger) and (gt_isTrigger):
        tp += 1
    elif (pred_isTrigger):
        fp += 1

    # case negative
    if (not pred_isTrigger) and (gt_isTrigger):
        fn += 1
    elif (not pred_isTrigger) and (not gt_isTrigger):
        tn += 1

    if gt_isTrigger and pred_isTrigger:
        diff.append(pred_trigger-gt_trigger)
        abs_diff.append(abs(pred_trigger-gt_trigger))

    return tp, fp, tn, fn, diff, abs_diff, pred_trigger, gt_trigger

if __name__ == '__main__':
    opt = parse_args()
    device = torch.device('cpu')

    if opt.plot:
        print(opt.save_path)
        plot_path = os.path.join('./plot', opt.save_path)
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)

    # dataset
    print('loading dataset...')
    test_set = earthquake('test', low_inten=opt.low_intensity, triangular=opt.triangular, z_score_normalize=opt.zscore, upsample=opt.upsample
                            , wiener=opt.wiener, frequency=opt.frequency, acoustic=opt.acoustic, window_size=opt.window_size, sliding_window=opt.sliding_window,
                            gaussian=opt.gaussian, aug=opt.aug, fixed_trigger=opt.fixed_trigger, fixed_trigger_point=opt.fixed_trigger_point)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)

    # model
    print('loading model...')
    model = load_model(opt, device, opt.frequency, opt.acoustic)

    output_dir = os.path.join('./results', opt.save_path)
    model_path = os.path.join(output_dir, 'model.pt')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    model.eval()
    step_tp = step_fp = step_fn = step_tn = 0
    low = 0
    high = 0
    noise = 0
    with tqdm(test_loader) as epoch:
        for data, target, freq_feat, acu_feat, isEarthquake in epoch:
            isLow = False
            isHigh = False
            isNoise = False
            
            data, target = data.to(device), target.to(device)
      
            # if not noise waveform, calculate log(SNR)
            tri = torch.where(target[0] == 1)[0]
            if len(tri) != 0:
                snr_tmp = snr_p(data[0, :, 0].cpu().numpy(), target[0].cpu().numpy())

                if snr_tmp is None:
                    snr_cur = 'noise'
                else:
                    snr_cur = np.log10(snr_tmp)
                    
                inten, _, _ = calc_intensity(data[0, :, 0].numpy(), data[0, :, 1].numpy(), data[0, :, 2].numpy(), 'Acceleration', 100)

                if inten[-1] == 'g' or inten[-1] == 'k':
                    inten = inten[0]

                if int(inten) > 3:
                    high += 1
                    isHigh = True
                else:
                    low += 1
                    isLow = True
            else:
                snr_cur = 'noise'
                inten = 'noise'
                noise += 1
                isNoise = True

            with torch.no_grad():
                out = model(data, freq_feat, acu_feat).squeeze()
                #out = model(data.permute(0,2,1))[1].squeeze()
                
                loss = loss_function_bce(out, target.squeeze(), opt.triangular, opt.gaussian, isEarthquake, opt.earthquake_weight, device)
            
            epoch.set_postfix(loss=loss.item())

            out = out.detach().numpy()
            target = target.squeeze().detach().numpy()

            a, b, c, d, e, f, pred_trigger, gt_trigger = evaluation(out, target, opt.threshold_prob, opt.threshold_trigger, opt.sample_tolerant, opt.threshold_type)

            if a == 1:
                res = 'tp'
                step_tp += 1
                cur = step_tp
            elif b == 1:
                res = 'fp'
                step_fp += 1
                cur = step_fp
            elif c == 1:
                res = 'tn'
                step_tn += 1
                cur = step_tn
            else:
                res = 'fn'
                step_fn += 1
                cur = step_fn

            if opt.plot and low <= 40 and isLow:
                plot(data, out, target, cur, res, pred_trigger, gt_trigger, snr_cur, inten, plot_path)
            elif opt.plot and high <= 40 and isHigh:
                plot(data, out, target, cur, res, pred_trigger, gt_trigger, snr_cur, inten, plot_path)
            elif opt.plot and noise <= 20 and isNoise:
                plot(data, out, target, cur, res, pred_trigger, gt_trigger, snr_cur, inten, plot_path)
            elif opt.plot and step_fp <= 40 and res == 'fp':
                plot(data, out, target, cur, res, pred_trigger, gt_trigger, snr_cur, inten, plot_path)

            if low > 40 and (high > 40 or opt.low_intensity) and noise > 20 and step_fp > 40:
                print('finish plotting..')
                break