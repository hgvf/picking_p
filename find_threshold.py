import numpy as np
import os
import argparse
import pandas as pd
import math
import logging
import pickle
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
    parser.add_argument("--threshold_type", type=str, default='all')
    parser.add_argument('--threshold_prob_start', type=float, default=0.45)
    parser.add_argument('--threshold_prob_end', type=float, default=0.9)
    parser.add_argument('--threshold_trigger_start', type=int, default=20)
    parser.add_argument('--threshold_trigger_end', type=int, default=65)
    parser.add_argument('--sample_tolerant', type=int, default=50)

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

    # dataset hyperparameters
    parser.add_argument('--low_intensity', type=bool, default=False)
    parser.add_argument('--triangular', type=bool, default=False)

    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--batch_size", type=int, default=100)
    
    opt = parser.parse_args()

    return opt

def evaluation(pred, gt, threshold_prob, threshold_trigger, sample_tolerant, mode):
    tp=fp=tn=fn = 0
    diff = []
    abs_diff = []
    isSingle = False

    if pred.shape[0] >= 2000:
        isSingle = True

    for i in range(pred.shape[0]):
        pred_isTrigger = False
        gt_isTrigger = False

        gt_trigger = 0

        if not isSingle:
            c = np.where(gt[i] == 1) 
        else:
            c = np.where(gt == 1)
        # if gt[i].any():
        if len(c[0]) > 0:
            gt_isTrigger = True
            
            gt_trigger = c[0][0]

        if mode == 'avg':
            if not isSingle:
                a = pd.Series(pred[i])  
            else:
                a = pd.Series(pred)
            win_avg = a.rolling(window=threshold_trigger).mean().to_numpy()

            c = np.where(win_avg >= threshold_prob, 1, 0)

            pred_trigger = 0
            if c.any():
                tri = np.where(c==1)
                pred_trigger = tri[0][0]-threshold_trigger+1
                pred_isTrigger = True
                
        elif mode == 'continue':
            pred[i] = np.where(pred[i] >= threshold_prob, 1, 0)
            
            a = pd.Series(pred[i])    
            data = a.groupby(a.eq(0).cumsum()).cumsum().tolist()

            if threshold_trigger in data:
                pred_trigger = data.index(threshold_trigger)-threshold_trigger+1
                pred_isTrigger = True
            else:
                pred_trigger = 0

        left_edge = (gt_trigger - sample_tolerant) if (gt_trigger - sample_tolerant) >= 0 else 0
        right_edge = (gt_trigger + sample_tolerant) if (gt_trigger + sample_tolerant) <= 3000 else 2999
        
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

        if isSingle:
            break
    
    return tp, fp, tn, fn, diff, abs_diff, pred_trigger, gt_trigger

def inference(model, test_loader, mode, device, opt, threshold_prob, threshold_trigger):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    test_loss = 0.0
    step = 0
    diff = []
    abs_diff = []

    model.eval()
    with tqdm(test_loader) as epoch:
        for data, target in epoch:
            data, target = data.to(device), target.to(device)
            
            with torch.no_grad():
                out = model(data).squeeze().to(device)
                #out = model(data.permute(0,2,1))[1].squeeze().to(device)

                if opt.sliding_window:
                    loss = loss_function_bce(out, target.squeeze(), opt.triangular, device, opt.window_size)
                else:
                    loss = loss_function_bce(out, target.squeeze(), opt.triangular, device)

                test_loss += loss.detach().cpu().item()
            
            epoch.set_postfix(loss=loss.item())

            out = out.detach().cpu().numpy().squeeze()
            target = target.detach().cpu().numpy().squeeze()
            target = target[:, opt.window_size-3000:]

            a, b, c, d, e, f, trigger, gt_trigger = evaluation(out, target, threshold_prob, threshold_trigger, opt.sample_tolerant, mode)
             
            tp += a
            fp += b
            tn += c
            fn += d
            diff += e
            abs_diff += f

    # statisical  
    precision = tp / (tp+fp) if (tp+fp) != 0 else 0
    recall = tp / (tp+fn) if (tp+fn) != 0 else 0
    fpr = fp / (tn+fp) if (tn+fp) != 0 else 100
    fscore = 2*precision*recall / (precision+recall) if (precision+recall) != 0 else 0
    test_loss /= len(test_loader)

    logging.info('======================================================')
    logging.info('threshold_prob: %.2f' %(threshold_prob))
    logging.info('threshold_trigger: %d' %(threshold_trigger))
    logging.info('TPR=%.4f, FPR=%.4f, Precision=%.4f, Fscore=%.4f, loss= %.4f' %(recall, fpr, precision, fscore, test_loss))
    logging.info('tp=%d, fp=%d, tn=%d, fn=%d' %(tp, fp, tn, fn))
    logging.info('abs_diff=%.4f, diff=%.4f' %(np.mean(abs_diff), np.mean(diff)))

    return fscore, abs_diff, diff

if __name__ == '__main__':
    opt = parse_args()

    output_dir = os.path.join('./results', opt.save_path)
    log_path = os.path.join(output_dir, 'threshold.log')
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        filename=log_path, 
                        filemode='a', 
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S",)
    print(opt.save_path)

    # 設定 device (opt.device = 'cpu' or 'cuda:X')
    if opt.device[:4] == 'cuda':
        gpu_id = opt.device[-1]
        #os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        torch.cuda.set_device(opt.device)
        device = torch.device(opt.device)
    else:
        print('device: cpu')
        device = torch.device('cpu')

    # dataset
    print('loading dataset...')
    test_set = earthquake('test', low_inten=opt.low_intensity, triangular=opt.triangular)
    test_loader = DataLoader(dataset=test_set, batch_size=opt.batch_size, shuffle=True, num_workers=8)

    # model
    print('loading model...')
    model = load_model(opt).to(device)

    model_path = os.path.join(output_dir, 'model.pt')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # start finding
    max_fscore = 0.0
    cnt = 0
    front = opt.sample_tolerant
    back = opt.sample_tolerant

    if opt.threshold_type == 'all':
        mode = ['avg', 'continue']  # avg, continue
    elif opt.threshold_type == 'avg':
        mode = ['avg']
    elif opt.threshold_type == 'continue':
        mode = ['continue']

    for m in mode:
        logging.info('======================================================')
        logging.info('Mode: %s' %(m))
        logging.info('======================================================')
        for prob in tqdm(np.arange(opt.threshold_prob_start, opt.threshold_prob_end, 0.05)):  # (0.45, 0.85)
            max_fscore = 0.0
            cnt = 0
            
            for trigger in np.arange(opt.threshold_trigger_start, opt.threshold_trigger_end, 5): # (10, 55)
                fscore, abs_diff, diff = inference(model, test_loader, m, device, opt, prob, trigger)
                print('prob: %.2f, trigger: %d, fscore: %.4f' %(prob, trigger, fscore))

                if fscore > max_fscore:
                    max_fscore = fscore
                    cnt = 0

                    with open(os.path.join(output_dir, 'abs_diff.pkl'), 'wb') as f:
                        pickle.dump(abs_diff, f)

                    with open(os.path.join(output_dir, 'diff.pkl'), 'wb') as f:
                        pickle.dump(diff, f)
                else:
                    cnt += 1

                if cnt == 3:
                    break