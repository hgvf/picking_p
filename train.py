import argparse
import os
import numpy as np
import logging
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from optimizer import *
from large_dataset import earthquake
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    
    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--clip_norm', type=float, default=0.01)
    parser.add_argument('--patience', type=float, default=5)
    parser.add_argument('--noam', type=bool, default=False)
    parser.add_argument('--warmup_step', type=int, default=1500)

    # save_path
    parser.add_argument("--save_path", type=str, default='tmp')
    parser.add_argument("--valid_step", type=int, default=3000)
    parser.add_argument('--valid_on_training', type=bool, default=False)

    parser.add_argument('--earthquake_weight', type=int, default=5)

    # model hyperparameters
    # For EQT
    parser.add_argument('--lstm_blocks', type=int, default=3)

    # For Transformer sliding window
    parser.add_argument('--sliding_window', type=bool, default=False)
    parser.add_argument('--d_ffn', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--window_size', type=int, default=30)

    # For LSTM
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=2)

    # For Conformer
    parser.add_argument('--conformer_class', type=int, default=8)

    # dataset hyperparameters
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--fixed_trigger', type=bool, default=False)
    parser.add_argument('--fixed_trigger_point', type=int, default=300)
    parser.add_argument('--low_intensity', type=bool, default=False)
    parser.add_argument('--triangular', type=bool, default=False)
    parser.add_argument('--gaussian', type=bool, default=False)
    parser.add_argument('--upsample', type=bool, default=False)
    parser.add_argument('--zscore', type=bool, default=False)
    parser.add_argument('--wiener', type=bool, default=False)
    parser.add_argument('--frequency', type=bool, default=False)
    parser.add_argument('--acoustic', type=bool, default=False)

    opt = parser.parse_args()

    return opt

def toLine(train_loss, valid_loss, epoch, n_epochs, isFinish):
    token = "Eh3tinCwQ87qfqD9Dboy1mpd9uMavhGV9u5ohACgmCF"

    if not isFinish:
        message = 'Epoch ['+ str(epoch) + '/' + str(n_epochs) + '] train_loss: ' + str(train_loss) +', valid_loss: ' +str(valid_loss)
    else:
        message = 'Finish training...'

    try:
        url = "https://notify-api.line.me/api/notify"
        headers = {
            'Authorization': f'Bearer {token}'
        }
        payload = {
            'message': message
        }
        response = requests.request(
            "POST",
            url,
            headers=headers,
            data=payload
        )
        if response.status_code == 200:
            print(f"Success -> {response.text}")
    except Exception as e:
        print(e)

def noam_optimizer(model, lr, warmup_step, device):
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    o = {}
    o["method"] = 'adam'
    o['lr'] = lr
    o['max_grad_norm'] = 0
    o['beta1'] = 0.9
    o['beta2'] = 0.999
    o['decay_method'] = 'noam'
    o['warmup_steps'] = warmup_step
    optimizer = build_optim(o, model, False, device)

    return optimizer

def train(model, optimizer, dataloader, valid_loader, device, cur_epoch, opt, output_dir):
    model.train()
    train_loss = 0.0
    min_loss = 1000

    train_loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, (data, target, freq_feat, acu_feat, isEarthquake) in train_loop:
        data, target = data.to(device), target.to(device)

        #out = model(data.permute(0,2,1))[1].squeeze().to(device)
        out = model(data, freq_feat, acu_feat).squeeze().to(device)

        loss = loss_function_bce(out, target.squeeze(), opt.triangular, opt.gaussian, isEarthquake, opt.earthquake_weight, device)
        loss = loss / opt.gradient_accumulation

        loss.backward()

        if ((idx+1) % opt.gradient_accumulation == 0) or ((idx+1) == len(dataloader)):
            nn.utils.clip_grad_norm_(model.parameters(), opt.clip_norm)
            optimizer.step()

            model.zero_grad()
            # optimizer.zero_grad()
        
        train_loss = train_loss + loss.detach().cpu().item()*opt.gradient_accumulation
        train_loop.set_description(f"[Train Epoch {cur_epoch+1}/{opt.epochs}]")
        train_loop.set_postfix(loss=loss.detach().cpu().item()*opt.gradient_accumulation)
        
        # valid & save model
        if (idx+1) % opt.valid_step == 0 and opt.valid_on_training:
            valid_loss = valid_on_training(model, valid_loader, device, cur_epoch, opt)
            model.train()

            if valid_loss < min_loss:
                min_loss = valid_loss

                # Saving model
                targetPath = os.path.join(output_dir, 'model.pt')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer,
                    'min_loss': min_loss,
                    'epoch': epoch
                }, targetPath)

            print('[Valid_on_training] epoch: %d -> loss: %.4f' %(cur_epoch+1, valid_loss))
            logging.info('[Valid_on_training] epoch: %d -> loss: %.4f' %(cur_epoch+1, valid_loss))
            print('Learning rate: %f' %(optimizer.learning_rate))

    train_loss = train_loss / (len(dataloader))
    return train_loss

def draw(wave, gt, save=False, name=None):
    plt.figure(figsize=(15, 10))
    plt.subplot(411)
    plt.title('Z')
    plt.plot(wave[0])
    
    plt.subplot(412)
    plt.title('N')
    plt.plot(wave[1])
    
    plt.subplot(413)
    plt.title('E')
    plt.plot(wave[2])
    
    plt.subplot(414)
    plt.title('Gt')
    plt.plot(gt)
    
    if save:
        if name is not None:
            plt.savefig('./'+name+'.png', dpi=300)
        else:
            plt.savefig('./test.png', dpi=300)
    else:
        plt.show()

def valid_on_training(model, dataloader, device, cur_epoch, opt):
    model.eval()
    dev_loss = 0.0

    valid_loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, (data, target, freq_feat, acu_feat, isEarthquake) in valid_loop:
        try:
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                #out = model(data.permute(0,2,1))[1].squeeze().to(device)
                out = model(data, freq_feat, acu_feat).squeeze().to(device)

            loss = loss_function_bce(out, target.squeeze(), opt.triangular, opt.gaussian, isEarthquake, opt.earthquake_weight, device)

            dev_loss = dev_loss + loss.detach().cpu().item()
            
            valid_loop.set_description(f"[Valid Epoch {cur_epoch+1}/{opt.epochs}]")
            valid_loop.set_postfix(loss=loss.detach().cpu().item())

            if (idx+1) % 500 == 0:
                break
        except Exception as e:
            print('Exception: ', e)

    valid_loss = dev_loss / 500
    return valid_loss

def valid(model, dataloader, device, cur_epoch, opt):
    model.eval()
    dev_loss = 0.0

    valid_loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, (data, target, freq_feat, acu_feat, isEarthquake) in valid_loop:
        try:
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                #out = model(data.permute(0,2,1))[1].squeeze().to(device)
                out = model(data, freq_feat, acu_feat).squeeze().to(device)

            loss = loss_function_bce(out, target.squeeze(), opt.triangular, opt.gaussian, isEarthquake, opt.earthquake_weight, device)

            dev_loss = dev_loss + loss.detach().cpu().item()
            
            valid_loop.set_description(f"[Valid Epoch {cur_epoch+1}/{opt.epochs}]")
            valid_loop.set_postfix(loss=loss.detach().cpu().item())
        except Exception as e:
            print('Exception: ', e)

    valid_loss = dev_loss / (len(dataloader))
    return valid_loss

if __name__ == '__main__':
    opt = parse_args()

    output_dir = os.path.join('./results', opt.save_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    log_path = os.path.join(output_dir, 'train.log')
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        filename=log_path, 
                        filemode='a', 
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S",)
    logging.info('start training')
    logging.info('configs: ')
    logging.info(opt)
    logging.info('======================================================')

    # 設定 device (opt.device = 'cpu' or 'cuda:X')
    if opt.device[:4] == 'cuda':
        gpu_id = opt.device[-1]
        #os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        torch.cuda.set_device(opt.device)
        device = torch.device(opt.device)
    else:
        print('device: cpu')
        device = torch.device('cpu')

    print('Save path: ', opt.save_path)
    logging.info('save path: %s' %(opt.save_path))
    logging.info('device: %s'%(device))

    # load model
    model = load_model(opt, device, opt.frequency, opt.acoustic)
    # checkpoint = torch.load("/mnt/nas5/weiwei/picking_p/results/transformer_pos_500_balance/model.pt")
    # model.load_state_dict(checkpoint['model'])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('trainable parameters: %d' %(trainable))
    print('trainable parameters: %d' %(trainable))

    # load dataset
    print('loading dataset...')
    train_set = earthquake('train', low_inten=opt.low_intensity, triangular=opt.triangular, z_score_normalize=opt.zscore, upsample=opt.upsample
                            , wiener=opt.wiener, frequency=opt.frequency, acoustic=opt.acoustic, window_size=opt.window_size, sliding_window=opt.sliding_window,
                            gaussian=opt.gaussian, aug=opt.aug, fixed_trigger=opt.fixed_trigger, fixed_trigger_point=opt.fixed_trigger_point)
    valid_set = earthquake('valid', low_inten=opt.low_intensity, triangular=opt.triangular, z_score_normalize=opt.zscore, upsample=opt.upsample
                            , wiener=opt.wiener, frequency=opt.frequency, acoustic=opt.acoustic, window_size=opt.window_size, sliding_window=opt.sliding_window,
                            gaussian=opt.gaussian, aug=opt.aug, fixed_trigger=opt.fixed_trigger, fixed_trigger_point=opt.fixed_trigger_point)

    # dataloader
    print('creating dataloader...')
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    valid_loader = DataLoader(dataset=valid_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    print('loading optimizer & scheduler...')
    if opt.noam:
        optimizer = noam_optimizer(model, opt.lr, opt.warmup_step, device)
    else:
        optimizer = optim.Adam(model.parameters(), opt.lr)

    # load checkpoint
    if opt.resume_training:
        logging.info('Resume training...')
        checkpoint = torch.load(opt.save_path+'model.pt')

        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optimizer']
        min_loss = checkpoint['min_loss']
        logging.info('resume training at epoch: %d' %(init_epoch))
    else:
        init_epoch = 0
        min_loss = 100000

    early_stop_cnt = 0
    for epoch in range(init_epoch, opt.epochs):
        train_loss = train(model, optimizer, train_loader, valid_loader, device, epoch, opt, output_dir)
        valid_loss = valid(model, valid_loader, device, epoch, opt)
    
        print('[Train] epoch: %d -> loss: %.4f' %(epoch+1, train_loss))
        print('[Eval] epoch: %d -> loss: %.4f' %(epoch+1, valid_loss))
        print('Learning rate: %f' %(optimizer.learning_rate))
        logging.info('[Train] epoch: %d -> loss: %.4f' %(epoch+1, train_loss))
        logging.info('[Eval] epoch: %d -> loss: %.4f' %(epoch+1, valid_loss))
        logging.info('======================================================')

        # Line notify
        toLine(train_loss, valid_loss, epoch, opt.epochs, False)

        # Early stopping
        if valid_loss < min_loss:
            min_loss = valid_loss

            # Saving model
            targetPath = os.path.join(output_dir, 'model.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer,
                'min_loss': min_loss,
                'epoch': epoch
            }, targetPath)

            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == opt.patience:
            logging.info('early stopping...')

            toLine(train_loss, valid_loss, epoch, opt.epochs, True)
            break

    print('Finish training...')
    toLine(train_loss, valid_loss, epoch, opt.epochs, True)