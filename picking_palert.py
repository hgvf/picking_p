import time
import numpy as np
import os
import torch
import PyEW, logging
import matplotlib.pyplot as plt
import argparse
import seisbench.models as sbm
from itertools import compress
from datetime import datetime, timedelta
from multiprocessing import Process, Manager
import multiprocessing
from collections import deque
from utils import *
from read_sta import *
from model import *

# 畫波型
def draw(z, n, e, station, tri_time, pred=None):
    if pred is not None:
        plt.subplot(411)
        plt.title('Z')
        plt.plot(z)

        plt.subplot(412)
        plt.title('N')
        plt.plot(n)

        plt.subplot(413)
        plt.title('E')
        plt.plot(e)

        plt.subplot(414)
        plt.title('Pred')
        plt.plot(pred)
        plt.ylim([0.0, 1.0])
    else:
        plt.subplot(311)
        plt.title('Z')
        plt.plot(z)

        plt.subplot(312)
        plt.title('N')
        plt.plot(n)

        plt.subplot(313)
        plt.title('E')
        plt.plot(e)

    plt.savefig('./plot/'+station+'_'+str(tri_time)+'.png', dpi=100)
    plt.clf()
    #plt.show()

def WaveSaver(MyModule, to_predict_waveform, station_wave):
    waveform = {}
    station = 'none'

    while True:
        # get raw waveform from WAVE_RING
        wave = MyModule.get_wave(0)
        
        threeAxis = []
        isLast = False
        #print(wave)

        if wave == {}:
            continue

        # 從 Z 軸開始蒐集
        if wave['channel'] == 'HLZ':
            #print('Z')
            station = wave['station']
            channel = wave['channel']
        elif station != 'none':
            # 蒐集同測站的 N, E 軸波型
            if wave['channel'] == 'HLN' and wave['station'] == station:
                #print('N')
                pass
            elif wave['channel'] == 'HLE' and wave['station'] == station:
                #print('E')
                isLast = True
        else:
            continue

        station = wave['station']
        channel = wave['channel']

        # if 三軸波型都有正常收到，就存進 waveform 裡
        if (station+"_"+channel+"_data") in waveform:
            waveform[station+"_"+channel+"_data"].append(wave['data'][:])

            # 蒐集這個 waveform 中屬於這個測站的波型，取最新的 30 秒來去做預測
            # to_predict_waveform: multiprocessing 的變數，多個 processes 共享
            if len(waveform[station+"_"+channel+"_data"]) >= 30:
                to_predict_waveform[station+"_"+channel+"_data"] = np.reshape(np.array(waveform[station+"_"+channel+"_data"])[-30:], -1)

                if isLast:
                    channel = 'HLZ'
                    station_wave.append(station+"_"+channel+"_"+str(wave['startt']))
        else:
            waveform[station+"_"+channel+"_data"] = deque(maxlen=40)
            waveform[station+"_"+channel+"_data"].append(wave['data'][:])

        '''
        if wave == {} or wave['channel'] != 'HLZ':
            continue
        threeAxis.append(wave)
        tmp_sta = wave['station']
        
        # get three axis waveform at once
        while True:
            tmp_wave = MyModule.get_wave(0)
            if tmp_wave != {} and tmp_wave['station'] == tmp_sta:
                threeAxis.append(tmp_wave)
                if len(threeAxis) >= 3:
                    break
        #print(threeAxis)
        
        # save three axis waveform and station info to a dictionary
        for j in range(3):
            station = threeAxis[j]['station']
            channel = threeAxis[j]['channel']
           
            if (station+"_"+channel+"_data") in waveform:
                waveform[station+"_"+channel+"_data"].append(threeAxis[j]['data'][:])

                if len(waveform[station+"_"+channel+"_data"]) >= 30:
                    to_predict_waveform[station+"_"+channel+"_data"] = np.reshape(np.array(waveform[station+"_"+channel+"_data"])[-30:], -1)

                    if j == 0:
                        if channel == 'HLZ':
                            station_wave.append(station+"_"+channel+"_"+str(threeAxis[j]['startt']))
            else:
                waveform[station+"_"+channel+"_data"] = deque(maxlen=40)
                waveform[station+"_"+channel+"_data"].append(threeAxis[j]['data'][:])
        '''

def Picker(device, waveform, station_wave):
    # ================================ load pretrained model ================================= #
    print('loading model...')
    #model_path = "/mnt/nas4/weiwei/earthquake_RNN/results/transformer_window_all/model.pt"
    #model = SingleP_transformer_window(128, 4, 6, 50).to(device)

    #model_path = "/mnt/nas4/weiwei/earthquake_RNN/results/rnn0_all/model.pt"
    #model = SingleP(32, 2, False).to(device)

    #model_path = "/mnt/nas4/weiwei/picking_p/results/EQT_lowIntensity/model.pt"
    model_path = model_path = "/mnt/nas4/weiwei/picking_p/results/EQT_triangular_allIntensity/model.pt"
    model = sbm.EQTransformer(in_samples=3000, classes=1, phases='P').to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    # ================================ load pretrained model ================================= #

    # palert station list, ex. east: 東部測站座標, east_sta: 東部測站名 (可以先忽略)
    north, north_sta, middle, middle_sta, south, south_sta, east, east_sta = get_palert_sta()

    # list & pre-defined variables & sleep for 3 s
    # time.sleep(3)
    sample_rate = 100
    wave_to_model = []
    lat = []
    lon = []
    channel_order = ['Z', 'N', 'E']

    # get station info
    info = get_StationInfo()
    
    # listen RING
    pick_cnt = 0
    while True:
        # 區域型測站 (可以先忽略)
        north_trigger = 0
        middel_trigger = 0
        south_trigger = 0
        east_trigger = 0

        # 對應到 WaveSaver 的 to_predict_waveform
        # 檢查有沒有需要預測的波型
        if len(station_wave) == 0:
            #print('waveform are less than 30s')
            continue
        
        # pop waiting list
        tmp = station_wave.pop(0).split("_")
        station = tmp[0]
        channel = tmp[1]
        startt = tmp[2]
        
        # channel name
        channel = [station+"_"+channel[:-1]+channel_order[i]+"_data" for i in range(3)]

        # check whether 3 channel in waveform list
        if [True if i in waveform.keys() else False for i in channel].count(True) != 3:
            continue

        # append station & channel & lat & lon into lists
        s, c = [], []
        s.append(station)
        c.append(channel)

        # concat the 3-axis waveforms
        wave = [waveform[c]/16.718 for idx, c in enumerate(channel)]
        wave_to_model.append(torch.cat((torch.FloatTensor(wave[0]).unsqueeze(-1), torch.FloatTensor(wave[1]).unsqueeze(-1), torch.FloatTensor(wave[2]).unsqueeze(-1)), axis=-1).squeeze())
        
        # 堆疊多個測站的波型
        if len(wave_to_model) == 300 or (len(station_wave) == 0 and len(wave_to_model)>0): 
            wave = torch.FloatTensor(torch.stack(wave_to_model))
            #lon, lat = np.array(lon), np.array(lat)
            wave_to_model = []
        else:
            continue
        
        # Mask 掉那些沒有出現在 nsta24 的測站
        #mask = (lon!=-1)
        #lon, lat = lon[mask], lat[mask]
        #wave = wave[mask]
        #recorded_startt = recorded_startt[mask]

        # ================================ start predicting ================================= #
        # TODO: 之後直接套用 earthworm_api 的 "predict()"

        # Compute 12-dim features: 算 features 時要用 e, n, z 順序下去算
        # (batch_size, seg_len, 12)
        wave = wave.to(device)
        #wave = compute_feat(wave)

        #print('wave: ', wave.shape)
        # ========================================
        with torch.no_grad():
            # prepare for prdicting p-wave
            #out3000 = model(wave).detach().squeeze().cpu().numpy()
            out3000 = model(wave.permute(0,2,1))[1].detach().squeeze().cpu().numpy()
        wave = wave.detach().cpu().numpy()
        
        draw(wave[0, :, -3], wave[0, :, -2], wave[0, :, -1], 'palert', pick_cnt, out3000[0])
        # 套入 threshold 判斷有無觸發
        #isTrigger3000, trigger_time = trigger(out3000, 50, 0.7, sample_rate, time.time()-time_before)
        isTrigger3000, trigger_time = trigger(out3000, 50, 0.7, sample_rate, time.time())
        # ================================ start predicting ================================= #

        # ================================ sending reports ================================= #
        # 計算有幾個測站需要發報，判斷有沒有滿足發報條件: ex. 20 測站 pick 到，就需要發報
        # For CWB
        #lon3000 = list(compress(lon, isTrigger3000))
        #lat3000 = list(compress(lat, isTrigger3000))
        #coord3000 = list(set(zip(lon3000, lat3000)))
        
        #n_trigger3000 = len(coord3000)
        #print('trigger: ', n_trigger3000)
        print('trigger: ', isTrigger3000.count(True))

        if True in isTrigger3000:
            tt = isTrigger3000.index(True)
            draw(wave[tt, :, -3], wave[tt, :, -2], wave[tt, :, -1], 'palert', pick_cnt, out3000[tt])
            print('pick_cnt: ', pick_cnt)
            #print('gt: ', p_picking_gt[tt])
            print('pred: ', trigger_time[tt])

        # 畫出有 pick 到的測站位置
        #if n_trigger3000 >= 10:
            #plot_taiwan('ensemble_'+str(pick_cnt), coord3000)

        pick_cnt += 1
        lon = []
        lat = []
        recorded_startt = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--time_before', type=float, default=0.0)
    opt = parser.parse_args()

    return opt

# ============================================================================================= #
if __name__ == '__main__':
    try:
        args = parse_args()

        # connect to earthworm, add WAVE_RING and PICK_RING
        print('connect to IMPORT_RING')
        MyModule = PyEW.EWModule(1000, 150, 255, 30.0, False)
        MyModule.add_ring(1007)     # IMPORT_RING
        #MyModule.add_ring(1037)      # TANK_RING

        # device
        if args.device[:4] == 'cuda':
            print('using gpu')
            device = torch.device(args.device)
        else:
            print('using cpu')
            device = torch.device('cpu')

        manager = Manager()
        waveform = manager.dict()
        station_wave = manager.list()
        
        # run the first thread to save raw waveform from WAVE_RING
        print('Creating wave_saver...')
        p1 = Process(target=WaveSaver, args=(MyModule, waveform, station_wave))
        p1.start()

        # run the second thread to listen PICK_RING and process data if a station is picked
        print('Creating picker...')
        p2 = Process(target=Picker, args=(args.device, waveform, station_wave))
        p2.start()

        p1.join()
        p2.join()

    except KeyboardInterrupt:
        MyModule.goodbye()
