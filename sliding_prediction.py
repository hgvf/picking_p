import numpy as np
import torch
import matplotlib.pyplot as plt

def sliding_prediction(model, wave, window_size, wave_length):
    # create empty list to store the prediction
    pred = []
    for i in range(wave_length):
        pred.append([])

    # start predicting
    for i in tqdm(range(0, wave_length-window_size), total=wave_length-window_size):
        out = model(wave[i:i+window_size, :].permute(1,0), window_size).detach().squeeze().numpy()

        for j in range(window_size):
            pred[i+j].append(out[j])

    return pred

def median_postprocess(pred):
    res = []
    for i in range(len(pred)):
        res.append(np.median(pred[i]))

pred = sliding_prediction(model, wave,3000, 6000)
res = median_postprocess(pred)
