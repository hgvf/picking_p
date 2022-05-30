import os

base_dir = "/mnt/nas3/earthquake_dataset_large"
dataset = ['Palert', 'TSMIP', 'CWBSN', 'STEAD/chunk1']
earthquake = []
noise = []

for datadir in dataset:
    if datadir == 'STEAD/chunk1':
        noise = os.listdir(os.path.join(base_dir, datadir))
    else:
        earthquake += os.listdir(os.path.join(base_dir, datadir))

intensity = {}
intensity['0'] = 0
intensity['1'] = 0
intensity['2'] = 0
intensity['3'] = 0
intensity['4'] = 0
intensity['5'] = 0
intensity['6'] = 0
intensity['7'] = 0

for f in earthquake:
    intensity[f[-4]] += 1

print(intensity)
