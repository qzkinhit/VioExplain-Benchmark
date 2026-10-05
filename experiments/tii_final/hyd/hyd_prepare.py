"""Prepare the UCI hydraulic condition monitoring data (Helwig et al. 2015, UCI id 447) for VioExplain.
Every sensor is resampled to 1 Hz by averaging non-overlapping blocks, so a load cycle is one window of 60 x 17.
Writes data/hydraulic/hyd_1hz.npz with X (2205, 60, 17), profile (2205, 5), sensor names."""
import numpy as np, hashlib, collections, json
D = '/path/to/vioexplain/data/hydraulic/'
SENS = ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 'FS1', 'FS2', 'TS1', 'TS2', 'TS3', 'TS4', 'VS1', 'CE', 'CP', 'SE']
cols = []
for s in SENS:
    a = np.loadtxt(D + s + '.txt', dtype=np.float64)
    f = a.shape[1] // 60
    cols.append(a.reshape(a.shape[0], 60, f).mean(2)); print(s, a.shape, 'factor', f, flush=True)
X = np.stack(cols, 2).astype(np.float32)
prof = np.loadtxt(D + 'profile.txt', dtype=np.float64)
np.savez_compressed(D + 'hyd_1hz.npz', X=X, profile=prof, sensors=np.array(SENS))
print('X', X.shape, 'profile', prof.shape)
for j, n in enumerate(['cooler', 'valve', 'pump', 'acc', 'stable']):
    print(n, dict(collections.Counter(prof[:, j].astype(int).tolist())))
