import cv2
import numpy as np
import h5py
import os
## load depth npy
t_new=0
path = "/home/martin-barry/Desktop/HES-SO/Event-Based/Surreal/05_05/05_05_c0002"
## load h5 
with h5py.File(os.path.join(path,"vid_slomo_depth.h5"), 'r') as f:
    depths =  f['vids'][:]
with h5py.File(os.path.join(path,"vid_slomo.h5"), 'r') as f:
    original =  f['vids'][:]
with h5py.File(os.path.join(path,"dvs.h5"), 'r') as f:
    dvs =  f['vids'][:]
spikes_hist = np.zeros_like(depths)

## video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.avi', fourcc, 30, (depths[0].shape[1]*2, depths[0].shape[0]))
for i in range(len(depths)):
    t_old = t_new
    t_new = t_old + 1/(30*12)
    spikes = dvs[(dvs[:, 0] >= t_old )* (dvs[:, 0] < t_new)]
    x = spikes[:, 1].astype(int)
    y = spikes[:, 2].astype(int)
    spikes_hist[i, y, x] = 255
    depth = depths[i]
    
    merged = cv2.addWeighted(depth, 0.5, spikes_hist[i], 0.5, 0)
    full = np.concatenate((original[i], merged), axis=1)
    full = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)
    ## append to video
    output.write(full)

    cv2.imshow('full', full)
    ## save video 
    
    # cv2.imshow('spikes',spikes_hist[i])
    cv2.waitKey(2)
output.release()
cv2.destroyAllWindows()
