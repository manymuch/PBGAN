import os
import numpy as np
from PIL import Image

import scipy.io 
mat = scipy.io.loadmat('train_32x32.mat') # input mat
AllPxs = np.array(mat['X']).transpose(3,0,1,2)
# 73257x32x32x3

PicPath = './svhn' # folder to store jpgs
idxs = 0
SubPath = os.path.join(PicPath, 'batch_'+str(idxs))
if not os.path.exists(SubPath):
    os.makedirs(SubPath)
for i in range(AllPxs.shape[0]):
    print(i)
    line = AllPxs[i]#.astype(np.uint8)
    im = Image.fromarray(line)
    #im.show()
    im.save(os.path.join(SubPath, str(i) + '.jpg'))
    if np.mod(i, 20000) == 0 and i>0:
        idxs = idxs + 1
        SubPath = os.path.join(PicPath, 'batch_'+str(idxs))
        if not os.path.exists(SubPath):
            os.makedirs(SubPath)
        print('new pack:' + str(idxs))