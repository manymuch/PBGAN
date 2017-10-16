"""
convert: Create training data from raw Imagenet batches.
"""

import pickle
import glob
import os

import numpy as np
from scipy.misc import imsave


PIXELS_DIR = "Imagenet64"
IMAGE_NUM = 120000
IMAGE_SIZE = 64
CHANNEL_SIZE = IMAGE_SIZE*IMAGE_SIZE

def unpack_file(fname):
    """
        Unpacks a Imagenet file.
    """

    with open(fname, "rb") as f:

        result = pickle.load(f)
    return result


def save_as_image(img_flat, fname):
    """
        Saves a data blob as an image file.
    """

    # consecutive 1024 entries store color channels of 32x32 image 
    img_R = img_flat[0:CHANNEL_SIZE].reshape((IMAGE_SIZE, IMAGE_SIZE))
    img_G = img_flat[CHANNEL_SIZE:CHANNEL_SIZE*2].reshape((IMAGE_SIZE, IMAGE_SIZE))
    img_B = img_flat[CHANNEL_SIZE*2:CHANNEL_SIZE*3].reshape((IMAGE_SIZE, IMAGE_SIZE))
    img = np.dstack((img_R, img_G, img_B))
    imsave(os.path.join(PIXELS_DIR, fname), img)


def main():
    """
        Entry point.
    """

    # use "data_batch_*" for just the training set
    for fname in glob.glob("train_data_batch_10"):
        data = unpack_file(fname)
        for i in range(IMAGE_NUM):
            img_flat = data["data"][i]
            name= "j" +str(i) + ".jpg"
            
            # save the image
            save_as_image(img_flat, name)


            # save the image
        save_as_image(img_flat, fname)
    
if __name__ == "__main__":
    main()
