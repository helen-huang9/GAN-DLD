import glob
import numpy as np
from PIL import Image

##  This file contains functions to load data from various datasets
#   Images are shaped to (L,W,1) to be compatible with Conv2D layer.
#   Pixel values range from 0 to 255 so you may need to apply a rescale layer (see CNN)
#   Labels are either [1,0] for genuine or [0,1] for forged

#Size of images to return
L = 64
W = 128

def get_CEDAR():
    """
    Loads CEDAR dataset. The dataset needs to be extracted from .rar file first
    Most CEDAR images appear to be about 1:2 for aspect ratio
    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
    """

    org_paths = glob.glob('./data/signatures/full_org/*.png')
    forg_paths = glob.glob('./data/signatures/full_forg/*.png')

    return load_genuine_and_forged(org_paths, forg_paths, 'CEDAR')

def get_bengali():
    """
    Loads Bengali dataset. The dataset needs to be extracted from .rar file first
    Most Bengali signature images appear to be about 1:3 for aspect ratio
    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
    """
    return get_indian('Bengali')

def get_hindi():
    """
    Loads Hindi dataset. The dataset needs to be extracted from .rar file first
    Most Hindi signature images appear to be about 1:3 for aspect ratio
    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
    """
    return get_indian('Hindi')

def get_indian(type):
    base_path = './data/BHSig260/' + type + '/'
    gfp = base_path+'list.genuine'
    ffp = base_path+'list.forgery'
    genuine_files = open(gfp, 'r').read().splitlines()
    forged_files = open(ffp, 'r').read().splitlines()
    genuine_files = [base_path+file for file in genuine_files]
    forged_files = [base_path+file for file in forged_files]

    return load_genuine_and_forged(genuine_files, forged_files, type)

def load_genuine_and_forged(genuine_files, forged_files, dataset):
    D = []
    print("Loading genuine " + dataset + " files")
    for file in genuine_files:
        image = Image.open(file).convert('L')
        image = image.resize( (L,W), resample=Image.ANTIALIAS )
        im_data = np.asarray(image).reshape((L,W,1))
        D.append((im_data, [1,0]))

    print("Loading forged " + dataset + " files")
    for file in forged_files:
        image = Image.open(file).convert('L')
        image = image.resize( (L,W), resample=Image.ANTIALIAS )
        im_data = np.asarray(image).reshape((L,W,1))
        D.append((im_data, [0,1]))

    # Shuffle dataset
    rng = np.random.default_rng()
    rng.shuffle(D)

    split = int(len(D)*0.8)
    train = D[:split]
    test = D[split:]
    X0, Y0 = zip(*train)
    X1, Y1 = zip(*test)

    return np.array(X0), np.array(Y0), np.array(X1), np.array(Y1)