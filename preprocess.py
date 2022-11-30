import glob
import numpy as np
from PIL import Image

##  This file contains functions to load data from various datasets
#   Images are shaped to (L,W,1) to be compatible with Conv2D layer.
#   Pixel values range from 0 to 255 so you may need to apply a rescale layer (see CNN)
#   Labels are either [1,0] for genuine or [0,1] for forged

# Size of images to return
L = 64
W = 128

# Labels
GENUINE_LABEL = [1,0]
FORGED_LABEL = [0,1]

def get_CEDAR(mixing=False):
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
    data = load_genuine_and_forged(org_paths, forg_paths, 'CEDAR')
    return shuffle_data(data)

def get_bengali():
    """
    See get_indian
    """
    return get_indian('Bengali')

def get_hindi():
    """
    See get_indian
    """
    return get_indian('Hindi')

def get_indian(language):
    """
    Gets data for either the Bengali or Hindi dataset depending on language
    Uses helper files list.genuine and list.forgery to get list of file 
    names in each class
    
    :param language: either 'Bengali' or 'Hindi'
    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
    """
    base_path = './data/BHSig260/' + language + '/'
    gfp = base_path+'list.genuine'
    ffp = base_path+'list.forgery'
    genuine_files = open(gfp, 'r').read().splitlines()
    forged_files = open(ffp, 'r').read().splitlines()
    genuine_files = [base_path+file for file in genuine_files]
    forged_files = [base_path+file for file in forged_files]

    data = load_genuine_and_forged(genuine_files, forged_files, language)
    return shuffle_data(data)

def get_all():
    D = []
    # CEDAR
    org_paths = glob.glob('./data/signatures/full_org/*.png')
    forg_paths = glob.glob('./data/signatures/full_forg/*.png')
    D += load_genuine_and_forged(org_paths, forg_paths, 'CEDAR')
    # Bengali
    base_path = './data/BHSig260/Bengali/'
    gfp = base_path+'list.genuine'
    ffp = base_path+'list.forgery'
    genuine_files = open(gfp, 'r').read().splitlines()
    forged_files = open(ffp, 'r').read().splitlines()
    genuine_files = [base_path+file for file in genuine_files]
    forged_files = [base_path+file for file in forged_files]
    D += load_genuine_and_forged(genuine_files, forged_files, 'Bengali')
    # Hindi
    base_path = './data/BHSig260/Hindi/'
    gfp = base_path+'list.genuine'
    ffp = base_path+'list.forgery'
    genuine_files = open(gfp, 'r').read().splitlines()
    forged_files = open(ffp, 'r').read().splitlines()
    genuine_files = [base_path+file for file in genuine_files]
    forged_files = [base_path+file for file in forged_files]
    D += load_genuine_and_forged(genuine_files, forged_files, 'Hindi')
    return shuffle_data(D)

def load_genuine_and_forged(genuine_files, forged_files, dataset):
    D = []
    print("Loading genuine " + dataset + " files")
    for file in genuine_files:
        image = Image.open(file).convert('L')
        image = image.resize( (L,W), resample=Image.ANTIALIAS )
        im_data = np.asarray(image).reshape((L,W,1))
        D.append((im_data, GENUINE_LABEL))

    print("Loading forged " + dataset + " files")
    for file in forged_files:
        image = Image.open(file).convert('L')
        image = image.resize( (L,W), resample=Image.ANTIALIAS )
        im_data = np.asarray(image).reshape((L,W,1))
        D.append((im_data, FORGED_LABEL))

    return D

def shuffle_data(data):
    rng = np.random.default_rng()
    rng.shuffle(data)

    split = int(len(data)*0.8)
    train = data[:split]
    test = data[split:]
    X0, Y0 = zip(*train)
    X1, Y1 = zip(*test)

    return np.array(X0), np.array(Y0), np.array(X1), np.array(Y1)