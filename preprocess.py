import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
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


def get_CEDAR():
    """
    Loads CEDAR dataset. The dataset needs to be extracted from .rar file first
    Most CEDAR images appear to be about 1:2 for aspect ratio
    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
    """
    genuine_files, forgery_files = get_CEDAR_filepaths()
    data = load_genuine_and_forged(genuine_files, forgery_files, 'CEDAR')
    return shuffle_data(data)

def get_CEDAR_features():
    """
    Loads CEDAR dataset and featurizes data. The dataset needs to be extracted from .rar file first
    Most CEDAR images appear to be about 1:2 for aspect ratio

    For each image, it adds 2 genuine images to try and get the model to compare with. 
    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
    """
    D=[]
    genuine_files, forgery_files = get_CEDAR_filepaths()
    genuine_files.sort()
    forgery_files.sort()
    person_id = None

    print("Loading genuine CEDAR features")
    for file in tqdm(genuine_files):
        file_id = int(file.split("_")[-1].split(".")[0])
        if file_id > 2:
            if person_id != file.split("_")[2]: # Trying not to load the same image over and over
                person_id = file.split("_")[2]
                example1 = "./data/signatures/full_org/original_" + str(person_id)+ "_1.png"
                ex1_image = Image.open(example1).convert('L').resize( (L,W), resample=Image.ANTIALIAS )
                ex1_data = np.asarray(ex1_image).reshape((L,W,1))

                example2 = "./data/signatures/full_org/original_" + str(person_id)+ "_2.png"
                ex2_image = Image.open(example2).convert('L').resize( (L,W), resample=Image.ANTIALIAS )
                ex2_data = np.asarray(ex2_image).reshape((L,W,1))

            image = Image.open(file).convert('L').resize( (L,W), resample=Image.ANTIALIAS )
            im_data = np.asarray(image).reshape((L,W,1))
            # Stack data so the first 2 channels are geunine, the last is unknown
            features = np.dstack((ex1_data,ex2_data,im_data))
            D.append((features, GENUINE_LABEL))

    print("Loading forged CEDAR features")
    for file in tqdm(forgery_files):
        file_id = int(file.split("_")[-1].split(".")[0])
        if file_id > 2:
            if person_id != file.split("_")[2]: # Trying not to load the same image over and over
                person_id = file.split("_")[2]
                example1 = "./data/signatures/full_org/original_" + str(person_id)+ "_1.png"
                ex1_image = Image.open(example1).convert('L').resize( (L,W), resample=Image.ANTIALIAS )
                ex1_data = np.asarray(ex1_image).reshape((L,W,1))

                example2 = "./data/signatures/full_org/original_" + str(person_id)+ "_2.png"
                ex2_image = Image.open(example2).convert('L').resize( (L,W), resample=Image.ANTIALIAS )
                ex2_data = np.asarray(ex2_image).reshape((L,W,1))

            image = Image.open(file).convert('L').resize( (L,W), resample=Image.ANTIALIAS )
            im_data = np.asarray(image).reshape((L,W,1))
            # Stack data so the first 2 channels are geunine, the last is unknown
            features = np.dstack((ex1_data,ex2_data,im_data))
            D.append((features, FORGED_LABEL))
    
    return shuffle_data(D)

def get_bengali():
    """
    Loads Bengali dataset
    """
    genuine, forged = get_indian_paths('Hindi')
    data = load_genuine_and_forged(genuine, forged, 'Hindi')
    return shuffle_data(data)


def get_hindi():
    """
    Load Hindi dataset
    """
    genuine, forged = get_indian_paths('Hindi')
    data = load_genuine_and_forged(genuine, forged, 'Hindi')
    return shuffle_data(data)

def get_indian():
    """
    Gets data for both the Bengali or Hindi dataset    
    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
    """
    D = []
    # Bengali
    genuine_files, forgery_files = get_indian_paths('Bengali')
    D += load_genuine_and_forged(genuine_files, forgery_files, 'Bengali')
    # Hindi
    genuine_files, forgery_files = get_indian_paths('Hindi')
    D += load_genuine_and_forged(genuine_files, forgery_files, 'Hindi')
    return shuffle_data(D)

def get_all():
    """
    Gets data for CEDAR, Bengali, and Hindi datasets
    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
    """
    D = []
    # CEDAR
    genuine_files, forgery_files = get_CEDAR_filepaths()
    D += load_genuine_and_forged(genuine_files, forgery_files, 'CEDAR')
    # Bengali
    genuine_files, forgery_files = get_indian_paths('Bengali')
    D += load_genuine_and_forged(genuine_files, forgery_files, 'Bengali')
    # Hindi
    genuine_files, forgery_files = get_indian_paths('Hindi')
    D += load_genuine_and_forged(genuine_files, forgery_files, 'Hindi')
    return shuffle_data(D)

def load_genuine_and_forged(genuine_files, forged_files, dataset):
    """
    Loads data given lists of genuine and forgery file paths.
    Returns a single array where each element is tuple(image, label).

    :return D [tuple(np.array , List<int>)]: 
    """
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
    """
    Takes a list of data samples, shuffles them and splits them into
    training and test images and labels
    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
    """
    rng = np.random.default_rng()
    rng.shuffle(data)

    split = int(len(data)*0.8)
    train = data[:split]
    test = data[split:]
    X0, Y0 = zip(*train)
    X1, Y1 = zip(*test)

    return np.array(X0), np.array(Y0), np.array(X1), np.array(Y1)


###############################################################################################
## Helper functions to get lists of file paths for datasets

def get_indian_paths(language):
    base_path = './data/BHSig260/' + language + '/'
    gfp = base_path+'list.genuine'
    ffp = base_path+'list.forgery'
    genuine_files = open(gfp, 'r').read().splitlines()
    forged_files = open(ffp, 'r').read().splitlines()
    genuine_files = [base_path+file for file in genuine_files]
    forged_files = [base_path+file for file in forged_files]
    return genuine_files, forged_files

def get_CEDAR_filepaths():
    org_paths = glob.glob('./data/signatures/full_org/*.png')
    forg_paths = glob.glob('./data/signatures/full_forg/*.png')
    return org_paths, forg_paths