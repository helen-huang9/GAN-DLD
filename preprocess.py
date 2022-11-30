import glob
import numpy as np
from PIL import Image

def get_CEDAR():
    """
    Loads CEDAR dataset. The dataset needs to be extracted from .rar file first

    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
    """
    org_paths = './data/signatures/full_org/*.png'
    forg_paths = './data/signatures/full_forg/*.png'

    D = []
    # Load original signatures
    print("Loading CEDAR original images")
    for im_path in glob.glob(org_paths):
        image = Image.open(im_path)
        im_data = np.asarray(image)
        D.append((im_data, 0))

    # Load forged signatures
    print("Loading CEDAR forged signatures")
    for im_path in glob.glob(forg_paths):
        image = Image.open(im_path)
        im_data = np.asarray(image)
        D.append((im_data, 1))
    
    # Shuffle dataset
    rng = np.random.default_rng()
    rng.shuffle(D)

    split = int(len(D)*0.8)
    train = D[:split]
    test = D[split:]
    X0, Y0 = zip(*train)
    X1, Y1 = zip(*test)
    return X0, Y0, X1, Y1
