import glob
from tqdm import tqdm
import numpy as np
import os
from preprocess import get_siamese_data
from PIL import Image
L = 64
W = 128

def get_stu():
    base_path = './data/diy/'
    genuine_paths = [base_path+'stu_stu1.jpg', base_path+'stu_stu2.jpg']
    forged_paths = [base_path+'josh_stu.jpg']

    pairs, labels = get_siamese_data(genuine_paths, forged_paths)
    X1, Y1 = np.array(pairs), np.array(labels)
    return X1, Y1