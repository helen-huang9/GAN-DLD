import glob
from tqdm import tqdm
import numpy as np
import os
from preprocess import get_siamese_data
from PIL import Image
L = 64
W = 128
SEED = 42

def get_siamese_dataset(dataset):
    if dataset == 'all':
        X0, Y0, X1, Y1 = get_all_siamese()
    elif dataset == 'bengali':
        X0, Y0, X1, Y1 = get_indian_siamese('Bengali')
    elif dataset == 'hindi':
        X0, Y0, X1, Y1 = get_indian_siamese('Hindi')
    else:
        X0, Y0, X1, Y1 = get_CEDAR_siamese()
    return X0, Y0, X1, Y1

def get_CEDAR_siamese():
    # 55 users in dataset - first 11 will be test data
    people_ids = [i for i in range(12,56)]
    all_pairs, all_labels = [],[]
    print("Loading CEDAR siamese pairs...")
    for id in tqdm(people_ids):
        genuine_paths = glob.glob('./data/signatures/full_org/original_' + str(id) + '_*.png')
        forged_paths = glob.glob('./data/signatures/full_forg/forgeries_' + str(id) + '_*.png')
        pairs, labels = get_siamese_data(genuine_paths, forged_paths)
        all_pairs += pairs
        all_labels += labels

    temp = list(zip(all_pairs, all_labels))
    X0, Y0 = shuffle_data(temp)

    print("Loading Test CEDAR siamese pairs...")
    people_ids = [i for i in range(1,12)]
    all_pairs, all_labels = [],[]
    for id in tqdm(people_ids):
        genuine_paths = glob.glob('./data/signatures/full_org/original_' + str(id) + '_*.png')
        forged_paths = glob.glob('./data/signatures/full_forg/forgeries_' + str(id) + '_*.png')
        pairs, labels = get_siamese_data(genuine_paths, forged_paths)
        all_pairs += pairs
        all_labels += labels

    X1, Y1 = np.array(all_pairs), np.array(all_labels)
    return X0, Y0, X1, Y1


def get_indian_siamese(language):
    base_path = './data/BHSig260/'+language.capitalize()
    people_ids = [d for d in os.listdir(base_path) if d.isdigit()]
    people_ids.sort()
    # 100 users in Bengali dataset - 20 test
    # 160 users in Hindi dataset - 32
    split = 20 if language.capitalize() == 'Bengali' else 32
    test_ids = people_ids[:split]
    train_ids = people_ids[split:]
    all_pairs, all_labels = [],[]
    print("Loading " + language + " siamese pairs...")
    for id in tqdm(train_ids):
        person_files = glob.glob(base_path + '/' + id + '/*.tif')
        genuine_paths = [path for path in person_files if 'G' in path]
        forged_paths = [path for path in person_files if 'F' in path]
        pairs, labels = get_siamese_data(genuine_paths, forged_paths)
        all_pairs += pairs
        all_labels += labels

    temp = list(zip(all_pairs, all_labels))
    X0, Y0 = shuffle_data(temp)

    print("Loading test" + language + " siamese pairs...")
    all_pairs, all_labels = [],[]
    for id in tqdm(test_ids):
        person_files = glob.glob(base_path + '/' + id + '/*.tif')
        genuine_paths = [path for path in person_files if 'G' in path]
        forged_paths = [path for path in person_files if 'F' in path]
        pairs, labels = get_siamese_data(genuine_paths, forged_paths)
        all_pairs += pairs
        all_labels += labels
    X1, Y1 = np.array(all_pairs), np.array(all_labels)
    return X0, Y0, X1, Y1

def get_all_siamese():
    X0, Y0, X1, Y1 = get_CEDAR_siamese()
    for lang in ['Bengali', 'Hindi']:
        a0, b0, a1, b1 = get_indian_siamese(lang)
        X0 = np.append(X0, a0, axis=0)
        Y0 = np.append(Y0, b0, axis=0)
        X1 = np.append(X1, a1, axis=0)
        Y1 = np.append(Y1, b1, axis=0)
    # Shuffle data
    rng = np.random.default_rng(seed=SEED)
    p0 = rng.permutation(len(X0))
    p1 = rng.permutation(len(X1))
    print("Permutating train ims")
    X0 = X0[p0]
    print("Permutating train labels")
    Y0 = Y0[p0]
    print("Permutating test ims")
    X1 = X1[p1]
    print("Permutating test labels")
    Y1 = Y1[p1]
    return X0, Y0, X1, Y1

def shuffle_data(data):
    """
    Takes a list of data samples, shuffles them and splits them into
    training and test images and labels
    :return X: Input Pairs,
            Y: Labels,
    """
    rng = np.random.default_rng(seed=SEED)
    rng.shuffle(data)
    X, Y = zip(*data)

    return np.array(X), np.array(Y)


def get_singles_dataset(dataset):
    if dataset == 'all':
        X0, Y0, X1, Y1 = get_all_singles()
    elif dataset == 'bengali':
        X0, Y0, X1, Y1 = get_indian_singles('Bengali')
    elif dataset == 'hindi':
        X0, Y0, X1, Y1 = get_indian_singles('Hindi')
    else:
        X0, Y0, X1, Y1 = get_CEDAR_singles()
    return X0, Y0, X1, Y1
def get_CEDAR_singles():
    people_ids = [i for i in range(12,56)]
    all_ims, all_labels = [],[]
    print("Loading CEDAR siamese pairs...")
    for id in tqdm(people_ids):
        genuine_paths = glob.glob('./data/signatures/full_org/original_' + str(id) + '_*.png')
        forged_paths = glob.glob('./data/signatures/full_forg/forgeries_' + str(id) + '_*.png')
        ims, labels = get_single_data(genuine_paths, forged_paths)
        all_ims += ims
        all_labels += labels

    temp = list(zip(all_ims, all_labels))
    X0, Y0 = shuffle_data(temp)

    print("Loading Test CEDAR siamese pairs...")
    people_ids = [i for i in range(1,4)]
    all_ims, all_labels = [],[]
    for id in tqdm(people_ids):
        genuine_paths = glob.glob('./data/signatures/full_org/original_' + str(id) + '_*.png')
        forged_paths = glob.glob('./data/signatures/full_forg/forgeries_' + str(id) + '_*.png')
        ims, labels = get_single_data(genuine_paths, forged_paths)
        all_ims += ims
        all_labels += labels

    X1, Y1 = np.array(all_ims), np.array(all_labels)
    return X0, Y0, X1, Y1

def get_indian_singles(language):
    base_path = './data/BHSig260/'+language.capitalize()
    people_ids = [d for d in os.listdir(base_path) if d.isdigit()]
    people_ids.sort()
    # 100 users in Bengali dataset - 20 test
    # 160 users in Hindi dataset - 32
    split = 20 if language.capitalize() == 'Bengali' else 32
    test_ids = people_ids[:split]
    train_ids = people_ids[split:]
    all_pairs, all_labels = [],[]
    print("Loading " + language + " siamese pairs...")
    for id in tqdm(train_ids):
        person_files = glob.glob(base_path + '/' + id + '/*.tif')
        genuine_paths = [path for path in person_files if 'G' in path]
        forged_paths = [path for path in person_files if 'F' in path]
        pairs, labels = get_single_data(genuine_paths, forged_paths)
        all_pairs += pairs
        all_labels += labels

    temp = list(zip(all_pairs, all_labels))
    X0, Y0 = shuffle_data(temp)

    print("Loading test" + language + " siamese pairs...")
    all_pairs, all_labels = [],[]
    for id in tqdm(test_ids):
        person_files = glob.glob(base_path + '/' + id + '/*.tif')
        genuine_paths = [path for path in person_files if 'G' in path]
        forged_paths = [path for path in person_files if 'F' in path]
        pairs, labels = get_single_data(genuine_paths, forged_paths)
        all_pairs += pairs
        all_labels += labels
    X1, Y1 = np.array(all_pairs), np.array(all_labels)
    return X0, Y0, X1, Y1

def get_all_singles():
    X0, Y0, X1, Y1 = get_CEDAR_singles()
    for lang in ['Bengali', 'Hindi']:
        a0, b0, a1, b1 = get_indian_singles(lang)
        X0 = np.append(X0, a0, axis=0)
        Y0 = np.append(Y0, b0, axis=0)
        X1 = np.append(X1, a1, axis=0)
        Y1 = np.append(Y1, b1, axis=0)
    # Shuffle data
    rng = np.random.default_rng(seed=SEED)
    p0 = rng.permutation(len(X0))
    p1 = rng.permutation(len(X1))
    print("Permutating train ims")
    X0 = X0[p0]
    print("Permutating train labels")
    Y0 = Y0[p0]
    print("Permutating test ims")
    X1 = X1[p1]
    print("Permutating test labels")
    Y1 = Y1[p1]
    return X0, Y0, X1, Y1



def get_single_data(genuine_paths, forged_paths):
    person_images = []
    person_labels = []
    for file in genuine_paths:
        image = Image.open(file).convert('L')
        image = image.resize( (L,W), resample=Image.ANTIALIAS )
        im_data = np.asarray(image).reshape((L,W,1))
        person_images.append(im_data)
        person_labels.append(0)

    for file in forged_paths:
        image = Image.open(file).convert('L')
        image = image.resize( (L,W), resample=Image.ANTIALIAS )
        im_data = np.asarray(image).reshape((L,W,1))
        person_images.append(im_data)
        person_labels.append(1)

    return person_images, person_labels

import tensorflow as tf
from sklearn.metrics import confusion_matrix
def evaluate(model, x, label):
    y_pred = tf.round(model.predict(x))
    y_pred = tf.reshape(y_pred, (-1,1))
    confusion = confusion_matrix(label, y_pred)
    tn = confusion[0,0]
    tp = confusion[1,1]
    fn = confusion[1,0]
    fp = confusion[0,1]
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print(confusion)
    # return accuracy, precision, recall, confusion
    