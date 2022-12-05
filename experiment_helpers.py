import glob
from tqdm import tqdm
from preprocess import get_siamese_data, get_indian_siamese, get_CEDAR_siamese, get_all_siamese
import numpy as np
import os
SEED = 42

def get_full_dataset(dataset):
    if dataset == 'all':
        X0, Y0, X1, Y1 = get_all_siamese()
    elif dataset == 'bengali':
        X0, Y0, X1, Y1 = get_indian_siamese('Bengali')
    elif dataset == 'hindi':
        X0, Y0, X1, Y1 = get_indian_siamese('Hindi')
    else:
        X0, Y0, X1, Y1 = get_CEDAR_siamese()
    return X0, Y0, X1, Y1

def get_filtered_dataset(dataset):
    if dataset == 'all':
        X0, Y0, X1, Y1 = get_all_filtered()
    elif dataset == 'bengali':
        X0, Y0, X1, Y1 = get_indian_filtered('Bengali')
    elif dataset == 'hindi':
        X0, Y0, X1, Y1 = get_indian_filtered('Hindi')
    else:
        X0, Y0, X1, Y1 = get_CEDAR_filtered()
    return X0, Y0, X1, Y1

def get_CEDAR_filtered():
    people_ids = [i for i in range(4,56)]
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
    people_ids = [i for i in range(1,4)]
    all_pairs, all_labels = [],[]
    for id in tqdm(people_ids):
        genuine_paths = glob.glob('./data/signatures/full_org/original_' + str(id) + '_*.png')
        forged_paths = glob.glob('./data/signatures/full_forg/forgeries_' + str(id) + '_*.png')
        pairs, labels = get_siamese_data(genuine_paths, forged_paths)
        all_pairs += pairs
        all_labels += labels

    X1, Y1 = np.array(all_pairs), np.array(all_labels)
    return X0, Y0, X1, Y1


def get_indian_filtered(language):
    base_path = './data/BHSig260/'+language.capitalize()
    people_ids = [d for d in os.listdir(base_path) if d.isdigit()]
    people_ids.sort()
    test_ids = people_ids[:3]
    train_ids = people_ids[3:]
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

def get_all_filtered():
    X0, Y0, X1, Y1 = get_CEDAR_filtered()
    for lang in ['Bengali', 'Hindi']:
        a0, b0, a1, b1 = get_indian_filtered(lang)
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