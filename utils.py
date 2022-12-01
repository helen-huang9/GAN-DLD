import numpy as np
from PIL import Image
import tensorflow.keras.backend as K

def make_siamese_pairs(images, labels):
    pair_images = []
    pair_labels = []
    # 2 element long list where each entry is all indices for that label
    idx = [np.where(labels==i)[0] for i in range(2)]
    for idxA in range(len(images)):
        current_image = images[idxA]
        label = labels[idxA]

        # randomly pick an image that belong to the same class
        pos_idx = np.random.choice(idx[label])
        positive_image = images[pos_idx]
        pair_images.append([current_image, positive_image])
        pair_labels.append([1])

        # Randomly pick an image that belongs to the other class
        neg_idx = np.where(labels != label)[0]
        negative_image = images[np.random.choice(neg_idx)]
        pair_images.append([current_image, negative_image])
        pair_labels.append([0])
    return np.array(pair_images), np.array(pair_labels)

def euclidean_distance(vectors):
    # unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))
