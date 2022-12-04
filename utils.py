import numpy as np
from PIL import Image
import tensorflow as tf
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
        if label != 1:
            pos_idx = np.random.choice(idx[label])
            positive_image = images[pos_idx]
            pair_images.append([current_image, positive_image])
            pair_labels.append([1])

        # Randomly pick an image that belongs to the other class
        if np.random.rand() >= 0.5:
            neg_idx = np.where(labels != label)[0]
            negative_image = images[np.random.choice(neg_idx)]
            pair_images.append([current_image, negative_image])
            pair_labels.append([0])
    return pair_images, pair_labels

def euclidean_distance(vectors):
    # unpack the vectors into separate lists
	a, b = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(a - b), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def contrastive_loss(y, preds, margin=1):
	# explicitly cast the true class label data type to the predicted
	# class label data type (otherwise we run the risk of having two
	# separate data types, causing TensorFlow to error out)
	y = tf.cast(y, preds.dtype)
	# calculate the contrastive loss between the true labels and
	# the predicted labels
	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))
	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	# return the computed contrastive loss to the calling function
	return loss