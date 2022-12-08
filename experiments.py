"""
Experiments for report:

For the CNN and Transformer:

1. Want to verify the model learns an individual's signature, especially with a smaller data set. 
    - TODO: Transformer with cross attention
    - Train and save models on each individual dataset and 'all', record accuracy, recall, precision
    - Train models on each dataset while holding out some individuals signatures 
    - Compare test accuracies, expecting that the model doesn't accurately correct new signatures

2. Can we train a model to learn a new signature?
    - Take the models from above that were trained with users held out
    - Train on held out users data, observe model convergence
    - Which trains faster?

3. If the above can be accomplished, can the model learn with no forged examples? (Just label 0 pairs)
    - Probably not
4. Can we induce bias to ensure forgeries don't get past?

NOTE: All experiments run with random SEED=42
TODO: How would a system like this actually be used?
 - in a real setting, probably want a general purpose forgery detector
 - this model probably just learns a few individual's signatures and learns to detect them
NOTE: Transformer trained with al in about 12 epochs for sure 15

"""
from models.transformer import get_siamese_transformer, get_basic_transformer
from models.cnn import get_siamese_cnn, get_basic_cnn
from preprocess import *
from experiment_helpers import *
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
def experiment_one(task):
    """
    Trains and evaluates each model:
        - as a siamese netowrk (task 1)
        - as a binary classifier (task 2)
        - and evaluates them (task 3)
    """
    for dataset in ['cedar', 'bengali', 'hindi', 'all']:
    # for dataset in ['hindi', 'all']:
        if task in ['1', 'all']:
            # X0, Y0, X1, Y1 = get_filtered_dataset(dataset)
            X0, Y0, X1, Y1 = get_siamese_dataset(dataset)
            split = int(len(Y0) * 0.8)
            X_train, Y_train = X0[:split], Y0[:split]
            X_train = [X_train[:,0], X_train[:,1]]
            cnn = get_siamese_cnn()
            transformer = get_siamese_transformer()
            print("TRAINING SIAMESE CNN")
            if dataset != 'hindi':
                not_a_gan_train(cnn, X_train, Y_train, dataset+'_siamese',True)
            print("TRAINING SIAMESE TRANSFORMER")
            not_a_gan_train(transformer, X_train, Y_train, dataset+'_siamese',False)
        if task in ['2', 'all']:
            X0, Y0, X1, Y1 = get_singles_dataset(dataset)
            split = int(len(Y0) * 0.8)
            X_train, Y_train = X0[:split], Y0[:split]
            cnn = get_basic_cnn()
            transformer = get_basic_transformer()
            print("TRAINING SINGLES CNN")
            not_a_gan_train(cnn, X_train, Y_train, dataset + '_single', isCnn=True)
            print("TRAINING SINGLES TRANSFORMER")
            not_a_gan_train(transformer, X_train, Y_train, dataset + '_single', False)

def test_model(X1, Y1, model, dataset, isCnn=True, isUnseen=False):
    name = 'CNN' if isCnn else 'Transformer'
    seen = 'unseen' if isUnseen else 'seen'
    y_pred = tf.round(model.predict([X1[:,0], X1[:,1]]))
    y_pred = tf.reshape(y_pred, (-1,1))
    confusion = confusion_matrix(Y1, y_pred)
    tn = confusion[0,0]
    tp = confusion[1,1]
    fn = confusion[1,0]
    fp = confusion[0,1]
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    f = open('experiment1.txt', 'a') 
    f.write("Model: " + name +  "  Dataset: " + dataset.upper() + "  Signatures: " + seen + "\n")
    print("Model: ", name, "  Dataset: ", dataset.upper(), " Signatures: ", seen)
    f.write('Accuracy: ' + str(accuracy) + '\n')
    f.write('Precision: ' + str(precision) + '\n')
    f.write('Recall: ' + str(recall) + '\n')
    print('Accuracy: ', str(accuracy))
    f.write('Confusion Matrix: \n')
    f.write(str(confusion))
    f.write("\n************************************************************** \n")
    f.close()
    print('Confusion Matrix: \n')
    print(str(confusion))

def not_a_gan_train(model, X_train, Y_train, dataset, isCnn):
    name = 'cnn' if isCnn else 'transformer'
    epochs = 20
    batch_size = 50
    model.fit(
            X_train, Y_train,
            epochs      = epochs,
            batch_size  = batch_size
            # validation_data = (X_train[:50], Y_train[:50])
        )
    model.save_weights('./'+name+'/'+dataset)

def experiment_two(task):
    """
    Proves we can get the model to train to recognize 3 signatures perfectly
    Model pre-trained on other languages trains faster
    Use task 1 for CNN, 2 for transformer

    The CNN seems to learn faster than the Transformer for this task
    """
    if task == '1':
        partial, full = get_siamese_cnn(), get_siamese_cnn()
        partial.load_weights('./cnn/filtered_cedar')
        full.load_weights('./cnn/filtered_all')
    elif task == '2':
        partial, full = get_siamese_transformer(), get_siamese_transformer()
        partial.load_weights('./transformer/filtered_cedar')
        full.load_weights('./transformer/filtered_all')

    _, _, x1, y1 = get_siamese_dataset('cedar')
    split = int(len(y1) * 0.8)
    X_train, Y_train = x1[:split], y1[:split]
    X_val, Y_val = x1[split:], y1[split:]
    print("Training CEDAR model")
    partial.fit(
        [X_train[:,0], X_train[:,1]], Y_train,
        epochs      = 30,
        batch_size  = 20,
        validation_data = ([X_val[:,0], X_val[:,1]], Y_val)
    )
    print("Training model trained on all")
    full.fit(
        [X_train[:,0], X_train[:,1]], Y_train,
        epochs      = 10,
        batch_size  = 20,
        validation_data = ([X_val[:,0], X_val[:,1]], Y_val)
    )
def experiment_three(task):
    """
    Want to test if we can train a model by only providing genuine signatures
    """
    if task == '1':
        model = get_siamese_transformer()
        model.load_weights('./cnn/filtered_all')
    if task == '2':
        model = get_siamese_transformer()
        model.load_weights('./transformer/filtered_all')
    # Get data
    x0, y0, x1, y1 = get_siamese_dataset('cedar')
    # Split into train and test
    split = int(len(y1) * 0.8)
    X_train, Y_train = x1[:split], y1[:split]
    X_val, Y_val = x1[split:], y1[split:]
    # Remove 1's from training data
    idx = [np.where(Y_train==i)[0] for i in range(2)]
    X_train = np.take(X_train, idx[0],axis=0)
    Y_train = np.take(y1, idx[0])
    print("Adding ", str(len(Y_train)), " signatures")
    model.fit(
        [X_train[:,0], X_train[:,1]], [Y_train],
        epochs      = 30,
        batch_size  = 20,
        validation_data = ([X_val[:,0], X_val[:,1]], [Y_val])
    )
    y_pred = tf.round(model.predict([X_val[:,0], X_val[:,1]]))
    y_pred = tf.reshape(y_pred, (-1,1))
    confusion = confusion_matrix(Y_val, y_pred)
    print(confusion)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--exp",    default='1',     choices='1 2 3'.split(), help="Experiment to run")
    parser.add_argument("--task",    default='all',     choices='1 2 3 all'.split(), help="Experiment to run")
    args = parser.parse_args()

    if args.exp == '1':
        exp = experiment_one
    elif args.exp == '2':
        assert(args.task in ['1','2'])
        exp = experiment_two
    elif args.exp == '3':
        assert(args.task in ['1','2'])
        exp = experiment_three

    exp(args.task)