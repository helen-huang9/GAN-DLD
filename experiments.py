"""
Experiments for report:

For the CNN and Transformer:

1. Want to verify the model learns an individual's signature, especially with a smaller data set. 
    - TODO: Transformer with cross attention
    - Train and save models on each individual dataset and 'all', record accuracy, recall, precision
    - Train models on each dataset while holding out some individuals signatures 
    - Compare test accuracies, expecting that the model doesn't accurately correct new signatures

2. Can we train a model to learn a new signature?
TODO: Lookup best practice for how to train with new samples in this case. Just do some epochs with new data?, Need to think about how this system would be used
    - Take the models from above that were trained with users held out, train with one pair at a time 
    - See how just training with new data effects old accuracy

3. Which model learns quicker? (English added to english model or english added to all model)
4. If the above can be accomplished, can the model learn with no forged examples? (Just label 0 pairs)
5. Can we induce bias to ensure forgeries don't get past?

NOTE: All experiments run with random SEED=42
TODO: How would a system like this actually be used?
 - in a real setting, probably want a general purpose forgery detector
 - this model probably just learns a few individual's signatures and learns to detect them
NOTE: Transformer trained with al in about 12 epochs for sure 15

"""
from models.transformer import get_transformer_model
from models.cnn import get_siamese_model
from preprocess import *
from experiment_helpers import *
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
def experiment_one(task):
    if task in ['1', 'all']:
        print("*************TASK 1*****************")
        for dataset in ['cedar', 'bengali', 'hindi', 'all']:
            X0, Y0, _, _ = get_full_dataset(dataset)
            train_models(X0, Y0, dataset, filtered=False)
    if task in ['2', 'all']:
        print("*************TASK 2*****************")
        # for dataset in ['cedar', 'bengali', 'hindi', 'all']:
        for dataset in ['all']:
            X0, Y0, _, _ = get_filtered_dataset(dataset)
            train_models(X0, Y0, dataset, filtered=True)
    if task in ['3', 'all']:
        print('*************TASK 3*****************')
        for dataset in ['cedar', 'bengali', 'hindi', 'all']:
            f = open('experiment1.txt', 'a')
            f.write("Accuracy on models with seen signatures: \n".upper())
            f.close()
            # Test models trained on full dataset
            _, _, X1, Y1 = get_full_dataset(dataset)
            cnn = get_siamese_model()
            cnn.load_weights('./cnn/full_'+dataset)

            transformer = get_transformer_model()
            transformer.load_weights('./transformer/full_'+dataset)

            test_model(X1, Y1, cnn, dataset)
            test_model(X1, Y1, transformer, dataset, False)
            # Test models on held out users
            _, _, X1, Y1 = get_filtered_dataset(dataset)
            cnn = get_siamese_model()
            cnn.load_weights('./cnn/filtered_'+dataset)

            transformer = get_transformer_model()
            transformer.load_weights('./transformer/filtered_'+dataset)
            test_model(X1, Y1, cnn, dataset)
            test_model(X1, Y1, transformer, dataset, False)

def test_model(X1, Y1, model, dataset, isCnn=True):
    name = 'CNN' if isCnn else 'Transformer'
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
    f.write("Model: " + name +  "  Dataset: " + dataset.upper() + '\n')
    print("Model: ", name, "  Dataset: ", dataset.upper())
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

def train_models(X0,Y0,dataset, filtered=False):
    name = 'filtered' if filtered else 'full'
    epochs = 20
    batch_size = 50
    cnn = get_siamese_model()
    history = cnn.fit(
            [X0[:,0], X0[:,1]], Y0,
            epochs      = epochs,
            batch_size  = batch_size,
            validation_data = ([X0[:50,0], X0[:50,1]], Y0[:50])
        )
    cnn.save_weights('./cnn/'+name+'_'+dataset)

    transformer = get_transformer_model()
    history = cnn.fit(
            [X0[:,0], X0[:,1]], Y0,
            epochs      = epochs,
            batch_size  = batch_size,
            validation_data = ([X0[:50,0], X0[:50,1]], Y0[:50])
        )
    transformer.save_weights('./transformer/'+name+'_'+dataset)


def experiment_two(task):
    """
    Proves we can get the model to train to recognize 3 signatures perfectly
    Model pre-trained on other languages trains faster
    Use task 1 for CNN, 2 for transformer

    The CNN seems to learn faster than the Transformer for this task
    """
    if task == '1':
        partial, full = get_siamese_model(), get_siamese_model()
        partial.load_weights('./cnn/filtered_cedar')
        full.load_weights('./cnn/filtered_all')
    elif task == '2':
        partial, full = get_transformer_model(), get_transformer_model()
        partial.load_weights('./transformer/filtered_cedar')
        full.load_weights('./transformer/filtered_all')

    _, _, x1, y1 = get_filtered_dataset('cedar')
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--exp",    default='1',     choices='1 2'.split(), help="Experiment to run")
    parser.add_argument("--task",    default='all',     choices='1 2 3 all'.split(), help="Experiment to run")
    args = parser.parse_args()

    if args.exp == '1':
        exp = experiment_one
    elif args.exp == '2':
        assert(args.task in ['1','2'])
        exp = experiment_two

    exp(args.task)