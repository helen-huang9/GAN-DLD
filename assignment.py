from preprocess import *
from models.cnn import get_siamese_model
from models.transformer import get_transformer_model
from sklearn.metrics import confusion_matrix
import tensorflow as tf


def train_model(model_type, dataset, task):
    if dataset == 'all':
        X0, Y0, X1, Y1 = get_all_siamese()
    elif dataset == 'bengali':
        X0, Y0, X1, Y1 = get_indian_siamese('Bengali')
    elif dataset == 'hindi':
        X0, Y0, X1, Y1 = get_indian_siamese('Hindi')
    else:
        X0, Y0, X1, Y1 = get_CEDAR_siamese()

    X_test, Y_test = X1[:len(X1)//2], Y1[:len(Y1)//2]
    X_val, Y_val = X1[len(X1)//2:], Y1[len(Y1)//2:]
    #############################################################
    ####### Model training and saving
    epochs = 20
    batch_size = 50
    if task in ['train', 'both']:
        if model_type == 'transformer':
            model = get_transformer_model()
            model.summary()
        else:
            model = get_siamese_model()  # CNN Model Default

        print("Starting to train model")
        history = model.fit(
            [X0[:,0], X0[:,1]], Y0,
            epochs      = epochs,
            batch_size  = batch_size,
            validation_data = ([X_val[:,0], X_val[:,1]], Y_val)
        )
        model.save_weights('./transformer/weights')
    #############################################################
    ### Model testing
    if task in ['test', 'both']:
        if task == 'test':
            model.load_weights('./'+model_type+'/weights')
        y_true = Y_test
        y = model([X_test[:, 0], X_test[:, 1]])
        y_pred = tf.round(y)
        y_pred = tf.reshape(y_pred,(-1,1))
        confusion = confusion_matrix(y_true, y_pred)
        #           |  pred0 | pred1 |
        # Actually 0|________|_______|
        # Actually 1|________|_______|
        print("Model Predictions:")
        print(y)
        print("Confusion matrix:")
        print(confusion)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--model",    default='cnn',     choices='cnn transformer gan'.split(), help="model to use")
    parser.add_argument("--dataset", default="cedar", choices='cedar bengali hindi indian all'.split(), type=str, help="dataset to train on")
    parser.add_argument("--task", default="both", choices='train test both'.split(), type=str, help="dataset to train on")
    args = parser.parse_args()

    train_model(args.model, args.dataset, args.task)