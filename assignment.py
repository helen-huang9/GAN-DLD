from preprocess import *
from models.cnn import get_siamese_model
from models.transformer import get_transformer_model
from sklearn.metrics import confusion_matrix
import tensorflow as tf 

def train_model(model_type, dataset):
    if dataset == 'all':
        X0, Y0, X1, Y1 = get_all_siamese()
    elif dataset == 'bengali':
        X0, Y0, X1, Y1 = get_indian_siamese('Bengali')
    elif dataset == 'hindi':
        X0, Y0, X1, Y1 = get_indian_siamese('Hindi')
    else:
        X0, Y0, X1, Y1 = get_CEDAR_siamese()

    epochs = 10
    batch_size = 50

    if model_type == 'transformer':
        model = get_transformer_model()
    else:
        model = get_siamese_model() # CNN Model Default
    
    print("Starting to train model")
    history = model.fit(
        [X0[:,0], X0[:,1]], Y0,
        epochs      = epochs,
        batch_size  = batch_size,
        validation_data = ([X1[:,0], X1[:,1]], Y1)
    )

    y_true = Y1
    y_pred = tf.round(model([X1[:,0], X1[:,1]]))
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    return model
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--model",    default='cnn',     choices='cnn transformer gan'.split(), help="model to use")
    parser.add_argument("--dataset", default="cedar", choices='cedar bengali hindi indian all'.split(), type=str, help="dataset to train on")
    args = parser.parse_args()

    train_model(args.model, args.dataset)