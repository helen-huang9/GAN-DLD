from preprocess import *
from models.cnn import get_CNN_model


def train_model(model_type, dataset):
    if model_type == 'cnn':
        model = get_CNN_model()

    if dataset == 'all':
        X0, Y0, X1, Y1 = get_all()
    elif dataset == 'indian':
        X0, Y0, X1, Y1 = get_indian()
    elif dataset == 'bengali':
        X0, Y0, X1, Y1 = get_bengali()
    elif dataset == 'hindi':
        X0, Y0, X1, Y1 = get_hindi()
    else:
        X0, Y0, X1, Y1 = get_CEDAR()

    epochs = 3
    batch_size = 250

    print("Starting to train model")
    history = model.fit(
        X0, Y0,
        epochs      = epochs,
        batch_size  = batch_size,
        validation_data = (X1, Y1)
    )

    return model
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--model",    default='cnn',     choices='cnn transformer gan'.split(), help="model to use")
    parser.add_argument("--dataset", default="bengali", choices='cedar bengali hindi indian all'.split(), type=str, help="dataset to train on")
    args = parser.parse_args()

    train_model(args.model, args.dataset)