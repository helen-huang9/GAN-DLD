from preprocess import get_CEDAR
from models.cnn import get_CNN_model


def train_model(model_type, dataset):
    if model_type == 'cnn':
        model = get_CNN_model()
    if dataset == 'cedar':
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
    parser.add_argument("--model",    default='cnn',     choices='cnn transformer gan'.split(), help="task to perform")
    parser.add_argument("--dataset", default="cedar", choices='cedar indian both'.split(), type=str, help="subtask to perform")
    args = parser.parse_args()

    train_model(args.model, args.dataset)