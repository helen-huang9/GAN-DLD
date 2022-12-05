"""
Experiments for report:

For the CNN and Transformer:

1. Want to verify the model learns an individual's signature, especially with a smaller data set. 
    - TODO: Transformer with cross attention
    - Train and save models on each individual dataset and 'all', record accuracy, recall, precision
    - Train models on each dataset while holding out some individuals signatures 
    - Compare test accuracies, expecting that the model doesn't accurately correct new signatures

2. If the above is true, we want to know how long it takes for the model to learn a new signature?
TODO: Lookup best practice for how to train with new samples in this case. Just do some epochs with new data?, Need to think about how this system would be used
    - Take the models from above that were trained with users held out, train with one pair at a time 
    - See how just training with new data effects old accuracy

3. Which model learns quicker? (English added to english model or english added to all model)
        

4. If the above can be accomplished, can the model learn with no forged examples? (Just label 0 pairs)
5. Can we induce bias to ensure forgeries don't get past?
NOTE: All experiments run with random SEED=42
TODO: How would a system like this actually be used?
NOTE: Transformer trained with al in about 12 epochs for sure 15

"""
from models.transformer import get_transformer_model
from models.cnn import get_siamese_model
from preprocess import *
from experiment_helpers import *
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--exp",    default='1',     choices='1 2 3 4'.split(), help="Experiment to run")
    parser.add_argument("--task",    default='all',     choices='1 2 3 all'.split(), help="Experiment to run")
    args = parser.parse_args()

    if args.exp == '1':
        exp = experiment_one

    exp(args.task)