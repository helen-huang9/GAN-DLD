import pandas as pd
import pylab as plt

##########################################################################################
############################# Accuracy######################################################
path = './data/siamese_transformer.csv'
df = pd.read_csv(path)
df = df.drop(columns=['Wall time', 'Step'])
df = df.rename(columns={'Value': 'Siamese Transformer'})

path = './data/siamese_cnn.csv'
d = pd.read_csv(path)
df['Siamese CNN'] = d['Value']

path = './data/single_cnn.csv'
d = pd.read_csv(path)
df['Basic CNN'] = d['Value']

path = './data/single_transformer.csv'
d = pd.read_csv(path)
df['Single Transformer'] = d['Value']

df.plot()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.show()
##########################################################################################
############################# Loss ######################################################

path = './data/transformer_siamese_loss.csv'
df = pd.read_csv(path)
df = df.drop(columns=['Wall time', 'Step'])
df = df.rename(columns={'Value': 'Siamese Transformer'})


path = './data/cnn_siamese_loss.csv'
d = pd.read_csv(path)
df['Siamese CNN Loss'] = d['Value']

path = './data/cnn_single_loss.csv'
d = pd.read_csv(path)
df['Basic CNN Loss'] = d['Value']

path = './data/transformer_single_loss.csv'
d = pd.read_csv(path)
df['Single Transformer'] = d['Value']

df.plot()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()