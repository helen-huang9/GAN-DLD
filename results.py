from models.transformer import get_siamese_transformer as gst
from models.cnn import get_siamese_cnn as gsc
from experiment_helpers import get_siamese_dataset as gsd
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# Load Data
_, _, X1, Y1 = gsd('all')
split = int(len(Y1) * 0.8)
X_test = X1[:split]
Y_test = Y1[:split]

# Test Transformer
transformer = gst()
transformer.load_weights('./transformer/all_siamese')

y_pred = tf.round(transformer.predict([X_test[:,0], X_test[:,1]]))
y_pred = tf.reshape(y_pred, (-1,1))
confusion = confusion_matrix(Y_test, y_pred)
tn = confusion[1,1]
tp = confusion[0,0]
fn = confusion[0,1]
fp = confusion[1,0]
precision = tp/(tp+fp)
recall = tp/(tp+fn)
accuracy = (tp+tn)/(tp+fp+fn+tn)
print("Transformer:")
print("accuracy: ", accuracy)
print("precision: ", precision)
print("recall: ", recall)

# Test CNN
cnn = gsc()
cnn.load_weights('./cnn/all_siamese')
y_pred = tf.round(cnn.predict([X_test[:,0], X_test[:,1]]))
y_pred = tf.reshape(y_pred, (-1,1))
confusion = confusion_matrix(Y_test, y_pred)
tn = confusion[1,1]
tp = confusion[0,0]
fn = confusion[0,1]
fp = confusion[1,0]
precision = tp/(tp+fp)
recall = tp/(tp+fn)
accuracy = (tp+tn)/(tp+fp+fn+tn)
print("CNN:")
print("accuracy: ", accuracy)
print("precision: ", precision)
print("recall: ", recall)