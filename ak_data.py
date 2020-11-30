import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from sklearn.datasets import load_files

dataset = tf.keras.utils.get_file(
    fname="aclImdb.tar.gz",
    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    extract=True,
)

# set path to dataset
IMDB_DATADIR = os.path.join(os.path.dirname(dataset), 'aclImdb')

classes = ['pos', 'neg']
train_data = load_files(os.path.join(IMDB_DATADIR, 'train'), shuffle=True, categories=classes)
test_data = load_files(os.path.join(IMDB_DATADIR,  'test'), shuffle=False, categories=classes)

x_train = np.array(train_data.data)
y_train = np.array(train_data.target)
x_test = np.array(test_data.data)
y_test = np.array(test_data.target)

print(f'x_train.shape = {x_train.shape}')  # (25000,)
print(f'y_train.shape = {y_train.shape}')  # (25000, 1)

# For a demo, only use a subset of the data for training
ITEMS=1000

x_train = x_train[:ITEMS]
y_train = y_train[:ITEMS]

print(f'Reduced dataset to {ITEMS} items')

x_test = x_test[:ITEMS//10]
y_test = y_test[:ITEMS//10]

print(f'Reduced testset to {ITEMS//10} items')
