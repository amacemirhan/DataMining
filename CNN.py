import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
from keras import layers
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Activation, Dense
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Activation,Flatten
from keras import metrics, regularizers
from tensorflow.keras.optimizers import RMSprop
from sklearn import metrics
ddf = pd.read_csv("yuz_veri_kumesi.csv", sep=";")
df = ddf[['Cumle','Sinif']]

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Cumle'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df['Cumle'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(df['Sinif']).values
print('Shape of label tensor:', Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
epochs = 16
batch_size = 256
model = Sequential()
model.add(Flatten())
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(LSTM(256, activation='relu', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(128, activation='relu', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(128, activation='relu', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(64, activation='relu', dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(512, activation='LeakyReLU'))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

accr= model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.4f}\n  Accuracy: {:0.4f}'.format(accr[0],accr[1]))
y_test = [row[0] for row in Y_test]

from sklearn import metrics
y_pred=model.predict(X_test)
dene = np.argmax(y_pred, axis=1)
a = np.argmax(Y_test, axis=1)
print("F1: ",f1_score(a[:],dene[:],average='weighted'))
