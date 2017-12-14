from __future__ import print_function
import numpy as np
#FIX SEED TO COMPUTE RANDOM NUMBERS
np.random.seed(1337)  # for reproducibility
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.models import Sequential
import pickle
from keras.models import load_model
import pandas as pd
from nlpFunctions import *
import argparse
from keras import backend as K

parser = argparse.ArgumentParser(description='Sentiment Analysis')
parser.add_argument('--epoch', action="store", dest="nb", type=int, default=40)
parser.add_argument('--split', action="store", dest="sp", type=float, default=.2)
parser.add_argument('--len', action="store", dest="ln", type=int, default=20)
opts = parser.parse_args()

# Embedding
max_features = 20000
#LENGTH OF SEQUENCES 
maxlen = opts.ln
#NUMBER OF VECTOR LSTM
embedding_size = 128
# Convolution
filter_length = 5
nb_filter = 64
pool_length = 4
# LSTM
lstm_output_size = 70
# Training
# OPTIMIMIZING TRAINING
batch_size = 30
# NUMBER OF EPOCH
nb_epoch = opts.nb
MAX_NB_WORDS= 3000
# MAX NUMBER OF SEQUENCES
MAX_SEQUENCE_LENGTH=50
# CROSS VALIDATION
VALIDATION_SPLIT= opts.sp
#NAME OF THE MODEL
MODEL='6CLASS'
#Opening data
data = openData('sentiment2.csv',',','latin')
comments = getList(data,'comment_message')
sents = orderLabels(data,'class')
#Labels distribution
labelsDistribution(sents,'GRAPHS/labelsDistribution')
shapeCorpus(comments,sents)
#Statistics of corpus
statistics(comments)
showLogComments(comments,50)
#Getting indexs of the words, 
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
#Getting index of words per comment
tokenizer.fit_on_texts(comments)
saveArq(tokenizer,'tokenizer')
sequences = getSequences(comments,tokenizer)
word_index = index(tokenizer)
#Creating sequences of data
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#Vectorizing labels
labels = to_categorical(np.asarray(sents))
#Dimensions of labels and data
showShape(data,labels)
#Spliting data, training and testing
x_train, y_train, x_val, y_val = createMatrix(data,labels,VALIDATION_SPLIT)
matrixDetails(x_train, y_train, x_val, y_val)
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, 64, dropout=0.2))
model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(7))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
metrics=['accuracy'])
model.summary()
train(model, x_train, y_train, 100, nb_epoch, x_val, y_val)
save(model,'MODELS/model'+MODEL)
K.clear_session()
