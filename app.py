#!/usr/bin/env python
# -*- coding: utf-8 -*-
     
import configparser
import argparse
import time
import numpy as np
np.random.seed(1337)  # for reproducibility
from flask import Flask
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.models import load_model
#from keras.datasets import imdb
import pickle
from keras.models import load_model

MODEL="6CLASS"
# Embedding
max_features = 20000
maxlen = 20
embedding_size = 128
# Convolution
filter_length = 5
nb_filter = 64
pool_length = 4
# LSTM
lstm_output_size = 70
# Training
batch_size = 30
nb_epoch = 40
MAX_NB_WORDS= 3000
MAX_SEQUENCE_LENGTH=50
VALIDATION_SPLIT=.2
MODEL='6CLASS'

#with open('MODELS/tokenizer','rb') as idxf:
with open('tokenizer','rb') as idxf:
   tok=pickle.load(idxf)

model=load_model('MODELS/model6CLASS')
model.summary()
app = Flask(__name__)

#senti=["Enojado","Molesto","Desacuerdo","Indiferente","Agrado","Entusiasmado","Eufórico"]
#senti=["furious","angry","angry0","Indiferent","happy","enthusiastic","Euphoric"]
#images=['home/image0.jpg','home/image1.jpg','home/image2.jpg','home/image3.jpg','home/image4.jpg','home/image5.jpg','home/image6.jpg']


senti=["furious","angry","angry0","Indiferent","happy","enthusiastic","Euphoric"]


def predict(text):
	seqs = tok.texts_to_sequences([text])
	print(text)
	word_index = tok.word_index
	print('Found %s unique tokens.' % len(word_index))
	sequence_pred = sequence.pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH)
	print(sequence_pred)
	prediction = model.predict(sequence_pred)
	print(prediction)
	return senti[np.argmax(prediction[0])]

@app.route("/", methods=['GET', 'POST'])
def index():
    senti=["furious","angry","angry0","Indiferent","happy","enthusiastic","Euphoric"]
    images=['static/angry.jpg','static/angry.jpg','static/angry.jpg','static/Indifferent.png','static/smile.jpg','static/smile.jpg','static/smile.jpg']
    lookup_keys = dict(zip(senti, images))
    print(request.method)
    if request.method == 'POST':
        q=request.form['querytext']
        prediction=predict(q)
        image_path = lookup_keys[prediction] # get the path
        print(image_path)
        return render_template("result.html",
                               prediction=prediction,
                               text=q,
                               image_url=image_path)
    return render_template("main.html")







if __name__ == '__main__':
    p = argparse.ArgumentParser("Twitter ML")
    p.add_argument("--host",default="127.0.0.1",
            action="store", dest="host",
            help="Root url [127.0.0.1]")
    p.add_argument("--port",default=5000,type=int,
            action="store", dest="port",
            help="Port url [500]")
    p.add_argument("--debug",default=False,
            action="store_true", dest="debug",
            help="Use debug deployment [Flase]")
    p.add_argument("-v", "--verbose",
            action="store_true", dest="verbose",
            help="Verbose mode [Off]")

    opts = p.parse_args()
    
    app.run(debug=opts.debug,
            host=opts.host,
            port=opts.port)
