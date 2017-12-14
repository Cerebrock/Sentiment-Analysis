import pandas as pd
import numpy as np
#FIX SEED TO COMPUTE RANDOM NUMBERS
np.random.seed(1337)  # for reproducibility
import pickle
from keras.preprocessing import sequence
#Creating data frame from corpus
def openData(corpus,separator,encoding):
    data = pd.read_csv(corpus, sep=separator , encoding=encoding)
    return data

def getList(dataframe,column):
    list = dataframe[column].astype(str).tolist()
    return list

def orderLabels(data,labels):
    sents = [x-1 for x in data[labels].tolist()]
    return sents

def shapeCorpus(comments,sents):
    print("Size of texts:", len(comments))
    print("Size of labels:", len(sents))


def statistics(comments):
    #STATISTICS OF TEXTS
    print ("Avg size of texts:",np.mean([len(x.split(' ')) for x in comments ]))
    print ("Max size of texts:",np.max([len(x.split(' ')) for x in comments ]))
    print ("Min size of texts:",np.min([len(x.split(' ')) for x in comments ]))

def showLogComments(comments, number):
    for x in comments:
        if len(x.split(' ')) > number:
            print(x)


def saveArq(tokenizer,name):
    with open(name,'wb') as idxf:
        pickle.dump(tokenizer, idxf, pickle.HIGHEST_PROTOCOL)

def getSequences(comments,tokenizer):
    sequences = tokenizer.texts_to_sequences(comments)
    return sequences


def index(tokenizer):
    word_index = tokenizer.word_index
    return word_index


def createMatrix(data,labels,VALIDATION_SPLIT):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    return x_train, y_train, x_val, y_val


def showShape(data,labels):
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

def matrixDetails(x_train, y_train, x_val, y_val):
    print('Shape of train:',x_train.shape)
    print('Shape of train:',y_train.shape)
    print('Shape of test:',x_val.shape)
    print('Shape of test labels:',y_val.shape)


def train(model, x_train, y_train, batch_size,nb_epoch,x_val,y_val):
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    validation_data=(x_val, y_val))


def save(model,name):
    model.save(name)


def labelsDistribution(sents,name):
    dictDistribution={}
    for i in set(sents):
        print("labels distribution "+str(i)+":",sents.count(i))
        dictDistribution[i]=sents.count(i)





