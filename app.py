from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import cv2

vocab = np.load('vocab.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}

incep = InceptionV3(weights='imagenet')
incep_new = Model(incep.input, incep.layers[-2].output)
vocab_size=1729
max_length=32
embeddings_dim=200


inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embeddings_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.load_weights('model_30.h5')


app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']=1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET','POST'])
def after():
    global model, incep, vocab, inv_vocab
    file=request.files['file1']
    file.save('static/file.jpg')

    img = cv2.imread('static/file.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (299,299))

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)

    incept = incep_new.predict(img).reshape(1,2048)
   # incept = np.reshape(incept, incept.shape[1])

    #incept = incep.predict(img).reshape(1,2048)

    print("="*50)
    print("Predict Features")



    in_text = 'startseq'
    for i in range(max_length):
        sequence = [vocab[w] for w in in_text.split() if w in vocab]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([incept,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = inv_vocab[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    

    return render_template('after.html',data=final)

if __name__=="__main__":
    app.run(debug=True)