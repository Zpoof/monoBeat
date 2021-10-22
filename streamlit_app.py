import streamlit as st
import os
import pandas as pd
import librosa
import librosa.display
import glob 
import matplotlib.pyplot as plt
import wave
import tensorflow as tf
import keras
from statistics import mode
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard,ProgbarLogger
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import itertools

def load_file_data (sound_file, duration = 12, sr = 16000):
    input_length=sr*duration

    data = []
    X, sr = librosa.load(sound_file, sr=sr, duration=duration,res_type='kaiser_fast') 
    dur = librosa.get_duration(y=X, sr=sr)
    # pad audio file same duration
    if (round(dur) < duration):
      y = librosa.util.fix_length(X, input_length)              
        
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)               
    feature = np.array(mfccs).reshape([-1,1])
    data.append(feature)
    return data

CLASSES = ['artifact','murmur','normal']
SAMPLE_RATE = 16000
MAX_SOUND_CLIP_DURATION=12   
best_model_file="/content/drive/MyDrive/best_model_trained.hdf5"

label_to_int = {k:v for v,k in enumerate(CLASSES)}
int_to_label = {v:k for k,v in label_to_int.items()}

model = Sequential()
model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=True,input_shape = (40,1)))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(len(CLASSES), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc','mse', 'mae', 'mape', 'cosine'])
model.summary()

model.load_weights(best_model_file)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#y_pred = model.predict(x_test, batch_size=32) 
#y_pred = np.argmax(y_pred,axis=1)

#scores = model.evaluate(x_test, y_test, verbose=0)

st.image(Image.open('/content/drive/MyDrive/log.png'))
uploaded_file = st.file_uploader("Choose a file", ['wav'])


if uploaded_file is not None:
  #audio_bytes = uploaded_file.read()
  st.audio(uploaded_file, format='audio/wav')
  epic = np.asarray(load_file_data(uploaded_file))

  #st.write(epic)
  #yes = [[epic[0]]]
  #st.write(len(epic))
  #st.write(type(epic))

  y_pred = model.predict(epic, batch_size=32)
  #st.write(y_pred)
  res = np.argmax(y_pred,axis=1)
  #st.write(res[0])
  #st.write(y_pred[0][1])
  #st.write(len(y_pred))
  #st.write("prediction test return :",mode(y_pred), "-", int_to_label[mode(y_pred)])
  st.header("Prediction: " + int_to_label[res[0]])
  st.header("Confidence: " + str(round(y_pred[0][res][0] * 100, 2)) + "%")
  st.header("Chance of Abnormality: " + str(round((y_pred[0][1] + y_pred[0][2]) * 100, 2)) + "%")


