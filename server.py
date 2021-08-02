import pandas as pd
from flask import Flask,render_template
from flask import request, jsonify
from flask_cors import CORS, cross_origin
import urllib.request
import joblib
import os.path
import csv
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import os
import json
from tqdm import tqdm
import librosa.display
from keras.layers import Input, Dense
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint
import random
import string
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
model_encode = InceptionV3(include_top=True, weights='imagenet')
model_encode = Model(model_encode.input, model_encode.layers[-2].output)


# Khai báo port của server
my_port = '5000'

app = Flask(__name__)
CORS(app)

# Khai bao ham xu ly request index
@app.route('/')
@cross_origin()
def index():
    return render_template('client.html')

model = load_model('./model/model_cough.h5')

def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inceptionv3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    return x
  
def sound_to_pic(link):
  audiofile = link
        
  # Loading the image with no sample rate to use the original sample rate and
  # kaiser_fast to make the speed faster according to a blog post about it (on references)
  X, sample_rate = librosa.load(audiofile, sr=None, res_type='kaiser_fast')

  # Setting the size of the image
  fig = plt.figure(figsize=[1,1])

  # This is to get rid of the axes and only get the picture 
  ax = fig.add_subplot(111)
  ax.axes.get_xaxis().set_visible(False)
  ax.axes.get_yaxis().set_visible(False)
  ax.set_frame_on(False)

  # This is the melspectrogram from the decibels with a linear relationship
  # Setting min and max frequency to account for human voice frequency
  S = librosa.feature.melspectrogram(y=X, sr=sample_rate)
  librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', fmin=50, fmax=280)

  # This is the melspectrogram from the decibels with a linear relationship
  # using the function for train, val and test to make the function easy to use and output
  # the images in different folders to use later with a generator
  name = './img/' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
  file  = str(name) + '.jpg'

  # Here we finally save the image file choosing the resolution 
  plt.savefig(file, dpi=500, bbox_inches='tight',pad_inches=0)
  return file

def file_to_str(link):
  file=sound_to_pic(link)
  image = preprocess(file) # preprocess the image
  fea_vec = model_encode.predict(image) # Get the encoding vector for the image
  fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
  print(123123)
  return str(np.argmax(model.predict(fea_vec.reshape(1,2048))))

@app.route('/',methods=['POST'])
def my_form_post():
    text=request.form['u']
    string_x = file_to_str(text)
    result=''
    if string_x=='0':
        result = 'healthy'
    elif string_x=='1':
        result = 'no_resp_illness_exposed'
    elif string_x=='2':
        result = 'positive_asymp'
    elif string_x=='3':
        result = 'positive_mild'
    elif string_x=='4':
        result = 'positive_moderate'
    elif string_x=='5':
        result = 'recovered_full'
    elif string_x=='6':
        result = 'resp_illness_not_identified'
    print(result)
    return render_template('client.html',text=result)

# Thuc thi server
if __name__ == '__main__':
    app.run(debug=True, host='localhost',port=my_port)