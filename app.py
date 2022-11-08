''' Created on 8th Nov 2022 
@author: Saquib Ayubi'''

from __future__ import division, print_function
#coding=utf-8
import sys
import os
import glob
import re
import numpy as np

#Keras

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

#Flask Utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pysgi import WSGIServer

#Define a Flask app
app= Flask(__name__)

#Model saved with Keras model.save()
MODEL_PATH='model_vgg16.h5'

#load your trained model
model=load_model(MODEL_PATH)


def model_predict(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))

    #preprocessing the image
    x=image.img_to_array(img)
    #scaling
    x=x/255
    x=np.expand_dims(x,axis=0)


    preds=model.predict(x)
    preds=np.argmax(preds,axis=1)
    if preds==0:
        preds='The person is CHRIS Evan or Captain America from Marvel'
    elif preds==1:
        preds='The Person is Henry Cavill or Superman from DC'
    else:
        preds='The Person is Robert Dowyne Jr. or IronMan'
    
    return preds

@app.route('/',methods=['GET'])
def index():
    #MAin Page
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method=='POST':
        #Get the file from post request
        f= request.files['file']

        #Save the file to ./uploads
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(
            basepath,'uploads', secure_filename(f.filename))
        f.save(file_path)

        #Make prediction
        preds=model_predict(file_path,model)
        result=preds
        return result
    return None

if __name__=='__main__':
    app.run(debug=True)    






