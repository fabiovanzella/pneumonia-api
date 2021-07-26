#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 09:31:16 2021

@author: fabio
"""


import flask
import numpy as np
import tensorflow as tf

from flask import request, jsonify

 # Disable all GPUS
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'
        
# Carrega o melhor modelo
model = tf.keras.models.load_model("model.weights.best.hdf5")

app = flask.Flask(__name__)
app.config["DEBUG"] = False

IMG_SIZE = (224,224)
FILE_UPLOAD = 'upload'

@app.route('/predict', methods=['POST'])
def predict():  
    
    file = request.files['file']
    file.save(FILE_UPLOAD)
        
    img = tf.keras.preprocessing.image.img_to_array(
            tf.keras.preprocessing.image.load_img(FILE_UPLOAD, target_size=IMG_SIZE, color_mode='rgb'))/255
            
    # Dimensiona a imagem
    img = np.expand_dims(img,0)
    
    print(img.shape)
    
    # Aplica a prediçao
    pred = model.predict(img)
    
    #print('Classification Report')
    
    resultado = jsonify(
        normal= str(round(pred[0][0], 2)),
        pneumonia= str(round(pred[0][1], 2))
    )
    return resultado

app.run()