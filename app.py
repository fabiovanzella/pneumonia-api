import os
import numpy as np
import tensorflow as tf
import wget

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
cors = CORS(app, resource={r"/*":{"origins": "*"}})
IMG_SIZE = (224,224)
FILE_UPLOAD = 'static/upload'

local_file = 'model.weights.best.hdf5'
remote_url = "https://github.com/fabiovanzella/pneumonia-api/releases/download/v0.2-model/model.weights.best.hdf5"
wget.download(remote_url, local_file)

model = tf.keras.models.load_model("model.weights.best.hdf5")

tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

@app.route('/', methods=['GET'])
def index():    
    return app.send_static_file('index.html')

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

def main():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()