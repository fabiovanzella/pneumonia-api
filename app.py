import os
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resource={r"/*":{"origins": "*"}})
IMG_SIZE = (224,224)
FILE_UPLOAD = 'upload'
model = tf.keras.models.load_model("model.weights.best.hdf5")

tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

@app.route('/', methods=['GET'])
def index():
    return "Hello word"

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