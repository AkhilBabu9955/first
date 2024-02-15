from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import keras
import numpy as np
import pickle
from fastai.vision.all import *
import pathlib
import cv2
from PIL import Image
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
app = Flask(__name__, template_folder='template') 
#app = Flask(__name__)
model = load_learner(r'D:\\project\\model.pkl')

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./image/" + imagefile.filename
    #imagefile.save(image_path)
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    im=Image.fromarray(image)
    image=np.asarray(im)
    

    # image = load_img(image_path, target_size=(224, 224, 3))
    # image = img_to_array(image)      #tensorflow
    # image = np.expand_dims(image,axis=0)
    # image = image/255.0
    y = model.predict(image)
    labels = {0:"Human",1:"Not Human"}
    y = np.argmax(y)
    output = labels[y]
    return render_template('index.html', output=output)


if __name__ == '__main__':
    app.run( debug=True)#port=500



