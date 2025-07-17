from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'your_secret'
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('model/model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pwd = request.form['password']
        if user == 'admin' and pwd == 'pass':
            session['user'] = user
            return redirect(url_for('predict'))
        return render_template('login.html', error='Invalid')
    return render_template('login.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        img = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(path)
        image = load_img(path, target_size=(224,224))
        arr = img_to_array(image)/255.0
        arr = np.expand_dims(arr, axis=0)
        pred = model.predict(arr)
        cls = np.argmax(pred, axis=1)[0]
        return render_template('predict.html', label=str(cls), prob=float(np.max(pred)))
    return render_template('predict.html', label=None)
