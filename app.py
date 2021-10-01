from flask import Flask, flash, render_template, url_for, request, jsonify
import os, base64
from werkzeug.utils import redirect, secure_filename
import urllib.request
# from flask_bootsrtap import Bootstrap
from models.torch_utils import transform_image, get_prediction

app = Flask(__name__)
app.secret_key = "secret key"
UPLOAD_FOLDER = 'static/images/'

image_to_predict = ""
image_data = ""

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    if not(len(image_to_predict)):
        flash("No image uploaded yet")
        return redirect(request.url)
    try:
        tensor = transform_image(image_to_predict)
        prediction = get_prediction(tensor)
        print("res is :", prediction)        
        return render_template('index.html', prediction = prediction, img_data= img_data)
    except:
        return render_template('index.html', error = "Error Occured")


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    global image_to_predict
    image_to_predict = file.read()
    if file.filename == '':
        flash("No image uploaded yet")
        return redirect(request.url)
    try:
        global img_data
        img_data = base64.b64encode(image_to_predict)
        name = file.filename
        img_data = ('data:image/'+ name.split('.')[1]+';base64,' + img_data.decode('utf-8'))
        return render_template('index.html', img_data=img_data)
    except:
        flash("Error occured")
        return redirect(request.url)

if __name__ == "__main__":
    app.run()