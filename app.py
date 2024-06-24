from flask import Flask, request, render_template, session
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
app.secret_key = 'your_secret_key'  

model = load_model(r"D:\ProjectNew\efficientnetmodelfinal.h5")

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class_full_forms = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "df": "Dermato Fibroma",
    "bkl": "Benign Keratosis Lesion",
    "nv": "Melanocytic Nevi",
    "mel": "Melanoma",
    "vasc": "Vascular Lesion",
    "healthy": "Healthy Skin"
}

@app.route('/')
def home():
    return render_template('indexnew.html')

@app.route('/upload', methods=['GET'])
def upload():
    return render_template('upload2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('upload2.html', error='No file part')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('upload2.html', error='No selected file')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        img_size = (224, 224)
        img = keras_image.load_img(file_path, target_size=img_size)
        img_array = keras_image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        predicted_class_index = np.argmax(model.predict(img_array))
        class_names = ["akiec", "bcc", "bkl", "df", "healthy", "mel", "nv", "vasc"]
        predicted_class_name = class_names[predicted_class_index]
        predicted_class_full_form = class_full_forms.get(predicted_class_name, "Unknown Class")
        
        session['predicted_class'] = predicted_class_full_form
        
        return render_template('prediction_result2.html', image_file=file_path, predicted_class=predicted_class_full_form)
    
    else:
        return render_template('upload2.html', error='File type not allowed')

@app.route('/symptoms')
def symptoms():
    predicted_class = session.get('predicted_class', 'Unknown') 
    return render_template('symptom.html', predicted_class=predicted_class)  

@app.route('/doctors')
def display_doctors():
    return render_template('Doctor_details.html')

if __name__ == '__main__':
    app.run(debug=True)
