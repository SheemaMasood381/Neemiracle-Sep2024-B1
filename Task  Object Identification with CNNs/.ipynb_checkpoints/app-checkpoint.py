from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Load the model
best_model = load_model('my_cnn_model.keras')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_and_predict(model, img_path, target_size=(32, 32)):
    """
    Preprocess the image and predict its class using the trained model.
    
    Args:
    - model: The trained Keras model.
    - img_path: Path to the image file.
    - target_size: Target size for the image (height, width).
    
    Returns:
    - predicted_class: The index of the predicted class.
    """
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image array
    
    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return redirect(request.url)
    files = request.files.getlist('images')
    results = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Predict the class
            predicted_class = preprocess_and_predict(best_model, filepath)
            predicted_class_name = class_names[predicted_class]
            results.append((filename, predicted_class_name))
        else:
            results.append((None, "Unsupported file type"))

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
