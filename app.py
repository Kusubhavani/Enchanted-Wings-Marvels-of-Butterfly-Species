from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('butterfly_model.h5')
labels = ['label1', 'label2', 'label3']  # replace with your actual labels

@app.route('/', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        img = image.load_img(request.files['file'], target_size=(224,224))
        x = image.img_to_array(img)/255
        pred = model.predict(np.expand_dims(x,0))
        species = labels[np.argmax(pred)]
        confidence = f"{np.max(pred)*100:.2f}%"
        return render_template('result.html', species=species, confidence=confidence)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
