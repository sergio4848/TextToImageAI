from flask import Flask, request, render_template
import numpy as np
from models.generator import build_generator
import matplotlib.pyplot as plt

app = Flask(__name__)
generator = build_generator()
generator.load_weights('path_to_your_trained_generator_weights.h5')  # Load pre-trained weights

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_command = request.form['command']
    noise = np.random.normal(0, 1, (1, 100))
    gen_img = generator.predict(noise)
    plt.imshow(gen_img[0, :, :, 0], cmap='gray')
    plt.show()
    return 'Image generated!'

if __name__ == '__main__':
    app.run(debug=True)
