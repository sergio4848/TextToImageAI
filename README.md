
# AI Text To Image

This project demonstrates how to develop an AI application that generates logos and photos using Generative Adversarial Networks (GANs). The application includes data collection, model training, and a web interface for user interaction.

## Project Structure

```
AI-Logo-Photo-Generator/
├── data/
│   └── collect_data.py
├── models/
│   ├── __init__.py
│   ├── discriminator.py
│   ├── generator.py
│   └── gan.py
├── training/
│   └── train_gan.py
├── webapp/
│   ├── app.py
│   └── templates/
│       └── index.html
├── utils/
│   └── save_images.py
└── requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-Logo-Photo-Generator.git
   cd AI-Logo-Photo-Generator
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Collection

Place your data collection and preprocessing scripts in `data/collect_data.py`. Ensure that your dataset is properly labeled and formatted for training.

### Model Definition

The GAN model consists of three parts:

- **Generator**: `models/generator.py`
- **Discriminator**: `models/discriminator.py`
- **GAN**: `models/gan.py`

### Training

Train the GAN model using the script in `training/train_gan.py`. Customize the training parameters as needed.

```bash
python training/train_gan.py
```

### Web Application

A simple web interface is provided using Flask. This allows users to generate images via a web form.

1. Run the Flask app:
   ```bash
   python webapp/app.py
   ```

2. Open a web browser and go to `http://127.0.0.1:5000`.

### Example Scripts

#### Generator Model (`models/generator.py`)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape

def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(28 * 28 * 1, activation='tanh'))
    model.add(Reshape((28, 28, 1)))
    return model
```

#### Discriminator Model (`models/discriminator.py`)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Flatten

def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model
```

#### GAN Model (`models/gan.py`)

```python
from tensorflow.keras.models import Sequential

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model
```

#### Training Script (`training/train_gan.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
from models.generator import build_generator
from models.discriminator import build_discriminator
from models.gan import build_gan
from utils.save_images import save_imgs
import tensorflow as tf

def train_gan(epochs, batch_size=128, save_interval=50):
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    half_batch = int(batch_size / 2)

    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

    generator = build_generator()
    z = tf.keras.layers.Input(shape=(100,))
    img = generator(z)
    discriminator.trainable = False
    valid = discriminator(img)
    combined = tf.keras.models.Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]
        noise = np.random.normal(0, 1, (half_batch, 100))
        gen_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.array([1] * batch_size)
        g_loss = combined.train_on_batch(noise, valid_y)

        if epoch % save_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]}] [D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")
            save_imgs(generator, epoch)

train_gan(epochs=10000, batch_size=32, save_interval=1000)
```

#### Save Images Utility (`utils/save_images.py`)

```python
import matplotlib.pyplot as plt
import numpy as np

def save_imgs(generator, epoch, examples=3, dim=(1, 3), figsize=(3, 1)):
    noise = np.random.normal(0, 1, (examples, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    for i in range(examples):
        axs[i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
        axs[i].axis('off')
    plt.show()
```

#### Web Application (`webapp/app.py`)

```python
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
```

#### HTML Template (`webapp/templates/index.html`)

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Logo and Photo Generator</title>
</head>
<body>
    <h1>AI Logo and Photo Generator</h1>
    <form action="/generate" method="post">
        <label for="command">Enter Command:</label>
        <input type="text" id="command" name="command" required>
        <button type="submit">Generate</button>
    </form>
</body>
</html>
```

## Contributing

Feel free to contribute by submitting a pull request. Please ensure that your contributions adhere to the existing coding style and include appropriate tests.

