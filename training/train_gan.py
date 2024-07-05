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
