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
