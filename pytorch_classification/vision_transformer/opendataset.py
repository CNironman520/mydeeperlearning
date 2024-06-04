import os
import pickle
import numpy as np
from PIL import Image

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data['data'], data['labels']

def save_images(images, labels, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(len(images)):
        img = Image.fromarray(images[i].astype('uint8'), 'RGB')
        img.save(os.path.join(folder, f'{labels[i]}.png'))

if __name__ == '__main__':
    images, labels = load_data('./cifar10/cifar-10-batches-py/data_batch_1')
    save_images(images, labels, 'cifar10_images')
