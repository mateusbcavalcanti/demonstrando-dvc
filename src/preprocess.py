import sys
import pandas as pd
import cv2

from PIL import Image, ImageDraw
import os
import numpy as np
import random

from sklearn.datasets import make_classification

def generate_image(size=128, max_shapes=3):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    n_shapes = random.randint(1, max_shapes)
    for _ in range(n_shapes):
        shape_type = random.choice(['circle','square'])
        color = (255,255,255)
        if shape_type=='circle':
            r = random.randint(8,28)
            cx = random.randint(r, size-r)
            cy = random.randint(r, size-r)
            cv2.circle(img, (cx,cy), r, color, -1)
            cv2.circle(mask, (cx,cy), r, 1, -1)
        else:
            s = random.randint(16,48)
            x = random.randint(0, size-s)
            y = random.randint(0, size-s)
            cv2.rectangle(img, (x,y), (x+s,y+s), color, -1)
            cv2.rectangle(mask, (x,y), (x+s,y+s), 2, -1)
    return img, mask

def save_dataset(n_train=100, n_val=10, size=128):
    for i in range(n_train):
        img, mask = generate_image(size=size)
        Image.fromarray(img).save(f'data/train/images/{i:03d}.png')
        Image.fromarray(mask).save(f'data/train/masks/{i:03d}.png')
    for i in range(n_val):
        img, mask = generate_image(size=size)
        Image.fromarray(img).save(f'data/val/images/{i:03d}.png')
        Image.fromarray(mask).save(f'data/val/masks/{i:03d}.png')


os.makedirs('data/train/images', exist_ok=True)
os.makedirs('data/train/masks', exist_ok=True)
os.makedirs('data/val/images', exist_ok=True)
os.makedirs('data/val/masks', exist_ok=True)

save_dataset(n_train=100, n_val=10, size=128)
print('Dataset criado: 100 treino, 10 val (128x128)')

    



