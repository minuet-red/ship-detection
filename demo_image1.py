from yolo import YOLO
from PIL import Image
import imutils
import cv2
import os

yolo = YOLO()

files = os.listdir('./test/')

for file in files:
    img = './test/' + file
    image = Image.open(img)
    r_image = yolo.detect_image(image)
    r_image.show()
    