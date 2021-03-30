from detect import YOLO
from PIL import Image
import imutils
import cv2
import os

yolo = YOLO()

files = os.listdir('./test/')

for file in files:
    if file.endswith('jpg') or file.endswith('bmp'):
        image_path = './test/' + file
        image = cv2.imread(image_path)
        boxes = yolo.detect_image(Image.open(image_path))
        print(image_path)
        if len(boxes) > 0:
            for box in boxes:
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                score = box[4]
                predicted_class = box[5]

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0),  2)
                #label = str(predicted_class) + ':' + str(score)[:4]
                label = str(predicted_class) + ' ' + str(score * 100)[:4] + '%'
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        image = imutils.resize(image, width=720)
        cv2.imshow('', image)
        cv2.waitKey(0)



