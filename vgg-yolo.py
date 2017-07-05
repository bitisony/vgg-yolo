import sys
import math
from functools import cmp_to_key
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, InputLayer
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.models import model_from_json
import numpy as np
from PIL import Image, ImageDraw

from yad2k.yad2k_yolo import yolo_eval, voc_label, cqt_yolo_eval
from yad2k.model import yolo_body, vgg_yolo_model


#img_file = '../img/person.jpg'
#img_file = '../img/dog.jpg'
img_file = 'img/000058.jpg'

# 入力サイズ等はここを変更
width = 416
height = 416
r_w = 13
r_h = 13
r_n = 5
classes = 20



file_post_fix = ''

# モデルの構築
tiny_yolo_model = vgg_yolo_model()

tiny_yolo_model.load_weights('vgg_yolo.h5')

with open('tiny-yolo.json', 'w') as fp:
    json_string = tiny_yolo_model.to_json()
    fp.write(json_string)

tiny_yolo_model.summary()


# run yolo
image = Image.open(img_file)
resized_image = image.resize((width, height), Image.BICUBIC)
image_data = np.array(resized_image, dtype='float32') / 255.0
x = np.expand_dims(image_data, axis=0)

preds = tiny_yolo_model.predict(x)
probs = np.zeros((r_h * r_w * r_n, classes+1), dtype=np.float)
thresh = 0.3

#for l in range(32):
#   layer_dump(tiny_yolo_model, x, l)

np.save('output/preds%s.npy' % file_post_fix, preds)

out_boxes, out_scores, out_classes = cqt_yolo_eval(preds, image.size, score_threshold = thresh, iou_threshold = 0.5, classes = classes)

dr = ImageDraw.Draw(image)

for i in range(len(out_classes)):
    cls = out_classes[i]
    score = out_scores[i]
    box = out_boxes[i]

    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    print(voc_label[cls], score, (left, top), (right, bottom))
    lt = (left, top)
    rt = (right, top)
    lb = (left, bottom)
    rb = (right, bottom)
    red = (255, 0, 0)
    dr.line((lt, rt), red, 2)
    dr.line((lt, lb), red, 2)
    dr.line((rt, rb), red, 2)
    dr.line((lb, rb), red, 2)

image.save(img_file+'out.png')


print("finish")

