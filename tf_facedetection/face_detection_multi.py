#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
注：针对face_detection.py稍微做了修改，使其能够遍历多个图片
@Project name: tf_facedetection
@Product name: PyCharm
@Time: 2018/3/7 17:02
@Author: ldgang
'''

#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
注：该案例源码是摘自 https://github.com/ShyBigBoy/face-detection-mtcnn
已调试
'''
import detect_face
from scipy import misc
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
gpu_memory_fraction=1.0


print('Creating networks and loading parameters')

with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}, log_device_placement=True))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

cur_dir = os.getcwd()
file = [s for s in os.listdir(cur_dir+'./images/')]
for key in file:
    image_path = cur_dir+"/images/"+key
    img = misc.imread(image_path)
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]#人脸数目
    print('找到人脸数目为：{}'.format(nrof_faces))

    print(bounding_boxes)

    crop_faces=[]
    for face_position in bounding_boxes:
        face_position=face_position.astype(int)
        print(face_position[0:4])
        cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        crop=img[face_position[1]:face_position[3],
                 face_position[0]:face_position[2],]

        crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )
        print(crop.shape)
        crop_faces.append(crop)
        plt.imshow(crop)
        plt.show()

    plt.imshow(img)
    plt.show()
