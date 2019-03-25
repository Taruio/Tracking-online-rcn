from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
from eco import ECOTracker

from tools import get_rect,sift_feature,cal_matches,setflann


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES1 = ('__background__',
           'car')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def get_frame_change(cap):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def get_frame(cap):
    ret,frame = cap.read()
    return frame


def Init_tracker(firstframe, position, session, network):
    frame = firstframe
    height, width = frame.shape[:2]
    if len(frame.shape) == 3:
        is_color = True
        # 彩色图每张（frame）为[h,w,3]的形式
    else:
        is_color = False
        frame = frame[:, :, np.newaxis]
        # 灰度图每张转化为[h,w,1]的形式
    # 目标位置信息posision的格式为(xmin,ymin,w,h)
    tracker = ECOTracker(is_color, session, network)
    # 初始化追踪器（类）
    # 第一帧操作
    bbox = position
    # 目标初始位置（第一帧位置）
    tracker.init(frame, bbox)
    bbox = (bbox[0]-1, bbox[1]-1,
            bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)
    # bbox改为左上角和右下角的位置
    frame = cv2.rectangle(frame,
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          (0, 255, 0),
                          2)
    frame = frame.squeeze()
    # 真实位置画框
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 由于cv2的imshow图像格式为BGR，而Image读取的图像格式为RGB，转换
    frame = cv2.putText(frame, str('start'), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
    # 在图像上显示帧数
    cv2.imshow('track', frame)
    cv2.waitKey(30)
    return tracker


def Track_target(tracker, frame, videocapture,view = True, count = 0, capture_flag = False ):
    bbox = tracker.update(frame, True)
    if view:
        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 0),
                              2)
        frame = frame.squeeze()
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 由于cv2的imshow图像格式为BGR，而Image读取的图像格式为RGB，转换
        if tracker.flag == 1:
            frame = cv2.putText(frame, str('missed'), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        frame = cv2.putText(frame, str(count), (5, 230), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        # 在图像上显示帧数
        cv2.imshow('track', frame)
        if capture_flag == True:
            videocapture.write(frame)
        cv2.waitKey(30)
        return tracker
    else:
        return tracker, bbox


def vis_detections(im, class_name, dets, tar_size, tar_kp, tar_des, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    max_score = 0
    max_pos = [0,0,0,0]
    # 转换色彩空间
    flann = setflann()
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        frame = cv2.rectangle(frame, (bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
        candid = im[int(bbox[1]):int(bbox[3]+1),int(bbox[0]):int(bbox[2]+1),:]
        cv2.imwrite(str(i)+str(candid[0][0][0])+'.jpg',candid)
        kp2, des2 = sift_feature(candid, tar_size)
        simi = cal_matches(flann, tar_kp, tar_des, kp2, des2)
        if simi >= max_score:
            max_score = simi
            max_pos[0] = int(bbox[0])
            max_pos[1] = int(bbox[1])
            max_pos[2] = int(bbox[2])
            max_pos[3] = int(bbox[3])
            # 从上向下：左上ｘ，左上ｙ，右上ｘ，右上ｙ
        if bbox[1] < 20:
            frame = cv2.putText(frame, '{:s} {:.3f}'.format(class_name, score),
                                (bbox[0], int(bbox[1] + bbox[3] - 2)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 0, 255), 1)
        else:
            frame = cv2.putText(frame,'{:s} {:.3f}'.format(class_name, score),
                                (bbox[0],int(bbox[1]-2)),cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 0, 255), 1)

    cv2.imshow(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),frame)
    cv2.waitKey(2000)
    return max_score,max_pos

def demo(sess, net, frame, targetsize, keypoint, descrip):
    """
    Detect object classes in an image using pre-computed object proposals.
    frame为传入的图像帧
    """

    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(sess, net, frame)

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        if cls == 'car':
            max_score, max_pos = vis_detections(frame, cls, dets, targetsize, keypoint, descrip, thresh=CONF_THRESH)
            print('similarity:', max_score)
            print('position:', max_pos)
    return max_pos


if __name__ == '__main__':
    # model path
    demonet = 'vgg16'
    dataset = 'pascal_voc_0712'
    tfmodel = r'/home/lhc/Desktop/Tracking-online/default/voc_2007_trainval/default/vgg16_faster_rcnn_iter_50000.ckpt'

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 21,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print('Loaded network {:s}'.format(tfmodel))

    # 加载用于跟踪的vgg16
    # load initial target
    target = cv2.imread(r'sequences/car.png')
    tar_size = target.shape[0:2]
    tar_kp, tar_des = sift_feature(target, tar_size)

    # load video and get the first frame to init
    cap = cv2.VideoCapture('car_zhedang_fast.avi')
    firstframe = get_frame(cap)

    position = demo(sess, net, firstframe, tar_size, tar_kp, tar_des)
    init_pos = [position[0], position[1], position[2]-position[0], position[3]-position[1]]
    fps_count = 0
    tracker = Init_tracker(firstframe, init_pos, sess, net)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('output.avi', fourcc, 5.0, (256, 256))
    while True:
        nextframe = get_frame(cap)
        fps_count += 1
        bbox = Track_target(tracker, nextframe, output, count=fps_count)

