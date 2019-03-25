import cv2
import glob
import os
from eco import ECOTracker
from eco.features import VGG16Net
import time
import numpy as np
from tools import get_rect

def get_frame(cap):
    ret,frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    return frame


def Init_tracker(firstframe, position, network):
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
    tracker = ECOTracker(is_color, network)
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


if __name__ == '__main__':
    fps_count = 0
    
    cap = cv2.VideoCapture(r'../car_zhedang_fast.avi')
    firstframe = get_frame(cap)
    vgg16 = VGG16Net()
    (left_up,right_down) = get_rect(firstframe)
    init_position = [left_up[0],left_up[1],right_down[0]-left_up[0],right_down[1]-left_up[1]]
    print(left_up,'left_up')
    print(right_down,'right_down')
    print(init_position)
    tracker = Init_tracker(firstframe, init_position, vgg16)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('output.avi', fourcc, 5.0, (256, 256))
    while True:
        nextframe = get_frame(cap)
        fps_count += 1
        bbox = Track_target(tracker, nextframe, output, count=fps_count)
