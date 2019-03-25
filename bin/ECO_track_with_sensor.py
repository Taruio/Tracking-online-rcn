import vrep
import cv2
import glob
import os
from eco import ECOTracker
from eco.features import VGG16Net
from PIL import Image
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
os.environ['VREP_SCENES_PATH'] = r'/home/lhc/Desktop/pyECO-master/'
vrep_scenes_path = os.environ['VREP_SCENES_PATH']

def Connect_vrep(ip = '127.0.0.1', port = 19997, path = vrep_scenes_path):
    '''输入vrep地址和端口,连接vrep并返回可用的clientID'''
    print('程序开始,准备连接V-rep')
    scene_path = path + 'car_cross_sensor.ttt'
    vrep.simxFinish(-1)
    while True:
        cID = vrep.simxStart(ip, port, True, True, 5000, 5)
        if cID > -1:
            print('连接成功')
            err = vrep.simxLoadScene(cID, scene_path, 0, vrep.simx_opmode_blocking)
            if err == 0:
                print('场景加载完成')
            else:
                print('场景加载失败')
            return cID
        else:
            time.sleep(0.50)
            print('连接失败,尝试重新连接')


def Get_Handle(cID, handle_name, mode):
    '''获取物体的handle'''
    while True:
        err, targetHandle = vrep.simxGetObjectHandle(cID, handle_name, mode)
        if err == vrep.simx_return_ok:
            print('已获取handle')
            return targetHandle
        else:
            print('获取handle失败,正在重试')
            time.sleep(0.5)


def Get_Image(cID, handle, mode = vrep.simx_opmode_buffer, count = -2):
    '''后续图像的获取'''
    count += 1
    err, res, visionimage = vrep.simxGetVisionSensorImage(cID, handle, 0, mode)
    visionimage = np.array(visionimage).reshape(256,256,3)
    visionimage = visionimage.astype(np.uint8)
    vrep.simxSynchronousTrigger(cID)
    return err, res, visionimage, count


def Init_vrep(cID, tstep):
    '''vrep参数的初始化'''
    vrep.simxSetFloatingParameter(cID, vrep.sim_floatparam_simulation_time_step, tstep, vrep.simx_opmode_oneshot)
    vrep.simxSynchronous(cID, True)
    vrep.simxStartSimulation(cID, vrep.simx_opmode_oneshot)


def Init_visionimage(cID, handle, mode = vrep.simx_opmode_streaming):
    '''视觉传感器初始化'''
    err, res, visionimage = vrep.simxGetVisionSensorImage(cID, handle, 0, mode)
    vrep.simxSynchronousTrigger(cID)


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
    tstep = 0.2
    # 这儿改好像没有用,dt需要在场景里面改
    fps_count = 0
    quadricopter = 'Quadricopter'
    visionsensor = 'Vision_sensor'
    car = 'Pioneer_p3dx'
    # vrep基本配置信息

    vgg16 = VGG16Net()

    cID = Connect_vrep()
    Init_vrep(cID, tstep)
    visionHandle = Get_Handle(cID, visionsensor, vrep.simx_opmode_blocking)
    Init_visionimage(cID, visionHandle)
    init_position = [23, 110, 60, 40]
    # position中[左上ｘ，左上ｙ，宽，高]
    _, _, visionImage, _ = Get_Image(cID, visionHandle)
    tracker = Init_tracker(visionImage, init_position, vgg16)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('output.avi',fourcc,5.0,(256,256))
    while True:
        err, res, visionImage, fps_count = Get_Image(cID, visionHandle, count = fps_count)
        bbox = Track_target(tracker, visionImage, output, count = fps_count)

