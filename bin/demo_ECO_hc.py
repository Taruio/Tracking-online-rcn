import glob
import os
import pandas as pd
import argparse
import numpy as np
import cv2
from eco import ECOTracker
from PIL import Image

import argparse

def main(video_dir):
    # load videos
    filenames = sorted(glob.glob(os.path.join(video_dir, "img/*.jpg")),
           key=lambda x: int(os.path.basename(x).split('.')[0]))
    # frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    frames = [np.array(Image.open(filename)) for filename in filenames]
    height, width = frames[0].shape[:2]
    if len(frames[0].shape) == 3:
        is_color = True
        # 彩色图每张（frame）为[h,w,3]的形式
    else:
        is_color = False
        frames = [frame[:, :, np.newaxis] for frame in frames]
        # 灰度图每张转化为[h,w,1]的形式
    gt_bboxes = pd.read_csv(os.path.join(video_dir, "groundtruth_rect.txt"), sep='\t|,| ',
            header=None, names=['xmin', 'ymin', 'width', 'height'],
            engine='python')
    # 读取目标位置信息（全部）

    title = video_dir.split('/')[-1]
    # starting tracking
    tracker = ECOTracker(is_color)
    # 初始化追踪器（类）

    for idx, frame in enumerate(frames):
        if idx == 0:
            # 第一帧
            bbox = gt_bboxes.iloc[0].values
            # 目标初始位置（第一帧位置）
            tracker.init(frame, bbox)
            bbox = (bbox[0]-1, bbox[1]-1,
                    bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)

            # bbox改为左上角和右下角的位置
        elif idx < len(frames) - 1:
            bbox = tracker.update(frame, True)
            # 追踪每一帧的目标位置
        else: # last frame
            bbox = tracker.update(frame, False)
            # 追踪最后一帧的目标位置
        # bbox xmin ymin xmax ymax
        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 0),
                              2)
        # 预测位置画框
        gt_bbox = gt_bboxes.iloc[idx].values
        gt_bbox = (gt_bbox[0], gt_bbox[1],
                   gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3])
        frame = frame.squeeze()
        frame = cv2.rectangle(frame,
                              (int(gt_bbox[0]-1), int(gt_bbox[1]-1)), # 0-index
                              (int(gt_bbox[2]-1), int(gt_bbox[3]-1)),
                              (255, 0, 0),
                              1)
        # 真实位置画框
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 由于cv2的imshow图像格式为BGR，而Image读取的图像格式为RGB，转换
        frame = cv2.putText(frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        # 在图像上显示帧数
        cv2.imshow(title, frame)
        cv2.waitKey(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='../sequences/Crossing/')
    args = parser.parse_args()
    # 运行参数为视频存放路径
    main(args.video_dir)
