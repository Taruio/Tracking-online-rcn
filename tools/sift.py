import numpy as np
import cv2


def sift_feature(img, size):
    sift = cv2.xfeatures2d.SIFT_create()
    img = cv2.resize(img, (size[1],size[0]))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoint, descrip = sift.detectAndCompute(img, None)
    return keypoint, descrip


def setflann(kdtree = 0, trees = 5, checks = 50):
    index_params = dict(algorithm = kdtree, trees = trees)
    search_params = dict(checks = checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return flann


def cal_matches(flann,kp1,des1,kp2,des2):
    if len(kp1) == 0 or len(des1) == 0 or len(kp2) == 0 or len(des2) == 0:
        return 0
    matches = flann.knnMatch(des1,des2,k=2)
    machesMask = [[0,0] for i in range(len(matches))]

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])
    similarity = (len(good)/len(matches))
    return similarity

