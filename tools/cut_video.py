import cv2
'''裁剪视频，cap后跟源视频名，start和end为需要裁剪的秒数，output后跟输出视频名'''
cap = cv2.VideoCapture('daolu1.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = 0
start = 20
end = 38
frames_num = (end - start) * fps
start_frame = fps * start
end_frame = fps * end

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('cutVideo1.avi',fourcc,fps,(width,height))

while(True):
    ret,frame = cap.read()
    count += 1
    if start_frame <= count <= end_frame:
        output.write(frame)
    elif count > end_frame:
        break

cap.release()
output.release()
cv2.destroyAllWindows()
