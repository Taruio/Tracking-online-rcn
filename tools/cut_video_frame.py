import cv2
'''裁剪视频，cap后跟源视频名，start和end为需要裁剪的秒数，output后跟输出视频名'''
cap = cv2.VideoCapture('daolu2_zhedang.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

start = 18
end = 26
insert_frame = 5

count = 0
frames_num = (end - start) * fps
start_frame = fps * start
end_frame = fps * end
frame_count = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('cutVideo1.avi',fourcc,fps,(width,height))

while(True):
    ret,frame = cap.read()
    count += 1
    if start_frame <= count <= end_frame:
        frame_count += 1
        if frame_count == insert_frame:
            output.write(frame)
            frame_count = 0
    elif count > end_frame:
        print('done')
        break

cap.release()
output.release()
cv2.destroyAllWindows()
