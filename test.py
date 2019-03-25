import cv2

cap = cv2.VideoCapture('sequences/car_zhedang.avi')
ret, frame = cap.read()
print(frame)
cv2.imwrite('123.jpg',frame)