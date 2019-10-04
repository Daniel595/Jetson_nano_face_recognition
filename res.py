import cv2
import os
import face_recognition
import numpy as np 



cnt = 0

small_frame = cv2.imread("4.jpg")
print(type(small_frame))
rgb_small_frame = small_frame

face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
for face_location in face_locations:
    top, right, bottom, left = face_location
    face_image = small_frame[top:bottom, left:right]
    face_image = cv2.resize(face_image, (150, 150))
    cv2.imshow("face", face_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = cv2.resize(face_image,(150,150))
    face_name = "res_" + str(cnt) + ".jpg"
    cv2.imwrite(face_name, face_image)
    cnt = cnt+1
    


'''
img = cv2.imread("4.png")
img = cv2.resize(img,(150,150))
cv2.imwrite("res.jpg", img)
'''

