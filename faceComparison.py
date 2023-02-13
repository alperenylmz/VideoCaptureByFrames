import cv2
import face_recognition

img = cv2.imread("/home/alperen/Downloads/videoCapImages/160.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread("/home/alperen/Downloads/videoCapImages/161.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2_encoding = face_recognition.face_encodings(rgb_img2)[0]

result = face_recognition.compare_faces([img_encoding], img2_encoding)
print("result: " + str(result))