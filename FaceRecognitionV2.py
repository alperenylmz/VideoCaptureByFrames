import face_recognition
import imutils
import pickle
import time
import cv2
import os
 
#find path of xml file containing haarcascade file
cascPathface = "/home/alperen/Downloads/faceRec/FaceDetect-master/haarcascade_frontalface_default.xml"

# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)

# load the known faces and embeddings saved in last file
data = pickle.loads(open('/home/alperen/Downloads/face_enc', "rb").read())

#Find path to the image you want to detect face and pass it here
image = cv2.imread("/home/alperen/Downloads/faceRec/FaceDetect-master/images/imm.webp")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#convert image to Greyscale for haarcascade
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(60, 60),
    flags=cv2.CASCADE_SCALE_IMAGE
)
 
# the facial embeddings for face in input
encodings = face_recognition.face_encodings(rgb)
names = []
# loop over the facial embeddings incase
# we have multiple embeddings for multiple fcaes
for encoding in encodings:
    #Compare encodings with encodings in data["encodings"]
    #Matches contain array with boolean values and True for the embeddings it matches closely
    #and False for rest
    matches = face_recognition.compare_faces(data["encodings"],
    encoding)
    #set name = unknown if no encoding matches
    name = "Harry Styles"
    # check to see if we have found a match
    if True in matches:
        # update the list of names
        names.append(name)

        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()