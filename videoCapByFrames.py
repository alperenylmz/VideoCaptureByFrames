# Importing all necessary libraries
import cv2
import os

# Read the video from specified path
cam = cv2.VideoCapture('/home/alperen/Downloads/Harry Styles - As It Was (Official Video) [www.downloader.wiki].mp4')

try:
	
	# creating a folder named data
	if not os.path.exists('data'):
		os.makedirs('data')

# if not created then raise error
except OSError:
	print ('Error: Creating directory of data')

# frame
currentframe = 500
second = 1.2
i = 0
fps = 24
totalFrame=int(second*fps)

for i in range (totalFrame):
    # reading from frame
    cam.set(1, i)
    ret,frame = cam.read()
    
		# if video is still left continue creating images
    name = '/home/alperen/Downloads/videoCapImages/' + str(currentframe) + '.jpg'
    print('Creating...' + name)
    
		# writing the extracted images
    cv2.imwrite(name, frame)

		# increasing counter so that it will
		# show how many frames are create
    currentframe += 1

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
