from imutils import paths
import face_recognition
import numpy as np
import pickle
import cv2
import os
import time


path = 'img'

images = []

userinput="TRUMP"


classNames = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
	curimg = cv2.imread(f'{path}/{cl}')
	images.append(curimg)
	classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
	encodeList=[]
	for img in images:
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		encode = face_recognition.face_encodings(img)[0]
		encodeList.append(encode)
	
	return encodeList
	
encodeListKnown = findEncodings(images)
print('Encoding Complete')

start = time.process_time()
imgS = face_recognition.load_image_file('Test/Test9.jpg')
imgS = cv2.resize(imgS,(0,0),None,1,1)
imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
faceLoc_test = face_recognition.face_locations(imgS)
encodecurr = face_recognition.face_encodings(imgS,faceLoc_test)
color=(255,0,0)
stroke=1
thickness=3
for i in range(len(faceLoc_test)):
	cv2.rectangle(imgS,(faceLoc_test[i][3],faceLoc_test[i][0]),(faceLoc_test[i][1] ,faceLoc_test[i][2]),color,2)


for encodeFace,faceloc in zip(encodecurr, faceLoc_test):
	matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
	faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
	matchIndex=np.argmin(faceDis)
	print(matches)
	print(faceloc)
	print(faceDis)
	
	
	
	if matches[matchIndex]:
		name = classNames[matchIndex].upper()
		print(name)
		
		cv2.putText(imgS,f'{name}',(faceloc[3],faceloc[0]-10),cv2.FONT_HERSHEY_COMPLEX,0.4,color,1)
		
		if name == userinput:
			h = faceloc[2] - faceloc[0]  #y1-y2
			w = faceloc[1] - faceloc[3]  #x1-x0
			
			sub_face = imgS[faceloc[0]:faceloc[0] + h ,faceloc[3]:faceloc[3] + w]
			
			sub_face = cv2.blur(sub_face,(10, 10))
			imgS[faceloc[0]:faceloc[0]+sub_face.shape[0], faceloc[3]:faceloc[3]+sub_face.shape[1]] = sub_face
			
			
			
			



print(time.process_time() - start)
cv2.imwrite('images/output.jpg',imgS)
cv2.imshow('Photo',imgS)

cv2.waitKey(0)
#cv2.rectangle(imgTrump_test,(faceLoc_test[3],faceLoc_test[0]),(faceLoc_test[1] ,faceLoc_test[2]),color,2)










