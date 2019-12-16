import cv2

#This is just to load the haarcascade classifier(its a classifier to detect perticular objects from the source).
detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#if using web cam give 0
imp_img = cv2.VideoCapture(0)

#read method returns 2 values, first is true or false whether it has read the image or not and second is the dimensions of img.
res, img = imp_img.read()

#convert to greyscale because the haarcascade is designed for grey scale img.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# here 1.3 is scale factor and 5 is the min neighbors, will get 4 coordinates x,y,w,h.
faces = detect.detectMultiScale(gray ,1.3 , 5)

#rectangle is a cv2 method,first argument is the image, 3rd is the color of square or rectangle and 4th is the width of border.
for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)

#to show img, first is the title of image.
cv2.imshow('trial img',img)
#waitkey is to keep the image for the amount of time desired, 0 means a stable img and will close only when you want.
cv2.waitKey(0)

imp_img.release()

cv2.destroyAllWindows()







