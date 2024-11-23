#Blade Nelson
#Written in August 2019
import cv2
import numpy as np


import os

# os.putenv('SDL_VIDEODRIVER', 'fbcon')  # frame cache

# os.putenv('SDL_VIDEODRV','fbcon')
# os.putenv('SDL_FBDEV', '/dev/fb0')
# os.putenv('SDL_MOUSEDRV','dummy')
# os.putenv('SDL_MOUSEDEV','/dev/null')
# os.putenv('DISPLAY','')

# os.environ["DISPLAY"] = ":0"

eyeData = "xml/eyes.xml"
faceData = "xml/face.xml"
DOWNSCALE = 3

#Bools for control
add_face_rect = True
add_objects = True
add_eye_rect = True

#OpenCV boiler plate
webcam = cv2.VideoCapture('./face2.mp4')
# cv2.namedWindow("Webcam Facial Tracking")
# cv2.namedWindow("Webcam Glasses Tracking", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Webcam Glasses Tracking", 320, 240)
classifier = cv2.CascadeClassifier(eyeData)
faceClass = cv2.CascadeClassifier(faceData)

#Loading glasses asset
glasses = cv2.imread('assets/glasses.png', cv2.IMREAD_UNCHANGED)
glasses2 = cv2.imread('assets/glasses2.png', cv2.IMREAD_UNCHANGED)

mode_g = 0

ratio = glasses.shape[1] / glasses.shape[0]

if webcam.isOpened(): # try to get the first frame
    rval, frame = webcam.read()
else:
    rval = False
    
# Initialize VideoWriter for output
frame_width = 320
frame_height = 240
output_fps = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, output_fps, (frame_width, frame_height))

#Main loop
while rval:
    # detect eyes and draw glasses
    minisize = (int(frame.shape[1]/DOWNSCALE),int(frame.shape[0]/DOWNSCALE))
    miniframe = cv2.resize(frame, minisize)
    faces = faceClass.detectMultiScale(miniframe)
    eyes = classifier.detectMultiScale(miniframe)
    
    if add_eye_rect:
        for eye in eyes:
            x, y, w, h = [v * DOWNSCALE for v in eye]

            pts1 = (x, y+h)
            pts2 = (x + w, y)
            # pts1 and pts2 are the upper left and bottom right coordinates of the rectangle
            cv2.rectangle(frame, pts1, pts2, color=(0, 255, 0), thickness=3)

            if add_objects:
                h = w / ratio
                y += h / 2

                y = int(y)
                x = int(x)
                w = int(w)
                h = int(h)
                # resize glasses to a new var called small glasses
                if mode_g == 0:
                    smallglasses = cv2.resize(glasses, (int(w), int(h)))
                else:
                    smallglasses = cv2.resize(glasses2, (int(w), int(h)))
                # the area you want to change
                bg = frame[y:y+h, x:x+w]
                np.multiply(bg, np.atleast_3d(255 - smallglasses[:, :, 3])/255.0, out=bg, casting="unsafe")
                np.add(bg, smallglasses[:, :, 0:3] * np.atleast_3d(smallglasses[:, :, 3]), out=bg)
                # put the changed image back into the scene
                frame[y:y+h, x:x+w] = bg

    if add_face_rect:
        for face in faces:
            x, y, w, h = [v * DOWNSCALE for v in face]

            pts1 = (x, y+h)
            pts2 = (x + w, y)
            # pts1 and pts2 are the upper left and bottom right coordinates of the rectangle
            cv2.rectangle(frame, pts1, pts2, color=(255, 0, 0), thickness=3)

    # Write the processed frame to the output file
    out.write(frame)

    # cv2.imshow("Webcam Glasses Tracking", frame)
    # cv2.imwrite("a.mp4", frame)

    # get next frame
    rval, frame = webcam.read()

    # key = cv2.waitKey(20)
    # key = ord('1')
    # if key in [27, ord('Q'), ord('q')]: # exit on ESC
    #     cv2.destroyWindow("Webcam Face Tracking")
    #     break

    #Keyboard input
    
    # if key == ord('1'):
    #     if add_face_rect:
    #         add_face_rect = False
    #     else:
    #         add_face_rect = True

    # if key == ord('2'):
    #     if add_eye_rect:
    #         add_eye_rect = False
    #     else:
    #         add_eye_rect = True

    # if key == ord('3'):
    #     if add_objects:
    #         add_objects = False
    #     else:
    #         add_objects = True
    
    # if key == ord('4'):
    #     if mode_g == 0:
    #         mode_g = 1
    #     elif mode_g == 1:
    #         mode_g = 0
            
# Release resources
webcam.release()
out.release()
cv2.destroyAllWindows()
