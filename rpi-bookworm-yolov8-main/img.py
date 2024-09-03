import cv2
from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
cpt = 0
maxFrames = 50
while cpt < maxFrames:
    im= picam2.capture_array()
    im=cv2.flip(im,1)
    cv2.imshow("Camera", im)
    cv2.imwrite('/home/99jhhan/rpi-bookworm-yolov8-main/images/stair_%d.jpg' %cpt, im)
    cpt += 1
    if cv2.waitKey(1)==ord('q'):
        break
    time.sleep(0.3)
cv2.destroyAllWindows()