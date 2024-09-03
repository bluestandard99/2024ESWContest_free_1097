import serial
import time
import cv2
import imutils
import numpy as np
import pytesseract
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import re
import os

# Initialize the serial connections to the 
dcmotor = serial.Serial('/dev/ttyACM1')
servor = serial.Serial('/dev/ttyACM0')



# Initialize the camera
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Load the YOLO model and class list
model = YOLO('best.pt')
with open("coco1.txt", "r") as file:
    class_list = file.read().split("\n")

license_count = 0
object_count = 0
stair_count = 0

def get_fixed_distance():
    return 25  # Return a fixed distance of 25 cm

frame_skip = 3  # Process every 3th frame
frame_count = 0


time.sleep(3)
servor.write(b'a')
time.sleep(2)
dcmotor.write(b'z')



while True:
    # Use the fixed distance
    distance = get_fixed_distance()

    image = picam2.capture_array()
    
    key = cv2.waitKey(1) & 0xFF

    # Process every 5th frame
    if frame_count % frame_skip == 0:
        # Resize image to speed up processing
        image_resized = cv2.resize(image, (640, 480))

        results = model.predict(image_resized)
        
        if len(results) > 0 and len(results[0].boxes.data) > 0:
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")

            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                if d >= len(class_list):
                    continue
                c = class_list[d]
                cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image_resized, c, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if c.lower() == "license_plate":
                    license_count += 1

                    if license_count >= 3:
                        offset = 10
                        y1 = y1 + offset
                        y2 = y2 + offset

                        license_plate = image[y1:y2, x1:x2]

                        gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                        gray = cv2.bilateralFilter(gray, 11, 17, 17)
                        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        edged = cv2.Canny(gray, 30, 200)

                        cv2.imshow("License Plate", license_plate)
                        cv2.imshow("Gray", gray)
                        cv2.imshow("Edged", edged)

                        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = imutils.grab_contours(cnts)
                        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

                        screenCnt = None
                        for c in cnts:
                            peri = cv2.arcLength(c, True)
                            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                            if len(approx) == 4:
                                screenCnt = approx
                                break

                        if screenCnt is not None:
                            screenCnt += [x1, y1]
                            cv2.drawContours(image_resized, [screenCnt], -1, (0, 255, 0), 3)
                            mask = np.zeros(gray.shape, np.uint8)
                            new_image = cv2.drawContours(mask, [screenCnt - [x1, y1]], 0, 255, -1)
                            new_image = cv2.bitwise_and(license_plate, license_plate, mask=mask)
                            (x, y) = np.where(mask == 255)
                            (topx, topy) = (np.min(x), np.min(y))
                            (bottomx, bottomy) = (np.max(x), np.max(y))
                            Cropped = gray[topx:bottomx+1, topy:bottomy+1]

                            custom_config = r'--oem 3 --psm 6'
                            text = pytesseract.image_to_string(Cropped, config=custom_config, lang='eng+kor')
                            numbers = re.sub(r'[^A-Za-z0-9]', '', text)
                            print("Detected Numbers:", numbers)

                            # Insert spaces between each character for espeak to spell them out
                            spelled_out = ' '.join(numbers)
                            print("Spelled Out:", spelled_out)
                            

                            # Convert the detected numbers to speech using espeak with adjusted speed
                            os.system(f'espeak -s 130 -a 200 "{spelled_out}"')
                            

                            cv2.imshow("Cropped", Cropped)
                            cv2.waitKey(0)

                        license_count = 0
                        break

                if c.lower() == "object":
                    object_count += 1

                    # Use the fixed distance
                    distance = get_fixed_distance()
                    
                    if distance <= 30 and object_count >= 5:
                        # Trigger voice output
                        
                                                
                        os.system('espeak -s 130 -v f5  "be careful  ...    obstacle ... obstacle ... obstacle "')
                        time.sleep(3)
                        servor.write(b'b')
                        time.sleep(3)
                        dcmotor.write(b'e')
                        
                        
                        
                        # Send perform motion command to Arduino
                        object_count = 0  # Reset count after notification

                if c.lower() == "stair":
                    stair_count += 1

                    
                    distance = get_fixed_distance()
                    

                    if stair_count >= 5 and distance <= 30:
                        # Trigger voice output
                        
                                               

                        os.system('espeak -s 130 -a 200 "becareful ... stair ... stair .... Will start climbing mode"')
                        time.sleep(3)
                        servor.write(b'c')
                        time.sleep(2)
                        servor.write(b'd')
                        time.sleep(2)
                        servor.write(b'c')
                        time.sleep(2)
                        servor.write(b'b')
                        time.sleep(2)
                        servor.write(b'a')
                        time.sleep(3)
                        dcmotor.write(b'e')
                        time.sleep(2)
                        dcmotor.write(b'e')
                        time.sleep(2)
                        dcmotor.write(b'e')
                        
                        
                        stair_count = 0  # Reset count after notification

    # Update the frame with annotations
    if frame_count % frame_skip == 0:
        cv2.imshow("Frame", image_resized)

    frame_count += 1

    if key == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()