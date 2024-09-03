import cv2
import imutils
import numpy as np
import pytesseract
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import re

picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

model = YOLO('best.pt')
with open("coco1.txt", "r") as file:
    class_list = file.read().split("\n")

license_plate_count = 0
object_count = 0
stair_count = 0

while True:
    image = picam2.capture_array()
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    results = model.predict(image)
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

        if c.lower() == "license_plate":
            license_plate_count += 1

            if license_plate_count >= 5:
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

                for c in cnts:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                    if len(approx) == 4:
                        screenCnt = approx
                        break
                else:
                    screenCnt = None

                if screenCnt is not None:
                    screenCnt += [x1, y1]
                    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
                    mask = np.zeros(gray.shape, np.uint8)
                    new_image = cv2.drawContours(mask, [screenCnt - [x1, y1]], 0, 255, -1)
                    new_image = cv2.bitwise_and(license_plate, license_plate, mask=mask)
                    (x, y) = np.where(mask == 255)
                    (topx, topy) = (np.min(x), np.min(y))
                    (bottomx, bottomy) = (np.max(x), np.max(y))
                    Cropped = gray[topx:bottomx+1, topy:bottomy+1]

                    custom_config = r'--oem 3 --psm 6'
                    text = pytesseract.image_to_string(Cropped, config=custom_config, lang='eng')
                    numbers = re.sub(r'[^A-Za-z0-9]', '', text)
                    print("Detected Numbers:", numbers)

                    cv2.imshow("Cropped", Cropped)
                    cv2.waitKey(0)

                license_count = 0
                break

    if key == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
