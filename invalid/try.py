import cv2
import numpy as np

for n in range(1, 157):
    img = cv2.imread('invalidxy'+str(n)+'.jpg', 0)

    _, contours, _ = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

    cnt = contours[0]
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    area = cv2.isContourConvex(cnt)

    count = 0
    for i in range(15, 25):
        for j in range(15, 25):
            if img[i][j] >= 250:
                count += 1

    print count

