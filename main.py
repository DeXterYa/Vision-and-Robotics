#!/usr/bin/env python2.7
import gym
import reacher3D.Reacher
import numpy as np
import cv2
import math
import scipy as sp
import collections
import time
import lecture_2_5 as l25
import lecture_3_2 as l32
import lecture_4_2 as l42
import lecture_5_1 as SVM

class MainReacher():
    def __init__(self):
        self.env = gym.make('3DReacherMy-v0')
        self.env.reset()
        self.classes = [['valid', 308], ['invalid', 312]]
        self.test_classifier = SVM.classifer(self.classes)
        self.test_classifier.train('SVM3.xml', True)

    def peak_pick_target(self, image):
        img_hist = cv2.calcHist([image], [0], None, [256], [0, 256])

        # Find the largest peak in histogram
        peak = np.argmax(img_hist)

        # Find the largest peak on the darker side
        img_hist_darker = img_hist[0:peak - 1]
        threshold = 0
        for n in range(peak - 2, 0 , -1):
            if img_hist_darker[n] != 0:
                threshold = n
                break

        return threshold

    def peak_pick_rgb(self, image):
        img_hist = cv2.calcHist([image], [0], None, [256], [0, 256])

        # Find the largest peak in histogram
        peak = np.argmax(img_hist)

        return peak
    def peak_pick_darkblue(self, image):
        img_hist = cv2.calcHist([image], [0], None, [256], [0, 256])

        # Find the largest peak in histogram
        peak = np.argmax(img_hist)

        # Find the largest peak on the darker side
        img_hist_darker = img_hist[0:peak - 1]
        threshold = 0
        count = 0
        for n in range(peak - 2, 0 , -1):
            if img_hist_darker[n] != 0:
                count += 1
                if count == 2:
                    threshold = n
                    break

        return threshold

    def detect_red(self, image):

        #peak = self.peak_pick_rgb(image[:,:,0])

        mask = cv2.inRange(image, (image[0][0][0]-0.1, 0, 0), (image[0][0][0]+0.1, 0, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return self.coordinate_convert(np.array([cx, cy]))

    def detect_green(self, image):
        #peak = self.peak_pick_rgb(image[:, :, 1])

        mask = cv2.inRange(image, (0, image[0][0][0]-0.1, 0), (0, image[0][0][0]+0.1, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return self.coordinate_convert(np.array([cx, cy]))

    def detect_blue(self, image):
        #peak = self.peak_pick_rgb(image[:, :, 2])

        mask = cv2.inRange(image, (0, 0, image[0][0][0]-0.1), (0, 0, image[0][0][0]+0.1))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return self.coordinate_convert(np.array([cx, cy]))

    def detect_darkblue(self, image):
        #peak = self.peak_pick_darkblue(image[:, :, 2])

        mask = cv2.inRange(image, (0, 0, 127.5*image[0][0][0]/255-0.1), (0, 0, 127.5*image[0][0][0]/255+0.1))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return self.coordinate_convert(np.array([cx, cy]))

    # 7 links colour detection
    def detect_darkgreen(self, image):


        mask = cv2.inRange(image, (0, 127.5 * image[0][0][0] / 255 - 0.1, 0),
                           (0, 127.5 * image[0][0][0] / 255 + 0.1, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return self.coordinate_convert(np.array([cx, cy]))

    def detect_darkred(self, image):


        mask = cv2.inRange(image, (127.5 * image[0][0][0] / 255 - 0.1, 0, 0),
                           (127.5 * image[0][0][0] / 255 + 0.1, 0, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return self.coordinate_convert(np.array([cx, cy]))

    def detect_purple(self, image):


        mask = cv2.inRange(image, (127.5 * image[0][0][0] / 255 - 0.1, 0, 127.5 * image[0][0][0] / 255 - 0.1),
                           (127.5 * image[0][0][0] / 255 + 0.1, 0, 127.5 * image[0][0][0] / 255 + 0.1))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        M = cv2.moments(mask)

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])


        return self.coordinate_convert(np.array([cx, cy]))

    # 7 links colout detection

    def detect_valid(self, image):
        # In this method you can focus on detecting the center of the blue circle
        # Isolate the blue colour in the image as a binary image
        r, g, b = cv2.split(image)
        # cv2.imwrite(filename="inhenced0.jpg", img= HSV2)
        (minVal, maxValg, minLoc, maxLoc) = cv2.minMaxLoc(g)
        (minVal, maxValb, minLoc, maxLoc) = cv2.minMaxLoc(b)
        (minVal, maxValr, minLoc, maxLoc) = cv2.minMaxLoc(r)


        lower = 170.0
        upper = 180.0
        #threshold = self.peak_pick_target(image[:,:,0])
        # mask = cv2.inRange(image,
        #                    (threshold-1, threshold-1, threshold-1),
        #                    (threshold+1, threshold+1, threshold+1))
        mask = cv2.inRange(image, (lower * image[0][0][0] / 255, lower * image[0][0][0] / 255, lower * image[0][0][0] / 255),
                           (upper * image[0][0][0] /255, upper * image[0][0][0] /255, upper * image[0][0][0] / 255))

        mask_origin = mask.copy()

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        cv2.imwrite(filename="(in)valid.jpg", img=mask)

        img = mask
        a, contours, b = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

        centres = []

        for i in range(len(contours)):
            moments = cv2.moments(contours[i])
            centres.append((int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])))

        if len(centres) >= 2:
            count1 = 0
            count2 = 0

            target_1 = mask_origin[(centres[0][1] - 5):(centres[0][1] + 5), (centres[0][0] - 5):(centres[0][0] + 5)]
            for i in range(0, 10):
                for j in range(0, 10):
                    if target_1[i][j] >= 250:
                        count1 += 1

            target_1 = mask_origin[(centres[1][1] - 5):(centres[1][1] + 5), (centres[1][0] - 5):(centres[1][0] + 5)]
            for i in range(0, 10):
                for j in range(0, 10):
                    if target_1[i][j] >= 250:
                        count2 += 1

            if count1 <= 96:
                moments = cv2.moments(contours[0])
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])

                return self.coordinate_convert(np.array([cx, cy]))
            else:
                moments = cv2.moments(contours[1])
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])

                return self.coordinate_convert(np.array([cx, cy]))

        else:
            return  self.coordinate_convert(np.array([0, 0]))

    def detect_valid_SVM(self, image):
        r, g, b = cv2.split(image)
        (minVal, maxValg, minLoc, maxLoc) = cv2.minMaxLoc(g)
        (minVal, maxValb, minLoc, maxLoc) = cv2.minMaxLoc(b)
        (minVal, maxValr, minLoc, maxLoc) = cv2.minMaxLoc(r)
        lower = 160
        upper = 197
        mask = cv2.inRange(image, (lower * maxValr / 255, lower * maxValg / 255, lower * maxValb / 255),
                           (upper * maxValr, upper * maxValg, upper * maxValb / 255))
        mask_origin = mask.copy()
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        a, contours, b = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        centres = []

        for i in range(len(contours)):
            moments = cv2.moments(contours[i])
            centres.append((int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])))
        for i in range(len(centres)):
            crop_img = mask_origin[centres[i][1] - 20:centres[i][1] + 20, centres[i][0] - 20:centres[i][0] + 20]
            prediction = self.test_classifier.img_classify(crop_img)[1][0]
            if prediction == 0:
                found = True
                return self.coordinate_convert(np.array([centres[i][0], centres[i][1]]))



    def detect_joint_angles(self, rgbev):
        jointPos4 = [rgbev[3][0], rgbev[3][2]]
        jointpos43 = [rgbev[3][0], rgbev[3][1]]
        jointPos34 = [rgbev[2][0], rgbev[2][2]]
        jointPos3 = [rgbev[2][0], rgbev[2][1]]
        jointPos2 = [rgbev[1][0], rgbev[1][1]]
        jointPos21 = [rgbev[1][0], rgbev[1][2]]

        jointPos1 = [rgbev[0][0], rgbev[0][2]]



        # Red point
        ja1 = math.atan2(jointPos1[1], jointPos1[0])

        # Green point
        x_green = jointPos2[0]
        y_green = jointPos2[1]
        z_green = jointPos21[1]

        green_vector = np.array([x_green, y_green, z_green])
        red_vector = np.array([jointPos1[0], 0, jointPos1[1]])
        theta_cos = np.inner(green_vector - red_vector, red_vector) / np.linalg.norm(green_vector - red_vector) / np.linalg.norm(red_vector)

        if theta_cos > 1:
            theta_cos = 1
        if theta_cos < -1:
            theta_cos = -1

        theta = math.acos(theta_cos)

        if y_green >= 0:
            ja2 = theta
        else:
            ja2 = -theta


        ja2 = self.angle_normalize(ja2)



        # Blue point

        x_blue = jointPos3[0]
        y_blue = jointPos3[1]
        z_blue = jointPos34[1]

        blue_vector = np.array([x_blue, y_blue, z_blue])

        theta_cos = np.inner(blue_vector - green_vector, green_vector - red_vector) / np.linalg.norm(
            blue_vector - green_vector) / np.linalg.norm(green_vector - red_vector)

        if theta_cos > 1:
            theta_cos = 1
        if theta_cos < -1:
            theta_cos = -1

        theta = math.acos(theta_cos)


        rotation1 = np.zeros((3,3))
        rotation1 = [[math.cos(ja2),-math.sin(ja2),0],[math.sin(ja2),math.cos(ja2),0],[0,0,1]]

        rotation_1 = np.zeros((3, 3))
        rotation_1 = [[math.cos(ja1), 0, -math.sin(ja1)], [0, 1, 0],
                      [math.sin(ja1), 0, math.cos(ja1)]]

        y_old = np.array([0, 1, 0])
        y_new = np.dot(np.dot(rotation_1, rotation1),y_old)

        judge = np.dot(y_new, blue_vector-green_vector)
        if judge > 0:
            ja3 = theta

        else:
            ja3 = -theta

        ja3 = self.angle_normalize(ja3)







        # Dark blue point

        x_darkblue = jointPos4[0]
        y_darkblue = jointpos43[1]
        z_darkblue = jointPos4[1]

        darkblue_vector = np.array([x_darkblue, y_darkblue, z_darkblue])

        theta_cos = np.inner(darkblue_vector - blue_vector, blue_vector - green_vector) / np.linalg.norm(
            darkblue_vector - blue_vector) / np.linalg.norm(blue_vector - green_vector)
        if theta_cos > 1:
            theta_cos = 1
        if theta_cos < -1:
            theta_cos = -1


        theta = math.acos(theta_cos)

        rotation2 = np.zeros((3, 3))
        rotation2 = [[math.cos(ja1), 0, -math.sin(ja1)], [0, 1, 0], [math.sin(ja1), 0, math.cos(ja1)]]

        z_old = np.array([0, 0, 1])
        z_new = np.dot(rotation2, z_old)
        judge = np.dot(z_new, darkblue_vector - blue_vector)

        if judge > 0:
            ja4 = theta

        else:
            ja4 = -theta

        ja4 = self.angle_normalize(ja4)

        return np.array([ja1, ja2, ja3, ja4])


    # 7 link
    def detect_joint_angles_7(self, arrxy, arrxz):
        red_xy = self.detect_red(arrxy)
        red_xz = self.detect_red(arrxz)
        red_pos = np.array([red_xy[0], red_xy[1], red_xz[1]])

        green_xy = self.detect_green(arrxy)
        green_xz = self.detect_green(arrxz)
        green_pos = np.array([green_xy[0], green_xy[1], green_xz[1]])

        blue_xy = self.detect_blue(arrxy)
        blue_xz = self.detect_blue(arrxz)
        blue_pos = np.array([blue_xy[0], blue_xy[1], blue_xz[1]])

        db_xy = self.detect_darkblue(arrxy)
        db_xz = self.detect_darkblue(arrxz)
        db_pos = np.array([db_xy[0], db_xy[1], db_xz[1]])

        dg_xy = self.detect_darkgreen(arrxy)
        dg_xz = self.detect_darkgreen(arrxz)
        dg_pos = np.array([dg_xy[0], dg_xy[1], dg_xz[1]])

        dr_xy = self.detect_darkred(arrxy)
        dr_xz = self.detect_darkred(arrxz)
        dr_pos = np.array([dr_xy[0], dr_xy[1], dr_xz[1]])

        pur_xy = self.detect_purple(arrxy)
        pur_xz = self.detect_purple(arrxz)
        pur_pos = np.array([pur_xy[0], pur_xy[1], pur_xz[1]])

        # Calculate the angles
        ja1 = math.atan2(red_pos[2], red_pos[0])

        theta_cos = np.inner(green_pos - red_pos, red_pos) / np.linalg.norm(
            green_pos - red_pos) / np.linalg.norm(red_pos)

        if theta_cos > 1:
            theta_cos = 1
        if theta_cos < -1:
            theta_cos = -1

        theta = math.acos(theta_cos)

        if green_pos[1] >= 0:
            ja2 = theta
        else:
            ja2 = -theta


        ja2 = self.angle_normalize(ja2)



        theta_cos = np.inner(blue_pos - green_pos, green_pos - red_pos) / np.linalg.norm(
            blue_pos - green_pos) / np.linalg.norm(green_pos - red_pos)

        if theta_cos > 1:
            theta_cos = 1
        if theta_cos < -1:
            theta_cos = -1

        theta = math.acos(theta_cos)

        rotation2 = np.zeros((3, 3))
        rotation2 = [[math.cos(ja2), -math.sin(ja2), 0], [math.sin(ja2), math.cos(ja2), 0], [0, 0, 1]]

        rotation1 = np.zeros((3, 3))
        rotation1 = [[math.cos(ja1), 0, -math.sin(ja1)], [0, 1, 0],
                      [math.sin(ja1), 0, math.cos(ja1)]]

        y_old = np.array([0, 1, 0])
        y_new = np.dot(np.dot(rotation1, rotation2), y_old)

        judge = np.dot(y_new, blue_pos - green_pos)
        if judge > 0:
            ja3 = theta

        else:
            ja3 = -theta

        ja3 = self.angle_normalize(ja3)



        theta_cos = np.inner(db_pos - blue_pos, blue_pos - green_pos) / np.linalg.norm(
            db_pos - blue_pos) / np.linalg.norm(blue_pos - green_pos)

        if theta_cos > 1:
            theta_cos = 1
        if theta_cos < -1:
            theta_cos = -1

        theta = math.acos(theta_cos)

        rotation3 = np.zeros((3, 3))
        rotation3 = [[math.cos(ja3), -math.sin(ja3), 0], [math.sin(ja3), math.cos(ja3), 0], [0, 0, 1]]



        z_old = np.array([0, 0, 1])
        z_new = np.dot(np.dot(np.dot(rotation1, rotation2), rotation3),z_old)

        judge = np.dot(z_new, db_pos - blue_pos)
        if judge > 0:
            ja4 = theta

        else:
            ja4 = -theta

        ja4 = self.angle_normalize(ja4)


        #ja5

        theta_cos = np.inner(dg_pos - db_pos, db_pos - blue_pos) / np.linalg.norm(
            dg_pos - db_pos) / np.linalg.norm(db_pos - blue_pos)

        if theta_cos > 1:
            theta_cos = 1
        if theta_cos < -1:
            theta_cos = -1

        theta = math.acos(theta_cos)

        rotation4 = np.zeros((3, 3))
        rotation4 = [[math.cos(ja4), 0, -math.sin(ja4)], [0, 1, 0],
                     [math.sin(ja4), 0, math.cos(ja4)]]
        z_old = np.array([0, 0, 1])
        z_new = np.dot(np.dot(np.dot(np.dot(rotation1, rotation2), rotation3), rotation4),z_old)

        judge = np.dot(z_new, dg_pos - db_pos)
        if judge > 0:
            ja5 = theta

        else:
            ja5 = -theta

        ja5 = self.angle_normalize(ja5)


        #ja6
        theta_cos = np.inner(dr_pos - dg_pos, dg_pos - db_pos) / np.linalg.norm(
            dr_pos - dg_pos) / np.linalg.norm(dg_pos - db_pos)

        if theta_cos > 1:
            theta_cos = 1
        if theta_cos < -1:
            theta_cos = -1

        theta = math.acos(theta_cos)

        rotation5 = np.zeros((3, 3))
        rotation5 = [[math.cos(ja5), 0, -math.sin(ja5)], [0, 1, 0],
                     [math.sin(ja5), 0, math.cos(ja5)]]
        y_old = np.array([0, 1, 0])
        y_new = np.dot(np.dot(np.dot(np.dot(np.dot(rotation1, rotation2), rotation3), rotation4), rotation5), y_old)

        judge = np.dot(y_new, dr_pos - dg_pos)
        if judge > 0:
            ja6 = theta

        else:
            ja6 = -theta

        ja6 = self.angle_normalize(ja6)

        # ja7
        theta_cos = np.inner(pur_pos - dr_pos, dr_pos - dg_pos) / np.linalg.norm(
            pur_pos - dr_pos) / np.linalg.norm(dr_pos - dg_pos)

        if theta_cos > 1:
            theta_cos = 1
        if theta_cos < -1:
            theta_cos = -1

        theta = math.acos(theta_cos)

        rotation6 = np.zeros((3, 3))
        rotation6 = [[math.cos(ja6), -math.sin(ja6), 0], [math.sin(ja6), math.cos(ja6), 0], [0, 0, 1]]

        z_old = np.array([0, 0, 1])
        z_new = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(rotation1, rotation2), rotation3), rotation4), rotation5), rotation6), z_old)

        judge = np.dot(z_new, pur_pos - dr_pos)
        if judge > 0:
            ja7 = theta

        else:
            ja7 = -theta

        ja7 = self.angle_normalize(ja7)

        return np.array([ja1, ja2, ja3, ja4, ja5, ja6, ja7])

    # 7 link



    def angle_normalize(self, angle):
        return (((angle + np.pi) % (2 * np.pi)) - np.pi)


    def Jacobian(self, rgbev, angles):

        rx, ry, rz = rgbev[0]
        gx, gy, gz = rgbev[1]
        bx, by, bz = rgbev[2]
        ex, ey, ez = rgbev[3]


        jacobian = np.zeros((6, 4))
        z_vector = np.array([0, 0, 1])
        y_vector = np.array([0, -1, 0])

        j1_pos = np.zeros((3, 1))
        j2_pos = np.array([rx, ry, rz]).reshape(3, 1)
        j3_pos = np.array([gx, gy, gz]).reshape(3, 1)
        j4_pos = np.array([bx, by, bz]).reshape(3, 1)
        ee_pos = np.array([ex, ey, ez]).reshape(3, 1)


        pos_3D = np.zeros(3)
        pos_3D[0: 3] = (ee_pos - j1_pos).T
        jacobian[0:3, 0] = np.cross(y_vector, pos_3D)
        jacobian[3:6, 0] = y_vector

        z_vector = np.array([-math.sin(angles[0]), 0, math.cos(angles[0])])


        pos_3D[0: 3] = (ee_pos - j2_pos).T
        jacobian[0:3, 1] = np.cross(z_vector, pos_3D)
        jacobian[3:6, 1] = z_vector



        pos_3D[0: 3] = (ee_pos - j3_pos).T
        jacobian[0:3, 2] = np.cross(z_vector, pos_3D)
        jacobian[3:6, 2] = z_vector

        rotation_1 = np.zeros((3, 3))
        rotation_1 = [[math.cos(angles[0]), 0, -math.sin(angles[0])], [0, 1, 0], [math.sin(angles[0]), 0, math.cos(angles[0])]]

        rotation_2 = np.zeros((3, 3))
        rotation_2 = [[math.cos(angles[1]), -math.sin(angles[1]), 0], [math.sin(angles[1]), math.cos(angles[1]), 0],
                      [0, 0, 1]]

        rotation_3 = np.zeros((3, 3))
        rotation_3 = [[math.cos(angles[2]), -math.sin(angles[2]), 0], [math.sin(angles[2]), math.cos(angles[2]), 0],
                      [0, 0, 1]]

        rotation_all = -np.dot(np.dot(rotation_1, rotation_2), rotation_3)

        y_vector = rotation_all[:, 1].reshape(1, 3)




        pos_3D[0: 3] = (ee_pos - j4_pos).T

        jacobian[0:3, 3] = np.cross(y_vector, pos_3D)
        jacobian[3:6, 3] = y_vector

        return jacobian

    # 7 links
    def Jacobian_7(self, arrxy, arrxz, angles):

        j1_pos = np.zeros((3, 1))

        red_xy = self.detect_red(arrxy)
        red_xz = self.detect_red(arrxz)
        j2_pos = np.array([red_xy[0], red_xy[1], red_xz[1]]).reshape(3, 1)

        green_xy = self.detect_green(arrxy)
        green_xz = self.detect_green(arrxz)
        j3_pos = np.array([green_xy[0], green_xy[1], green_xz[1]]).reshape(3, 1)

        blue_xy = self.detect_blue(arrxy)
        blue_xz = self.detect_blue(arrxz)
        j4_pos = np.array([blue_xy[0], blue_xy[1], blue_xz[1]]).reshape(3, 1)

        db_xy = self.detect_darkblue(arrxy)
        db_xz = self.detect_darkblue(arrxz)
        j5_pos = np.array([db_xy[0], db_xy[1], db_xz[1]]).reshape(3, 1)

        dg_xy = self.detect_darkgreen(arrxy)
        dg_xz = self.detect_darkgreen(arrxz)
        j6_pos = np.array([dg_xy[0], dg_xy[1], dg_xz[1]]).reshape(3, 1)

        dr_xy = self.detect_darkred(arrxy)
        dr_xz = self.detect_darkred(arrxz)
        j7_pos = np.array([dr_xy[0], dr_xy[1], dr_xz[1]]).reshape(3, 1)

        pur_xy = self.detect_purple(arrxy)
        pur_xz = self.detect_purple(arrxz)
        ee_pos = np.array([pur_xy[0], pur_xy[1], pur_xz[1]]).reshape(3, 1)

        jacobian = np.zeros((3, 7))
        z_vector = np.array([0, 0, 1])
        y_vector = np.array([0, -1, 0])

        pos_3D = np.zeros(3)
        pos_3D[0: 3] = (ee_pos - j1_pos).T
        jacobian[0:3, 0] = np.cross(y_vector, pos_3D)

        z_vector = np.array([-math.sin(angles[0]), 0, math.cos(angles[0])])

        pos_3D[0: 3] = (ee_pos - j2_pos).T
        jacobian[0:3, 1] = np.cross(z_vector, pos_3D)


        pos_3D[0: 3] = (ee_pos - j3_pos).T
        jacobian[0:3, 2] = np.cross(z_vector, pos_3D)


        rotation_1 = np.zeros((3, 3))
        rotation_1 = [[math.cos(angles[0]), 0, -math.sin(angles[0])], [0, 1, 0],
                      [math.sin(angles[0]), 0, math.cos(angles[0])]]

        rotation_2 = np.zeros((3, 3))
        rotation_2 = [[math.cos(angles[1]), -math.sin(angles[1]), 0], [math.sin(angles[1]), math.cos(angles[1]), 0],
                      [0, 0, 1]]

        rotation_3 = np.zeros((3, 3))
        rotation_3 = [[math.cos(angles[2]), -math.sin(angles[2]), 0], [math.sin(angles[2]), math.cos(angles[2]), 0],
                      [0, 0, 1]]

        rotation_all = -np.dot(np.dot(rotation_1, rotation_2), rotation_3)

        y_vector = rotation_all[:, 1].reshape(1, 3)

        pos_3D[0: 3] = (ee_pos - j4_pos).T

        jacobian[0:3, 3] = np.cross(y_vector, pos_3D)

        rotation_4 = np.zeros((3, 3))
        rotation_4 = [[math.cos(angles[3]), 0, -math.sin(angles[3])], [0, 1, 0],
                     [math.sin(angles[3]), 0, math.cos(angles[3])]]
        rotation_all = np.dot(rotation_all, rotation_4)
        y_vector = rotation_all[:, 1].reshape(1, 3)
        pos_3D[0: 3] = (ee_pos - j5_pos).T

        jacobian[0:3, 4] = np.cross(y_vector, pos_3D)

        rotation_5 = np.zeros((3, 3))
        rotation_5 = [[math.cos(angles[4]), 0, -math.sin(angles[4])], [0, 1, 0],
                      [math.sin(angles[4]), 0, math.cos(angles[4])]]
        rotation_all = np.dot(rotation_all, rotation_5)
        z_vector = rotation_all[:, 2].reshape(1, 3)
        pos_3D[0: 3] = (ee_pos - j6_pos).T

        jacobian[0:3, 5] = np.cross(z_vector, pos_3D)

        rotation_6 = np.zeros((3, 3))
        rotation_6 = [[math.cos(angles[5]), -math.sin(angles[5]), 0], [math.sin(angles[5]), math.cos(angles[5]), 0],
                      [0, 0, 1]]
        rotation_all = np.dot(rotation_all, rotation_6)
        y_vector = rotation_all[:, 1].reshape(1, 3)
        pos_3D[0: 3] = (ee_pos - j7_pos).T

        jacobian[0:3, 6] = np.cross(y_vector, pos_3D)

        return jacobian

    # 7 links


    def IK(self, rgbev, angles):
        ee_pos = np.array(rgbev[3])

        desired_position = np.array(rgbev[4])



        pos_error = desired_position - ee_pos

        Jac = np.matrix(self.Jacobian(rgbev, angles), dtype='float')[0:3,:]



        if np.linalg.matrix_rank(Jac, 0.4) < 3:
            Jac_inv = Jac.T

        else:
            Jac_inv = Jac.T * np.linalg.inv(Jac*Jac.T)



        q_dot = Jac_inv*np.matrix(pos_error, dtype='float' ).T

        return np.squeeze(np.array(q_dot.T))

    # 7 links
    def IK_7(self, arrxy, arrxz, angles):

        pur_xy = self.detect_purple(arrxy)
        pur_xz = self.detect_purple(arrxz)
        ee_pos = np.array([pur_xy[0], pur_xy[1], pur_xz[1]])

        target_xy = self.detect_valid(arrxy)
        target_xz = self.detect_valid(arrxz)
        desired_position = np.array([target_xy[0], target_xy[1], target_xz[1]])


        pos_error = desired_position - ee_pos

        Jac = np.matrix(self.Jacobian_7(arrxy, arrxz, angles), dtype='float')



        if np.linalg.matrix_rank(Jac, 0.4) < 3:
            Jac_inv = Jac.T

        else:
            Jac_inv = Jac.T * np.linalg.inv(Jac*Jac.T)



        q_dot = Jac_inv*np.matrix(pos_error, dtype='float' ).T

        return np.squeeze(np.array(q_dot.T))
    # 7 links

    def ts_pd_control(self, curr_ee_pos, curr_ee_vel, desired_ee_pos):
        P = np.array([100, 100, 100])
        D = np.array([20, 20, 20])


        P_error = np.matrix(desired_ee_pos - curr_ee_pos).T
        D_error = np.zeros(shape=(3, 1)) - np.matrix(curr_ee_vel).T

        PD_error = np.diag(P)*P_error + np.diag(D)*D_error

        return PD_error


    def grav(self, angles):

        g = 9.81


        torque_1 = 7.0/2 * g *math.cos(angles[0]) + 5.0/2 * g * math.cos(angles[1]) * math.cos(angles[0]) + 3.0/2 * g * math.cos(angles[1] + angles[2]) * math.cos(angles[0])
        torque_1 += g * math.cos(angles[0]) * math.cos(angles[1]) * math.cos(angles[2]) * math.cos(angles[3]) / 2
        torque_1 -= g * math.cos(angles[0]) * math.sin(angles[1]) * math.sin(angles[2]) * math.cos(angles[3]) / 2
        torque_1 -= g * math.sin(angles[0]) * math.sin(angles[3]) / 2

        torque_2 = - 5.0/2 * g * math.sin(angles[1]) * math.sin(angles[0]) - 3.0/2 * g * math.sin(angles[1] + angles[2]) * math.sin(angles[0])
        torque_2 -= g * math.sin(angles[0]) * math.sin(angles[1]) * math.cos(angles[2]) * math.cos(angles[3]) / 2
        torque_2 -= g * math.sin(angles[0]) * math.cos(angles[1]) * math.sin(angles[2]) * math.cos(angles[3]) / 2

        torque_3 = - 3.0/2 * g * math.sin(angles[1] + angles[2]) * math.sin(angles[0])
        torque_3 -= g * math.sin(angles[0]) * math.cos(angles[1]) * math.sin(angles[2]) * math.cos(angles[3]) / 2
        torque_3 -= g * math.sin(angles[0]) * math.sin(angles[1]) * math.cos(angles[2]) * math.cos(angles[3]) / 2

        torque_4 =  g * math.cos(angles[3]) *math.cos(angles[0]) /2
        torque_4 -= g * math.sin(angles[0]) * math.cos(angles[1]) * math.cos(angles[2]) * math.sin(angles[3]) / 2
        torque_4 += g * math.sin(angles[0]) * math.sin(angles[1]) * math.sin(angles[2]) * math.sin(angles[3]) / 2



        return np.matrix([torque_1, torque_2, torque_3, torque_4]).T







    def coordinate_convert(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution])

    def go(self):
        #The robot has several simulated modes:
        #These modes are listed in the following format:
        #Identifier (control mode) : Description : Input structure into step function

        #POS : A joint space position control mode that allows you to set the desired joint angles and will position the robot to these angles : env.step((np.zeros(3),np.zeros(3), desired joint angles, np.zeros(3)))
        #POS-IMG : Same control as POS, however you must provide the current joint angles and velocities : env.step((estimated joint angles, estimated joint velocities, desired joint angles, np.zeros(3)))
        #VEL : A joint space velocity control, the inputs require the joint angle error and joint velocities : env.step((joint angle error (velocity), estimated joint velocities, np.zeros(3), np.zeros(3)))
        #TORQUE : Provides direct access to the torque control on the robot : env.step((np.zeros(3),np.zeros(3),np.zeros(3),desired joint torques))
        self.env.controlMode="VEL"
        #Run 100000 iterations
        prev_JAs = np.zeros(4)
        #prev_JAs = np.zeros(7)

        desiredJointAngles = np.array([-2*np.pi/3, np.pi/3, -np.pi/4, np.pi/4])



        # Uncomment to have gravity act in the z-axis

        prevEePos = np.zeros(shape=(1, 3))
        #self.env.world.setGravity((0, 0, -9.81))




        desiredJointAngles = np.array([-2*np.pi / 3, np.pi / 3, -np.pi / 4, np.pi / 3, np.pi / 3, np.pi / 3, np.pi / 3])
        for n in range(1000000):
            # The change in time between iterations can be found in the self.env.dt variable
            dt = self.env.dt
            # self.env.render returns 2 RGB arrays of the robot, one for the xy-plane, and one for the xz-plane
            arrxy,arrxz = self.env.render('rgb-array')

            rgbev = np.empty([5, 3])
            rgbev[0][0:2] = self.detect_red(arrxy)
            rgbev[0][2] = self.detect_red(arrxz)[1]
            rgbev[1][0:2] = self.detect_green(arrxy)
            rgbev[1][2] = self.detect_green(arrxz)[1]
            rgbev[2][0:2] = self.detect_blue(arrxy)
            rgbev[2][2] = self.detect_blue(arrxz)[1]
            rgbev[3][0:2] = self.detect_darkblue(arrxy)
            rgbev[3][2] = self.detect_darkblue(arrxz)[1]
            rgbev[4][0:2] = self.detect_valid_SVM(arrxy)
            rgbev[4][2] = self.detect_valid_SVM(arrxz)[1]


            # POS-IMG
            # angle = self.detect_joint_angles(rgbev)
            # detectedJointVels = self.angle_normalize(angle - prev_JAs) / dt
            # prev_JAs = angle
            # self.env.step((angle, detectedJointVels, desiredJointAngles, np.zeros(4)))
            # POS-IMG


            # Change the mode to VEL
            # VEL
            angle = self.detect_joint_angles(rgbev)
            detectedJointVels = self.angle_normalize(angle - prev_JAs) / dt
            jointAngles = self.IK(rgbev, angle)
            prev_JAs = angle
            self.env.step((jointAngles, detectedJointVels, np.zeros(4), np.zeros(4)))
            # VEL



            # This is the mode with gravity. If there is no gravity, please comment '+ grav_opposite_torques'
            #Change the mode to 'TORQUE'

            # TORQUE
            # angle = self.detect_joint_angles(rgbev)
            # ee_pos = np.array(rgbev[3])
            # ee_tar = np.array(rgbev[4])
            # ee_vel = (ee_pos - prevEePos) / dt
            # prevEePos = ee_pos
            # J = self.Jacobian(rgbev, angle)[0:3, :]
            # ee_desired_force = self.ts_pd_control(ee_pos, ee_vel, ee_tar)
            # grav_opposite_torques = self.grav(self.env.ground_truth_joint_angles)
            # torques = J.T * ee_desired_force + grav_opposite_torques
            # self.env.step((np.zeros(4), np.zeros(4), np.zeros(4), torques))
            # TORQUE





            # This part of code belongs to Open Chanllenge.
            # If you want to use, please rename the 'Reacher' to 'Reacher_4links' in file 'Reacher3D'
            # And rename the 'Reacher_7links to 'Reacher' in file 'Reacher3D'
            # And change 'prev_JAs = np.zeros(4)' to 'prev_JAs = np.zeros(7)'

            # 7 links

            # Change the mode to POS-IMG
            # POS-IMG
            # angle = self.detect_joint_angles_7(rgbev)
            # detectedJointVels = self.angle_normalize(angle - prev_JAs) / dt
            # prev_JAs = angle
            # self.env.step((angle, detectedJointVels, desiredJointAngles, np.zeros(4)))



            # Change the mode to VEL
            # VEL
            # angle = self.detect_joint_angles_7(arrxy, arrxz)
            # jointAngles = self.IK_7(arrxy, arrxz, angle)
            # detectedJointVels = self.angle_normalize(angle - prev_JAs) / dt
            # prev_JAs = angle
            # self.env.step((jointAngles, detectedJointVels, np.zeros(4), np.zeros(4)))

            # 7 links


#main method
def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()
Hi there call