#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python2.7
import gym
import reacher3D.Reacher
import numpy as np
import cv2
import math
import scipy as sp
import collections
import time
import matplotlib.pyplot as plt
import lecture_2_5 as l25
import lecture_3_2 as l32
import lecture_4_2 as l42
import lecture_5_1 as SVM
import os
import sys


# In[ ]:


class MainReacher():
    
    def __init__(self):
        self.env = gym.make('3DReacherMy-v0')
        self.env.reset()
        self.classes=[['valid', 308], ['invalid', 312]]
        self.test_classifier=SVM.classifer(self.classes)
        self.test_classifier.train('SVM3.xml', True)



    def detect_red(self,image):
        r, g, b = cv2.split(image)
        (minVal, maxValr, minLoc, maxLoc) = cv2.minMaxLoc(r)
        mask = cv2.inRange(image, (245.0*maxValr/255,0,0), (255*maxValr/255, 0,0))
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return self.coordinate_convert(np.array([cx,cy]))
        
    def detect_green(self,image):
        r, g, b = cv2.split(image)
        (minVal, maxValg, minLoc, maxLoc) = cv2.minMaxLoc(g)
        mask = cv2.inRange(image, (0,245.0*maxValg/255,0), (0,255*maxValg/255, 0))
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return self.coordinate_convert(np.array([cx,cy]))
    
    def detect_blue(self,image):
        r, g, b = cv2.split(image)
        (minVal, maxValb, minLoc, maxLoc) = cv2.minMaxLoc(b)
        mask = cv2.inRange(image, (0,0,245.0*maxValb/255),(0,0,255*maxValb/255))
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return self.coordinate_convert(np.array([cx,cy]))

    def detect_darkblue(self,image):
        r, g, b = cv2.split(image)
        (minVal, maxValb, minLoc, maxLoc) = cv2.minMaxLoc(b)
        mask = cv2.inRange(image, (0,0,126.0*maxValb/255),(0,0,129.0*maxValb/255))
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations=1)
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return self.coordinate_convert(np.array([cx,cy]))
    
#     def detect_valid(self, image):
#         # In this method you can focus on detecting the center of the blue circle
#         # Isolate the blue colour in the image as a binary image
#         r, g, b = cv2.split(image)
#         # cv2.imwrite(filename="inhenced0.jpg", img= HSV2)
#         (minVal, maxValg, minLoc, maxLoc) = cv2.minMaxLoc(g)
#         (minVal, maxValb, minLoc, maxLoc) = cv2.minMaxLoc(b)
#         (minVal, maxValr, minLoc, maxLoc) = cv2.minMaxLoc(r)

#         # with open("angle.txt", "a") as file:
#         #     file.write(str(minVal)+"\n")
#         #     file.write(str(maxVal)+"\n\n")
#         lower = 170.0
#         upper = 180.0
#         mask = cv2.inRange(image, (lower * image[0][0][0] / 255, lower * image[0][0][0] / 255, lower * image[0][0][0] / 255),
#                            (upper * image[0][0][0] /255, upper * image[0][0][0] /255, upper * image[0][0][0] / 255))
#         mask_origin = mask.copy()
#         # This applies a dilate that basically makes the binary region larger (the more iterations the larger it becomes)
#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.dilate(mask, kernel, iterations=1)
#         cv2.imwrite(filename="(in)valid.jpg", img=mask)
#         # Obtain the moments of the binary image
#         img = mask
#         a, contours, b = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
#         print len(contours)
#         centres = []
#         output = img.copy()
#         for i in range(len(contours)):
#             moments = cv2.moments(contours[i])
#             centres.append((int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])))
#             # cv2.circle(mask_origin, centres[-1], 3, (0, 0, 0), -1)
#         print centres

#         count1 = 0
#         count2 = 0

#         target_1 = mask_origin[(centres[0][1]-5):(centres[0][1]+5),(centres[0][0]-5):(centres[0][0]+5)]
#         for i in range(0, 10):
#             for j in range(0, 10):
#                 if target_1[i][j] >= 250:
#                     count1 += 1

#         print count1

#         target_1 = mask_origin[(centres[1][1] - 5):(centres[1][1] + 5), (centres[1][0] - 5):(centres[1][0] + 5)]
#         for i in range(0, 10):
#             for j in range(0, 10):
#                 if target_1[i][j] >= 250:
#                     count2 += 1

#         print count2

#         if count1 <= 96:
#             moments = cv2.moments(contours[0])
#             cx = int(moments['m10'] / moments['m00'])
#             cy = int(moments['m01'] / moments['m00'])

#             return self.coordinate_convert(np.array([cx, cy]))
#         else:
#             moments = cv2.moments(contours[1])
#             cx = int(moments['m10'] / moments['m00'])
#             cy = int(moments['m01'] / moments['m00'])

#             return self.coordinate_convert(np.array([cx, cy]))

    def detect_valid(self,image):
        r, g, b = cv2.split(image)
        (minVal, maxValg, minLoc, maxLoc) = cv2.minMaxLoc(g)
        (minVal, maxValb, minLoc, maxLoc) = cv2.minMaxLoc(b)
        (minVal, maxValr, minLoc, maxLoc) = cv2.minMaxLoc(r)
        lower= 160
        upper = 197
        mask = cv2.inRange(image, (lower*maxValr/255,lower*maxValg/255,lower*maxValb/255),(upper*maxValr,upper*maxValg,upper*maxValb/255))
        mask_origin = mask.copy()
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations=1)
        a, contours, b = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        centres = []

        for i in range(len(contours)):
            moments = cv2.moments(contours[i])
            centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
        for i in range(len(centres)):
            crop_img = mask_origin[centres[i][1]-20:centres[i][1]+20, centres[i][0]-20:centres[i][0]+20]
            prediction = self.test_classifier.img_classify(crop_img)[1][0]
            if prediction == 0:
                found = True
                return self.coordinate_convert(np.array([centres[i][0],centres[i][1]]))



    def detect_joint_angles(self, rgbev):
        jointPos4 = [rgbev[3][0], rgbev[3][2]]
        jointpos43 = [rgbev[3][0], rgbev[3][1]]
        jointPos34 = [rgbev[2][0], rgbev[2][2]]
        jointPos3 = [rgbev[2][0], rgbev[2][1]]
        jointPos2 = [rgbev[1][0], rgbev[1][1]]
        jointPos21 = [rgbev[1][0], rgbev[1][2]]
        jointPos12 = [rgbev[0][0], rgbev[0][1]]
        jointPos1 = [rgbev[0][0], rgbev[0][2]]
        targetxz = [rgbev[4][0], rgbev[4][2]]

        detecttarget = [rgbev[4][0], rgbev[4][1]]
        detecttarget2 = [rgbev[4][0], rgbev[4][2]]
        # Solve using trigonometry

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

        j1_pos = np.zeros((3,1))
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


    def IK(self, rgbev, angles):
        ee_pos = np.array(rgbev[3])
        desired_position = np.array(rgbev[4])

        pos_error = desired_position - ee_pos

        Jac = np.matrix(self.Jacobian(rgbev, angles), dtype='float')[0:3,:]
        
        relative = np.eye(3)*0.001
        if np.linalg.matrix_rank(Jac, 0.4) < 3:
            Jac_inv = Jac.T * np.linalg.inv(Jac*Jac.T+relative)
        else:
            Jac_inv = Jac.T * np.linalg.inv(Jac*Jac.T)

        q_dot = Jac_inv*np.matrix(pos_error, dtype='float' ).T

        return np.squeeze(np.array(q_dot.T))

    def coordinate_convert(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution])

    def check_gimble(self, angles):
#         print("enter check")
#         print(angles
        result = [False]
        for i in range(1, 4):
            if(np.abs(angles[i])<0.15) | (np.abs(angles[i])>2.60):
                result[0] = True
                result.append(i)
#         print(result)
#         print("leave check")
        return result
    
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
        prev_jvs = collections.deque(np.zeros(4), 1)

        # Uncomment to have gravity act in the z-axis
        # self.env.world.setGravity((0,0,-9.81))

        colour = 255.0
        darkblue = 127.5
        lightvalue = 1.0

        for i in range(10000000):
            # The change in time between iterations can be found in the self.env.dt variable
            dt = self.env.dt
            # self.env.render returns 2 RGB arrays of the robot, one for the xy-plane, and one for the xz-plane
            arrxy,arrxz = self.env.render('rgb-array')
            rgbev = np.empty([5,3])
            rgbev[0][0:2] = self.detect_red(arrxy)
            rgbev[0][2] = self.detect_red(arrxz)[1]
            rgbev[1][0:2] = self.detect_green(arrxy)
            rgbev[1][2] = self.detect_green(arrxz)[1]
            rgbev[2][0:2] = self.detect_blue(arrxy)
            rgbev[2][2] = self.detect_blue(arrxz)[1]
            rgbev[3][0:2] = self.detect_darkblue(arrxy)
            rgbev[3][2] = self.detect_darkblue(arrxz)[1]
            rgbev[4][0:2] = self.detect_valid(arrxy)
            rgbev[4][2] = self.detect_valid(arrxz)[1]
            
            angle = self.detect_joint_angles(rgbev)    
            detectedJointVels = self.angle_normalize(angle - prev_JAs) / dt
            jointAngles = self.IK(rgbev, angle)
            gimble = self.check_gimble(angle)
            if (len(gimble) == 4):
                detectedJointVels[0]= 20
                detectedJointVels[0]= 0
                detectedJointVels[0]= 0
                detectedJointVels[0]= 0
            prev_JAs = angle
            self.env.step((jointAngles, detectedJointVels, np.zeros(4), np.zeros(4)))
            if (i % 50 == 0):
                print ("{}\nsuccess times: {}".format(i, self.env.success))
                print ("fail times: {}".format(self.env.fail))
            # self.env.step((np.zeros(4),np.zeros(4),np.zeros(4), np.zeros(4)))
            # The step method will send the control input to the robot, the parameters are as follows: (Current Joint Angles/Error, Current Joint Velocities, Desired Joint Angles, Torque input)

#main method
def main():
    reach = MainReacher()
    reach.go()
    try:    
        reach.go()
    except IOError as (errno, strerror):
        print "I/O error({0}): {1}".format(errno, strerror)
        duration = 3  # second
        freq = 440  # Hz
        os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    except ValueError:
        print "Value Error"
        duration = 3  # second
        freq = 440  # Hz
        os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    except:
        print "Unexpected error:", sys.exc_info()[0]
        duration = 3  # second
        freq = 440  # Hz
        os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
        raise

    
if __name__ == "__main__":
    main()


# In[ ]:




