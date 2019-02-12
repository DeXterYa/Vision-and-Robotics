
# coding: utf-8

# In[1]:


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


# In[2]:


# classes=[['valid', 308], ['invalid', 312]]
# test_classifier=SVM.classifer(classes)
# # test_classifier.addTrainSamples('sample')
# # test_classifier.train()
# test_classifier.train('SVM.xml', True)


# In[ ]:


class MainReacher():
    def __init__(self):
        self.env = gym.make('3DReacherMy-v0')
        self.env.reset()
        self.classes=[['valid', 308], ['invalid', 312]]
        self.test_classifier=SVM.classifer(self.classes)
        # test_classifier.addTrainSamples('sample')
        # test_classifier.train()
        self.test_classifier.train('SVM.xml', True)


    def detect_red(self,image):
        r, g, b = cv2.split(image)
        #In this method you should focus on detecting the center of the green circle
        #SAME AS DETECT_BLUE JUST WITH DIFFERENT COLOUR LIMITS
        (minVal, maxValg, minLoc, maxLoc) = cv2.minMaxLoc(g)
        (minVal, maxValb, minLoc, maxLoc) = cv2.minMaxLoc(b)
        (minVal, maxValr, minLoc, maxLoc) = cv2.minMaxLoc(r)
        #In this method you can focus on detecting the center of the blue circle
        #Isolate the blue colour in the image as a binary image
        mask = cv2.inRange(image, (245.0*maxValr/255,0,0), (255*maxValr/255, 0,0))
#         kernel = np.ones((5,5),np.uint8)
#         mask = cv2.dilate(mask,kernel,iterations=2)
#         mask=cv2.erode(mask,kernel,iterations=3)
#         cv2.imwrite(filename="red.jpg", img= mask)
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        return self.coordinate_convert(np.array([cx,cy]))

    def detect_green(self,image):
        r, g, b = cv2.split(image)
        #In this method you should focus on detecting the center of the green circle
        #SAME AS DETECT_BLUE JUST WITH DIFFERENT COLOUR LIMITS
        (minVal, maxValg, minLoc, maxLoc) = cv2.minMaxLoc(g)
        (minVal, maxValb, minLoc, maxLoc) = cv2.minMaxLoc(b)
        (minVal, maxValr, minLoc, maxLoc) = cv2.minMaxLoc(r)
        #In this method you can focus on detecting the center of the blue circle
        #Isolate the blue colour in the image as a binary image
        mask = cv2.inRange(image, (0,245.0*maxValg/255,0), (0,255*maxValg/255, 0))
#         kernel = np.ones((5,5),np.uint8)
#         mask = cv2.dilate(mask,kernel,iterations=3)
#         cv2.imwrite(filename="green.jpg", img= mask)
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        return self.coordinate_convert(np.array([cx,cy]))

    def detect_blue(self,image):
        r, g, b = cv2.split(image)
        (minVal, maxValg, minLoc, maxLoc) = cv2.minMaxLoc(g)
        (minVal, maxValb, minLoc, maxLoc) = cv2.minMaxLoc(b)
        (minVal, maxValr, minLoc, maxLoc) = cv2.minMaxLoc(r)
        #In this method you can focus on detecting the center of the blue circle
        #Isolate the blue colour in the image as a binary image
        mask = cv2.inRange(image, (0,0,245.0*maxValb/255),(0,0,255*maxValb/255))
        #This applies a dilate that basically makes the binary region larger (the more iterations the larger it becomes)
#         kernel = np.ones((5,5),np.uint8)
#         mask = cv2.dilate(mask,kernel,iterations=3)
#         cv2.imwrite(filename="blue.jpg", img= mask)
        #Obtain the moments of the binary image
        M = cv2.moments(mask)
        #Calculate pixel coordinates for the center of the blob
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        #Convert pixel coordinates to world coordinates
        return self.coordinate_convert(np.array([cx,cy]))

    def detect_darkblue(self,image):
        #In this method you can focus on detecting the center of the blue circle
        #Isolate the blue colour in the image as a binary image
        r, g, b = cv2.split(image)
        # cv2.imwrite(filename="inhenced0.jpg", img= HSV2)
        (minVal, maxValg, minLoc, maxLoc) = cv2.minMaxLoc(g)
        (minVal, maxValb, minLoc, maxLoc) = cv2.minMaxLoc(b)
        (minVal, maxValr, minLoc, maxLoc) = cv2.minMaxLoc(r)

        # with open("angle.txt", "a") as file:
        #     file.write(str(minVal)+"\n")
        #     file.write(str(maxVal)+"\n\n")
        mask = cv2.inRange(image, (0,0,126.0*maxValb/255),(0,0,129.0*maxValb/255))
        #This applies a dilate that basically makes the binary region larger (the more iterations the larger it becomes)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations=1)
#         cv2.imwrite(filename="deep.jpg", img= mask)
        #Obtain the moments of the binary image
        M = cv2.moments(mask)
        #Calculate pixel coordinates for the center of the blob
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        #Convert pixel coordinates to world coordinates
        return self.coordinate_convert(np.array([cx,cy]))

    def detect_valid(self,image):
        #In this method you can focus on detecting the center of the blue circle
        #Isolate the blue colour in the image as a binary image
        r, g, b = cv2.split(image)
        # cv2.imwrite(filename="inhenced0.jpg", img= HSV2)
        (minVal, maxValg, minLoc, maxLoc) = cv2.minMaxLoc(g)
        (minVal, maxValb, minLoc, maxLoc) = cv2.minMaxLoc(b)
        (minVal, maxValr, minLoc, maxLoc) = cv2.minMaxLoc(r)
        #start
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(image)
        (minVal, maxValg, minLoc, maxLoc) = cv2.minMaxLoc(V)
        #end
        # with open("angle.txt", "a") as file:
        #     file.write(str(minVal)+"\n")
        #     file.write(str(maxVal)+"\n\n")
        lower= 170
        upper = 187
        mask = cv2.inRange(image, (lower*maxValr/255,lower*maxValg/255,lower*maxValb/255),(upper*maxValr,upper*maxValg,upper*maxValb/255))
        mask_origin = mask.copy()
        #This applies a dilate that basically makes the binary region larger (the more iterations the larger it becomes)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations=1)
#         cv2.imwrite(filename="(in)valid.jpg", img= mask)
        #Obtain the moments of the binary image
        img = mask
        a, contours, b = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        centres = []
        output = img.copy()
        for i in range(len(contours)):
            moments = cv2.moments(contours[i])
            centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
        for i in range(len(centres)):
            crop_img = mask_origin[centres[i][1]-20:centres[i][1]+20, centres[i][0]-20:centres[i][0]+20]
#             cv2.imwrite('crop'+str(i)+'.jpg',crop_img)
            prediction = self.test_classifier.classify('crop'+str(i)+'.jpg')[1][0]
            if prediction == 0:
                return self.coordinate_convert(np.array([centres[i][0],centres[i][1]]))

    def detect_joint_angles(self, image_xy, image_xz):
        jointPos4 = self.detect_darkblue(image_xz)
        jointpos43 = self.detect_darkblue(image_xy)
        jointPos34 = self.detect_blue(image_xz)
        jointPos3 = self.detect_blue(image_xy)
        jointPos2 = self.detect_green(image_xy)
        jointPos21 = self.detect_green(image_xz)
        jointPos12 = self.detect_red(image_xy)
        jointPos1 = self.detect_red(image_xz)
        targetxz = self.detect_valid(image_xz)
        targetxy = self.detect_valid(image_xy)

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
        elif theta_cos < -1:
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
        elif theta_cos < -1:
            theta_cos = -1
        theta = math.acos(theta_cos)
        rotation1 = np.zeros((3,3))
        rotation1 = [[math.cos(ja2),-math.sin(ja2),0],[math.sin(ja2),math.cos(ja2),0],[0,0,1]]
        y_old = np.array([0, 1, 0])
        y_new = np.dot(rotation1, y_old)
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
        elif theta_cos < -1:
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

    def Jacobian(self, arrxy, arrxz):

        rx, ry = self.detect_red(arrxy)
        rx, rz = self.detect_red(arrxz)

        gx, gy = self.detect_green(arrxy)
        gx, gz = self.detect_green(arrxz)

        bx, by = self.detect_blue(arrxy)
        bx, bz = self.detect_blue(arrxz)

        ex, ey = self.detect_darkblue(arrxy)
        ex, ez = self.detect_darkblue(arrxz)


        jacobian = np.zeros((6, 4))
        z_vector = np.array([0, 0, 1])
        y_vector = np.array([0, 1, 0])

        j1_pos = np.zeros((3,1))
        j2_pos = np.array([rx, ry, rz]).reshape(3, 1)
        j3_pos = np.array([gx, gy, gz]).reshape(3, 1)
        j4_pos = np.array([bx, by, bz]).reshape(3, 1)
        ee_pos = np.array([ex, ey, ez]).reshape(3, 1)

        pos_3D = np.zeros(3)
        pos_3D[0: 3] = (ee_pos - j1_pos).T
        jacobian[0:3, 0] = np.cross(y_vector, pos_3D)
        jacobian[3:6, 0] = y_vector

        pos_3D[0: 3] = (ee_pos - j2_pos).T
        jacobian[0:3, 1] = np.cross(z_vector, pos_3D)
        jacobian[3:6, 1] = z_vector

        pos_3D[0: 3] = (ee_pos - j3_pos).T
        jacobian[0:3, 2] = np.cross(z_vector, pos_3D)
        jacobian[3:6, 2] = z_vector

        pos_3D[0: 3] = (ee_pos - j4_pos).T
        jacobian[0:3, 3] = np.cross(y_vector, pos_3D)
        jacobian[3:6, 3] = y_vector
        return jacobian


    def IK(self, arrxy, arrxz):
        ex, ey = self.detect_darkblue(arrxy)
        ex, ez = self.detect_darkblue(arrxz)
        ee_pos = np.array([ex, ey, ez]).reshape(1, 3)
        
        vx, vy = self.detect_valid(arrxy)
        vx, vz = self.detect_valid(arrxz)
        desired_position = np.array([vx, vy, vz]).reshape(1, 3)
        print("++")
        print (ee_pos)
        print (desired_position)
        
        pos_error = desired_position - ee_pos

        Jac = np.matrix(self.Jacobian(arrxy, arrxz))[0:3,:]

        if np.linalg.matrix_rank(Jac, 0.4) < 3:
            Jac_inv = Jac.T
        else:
            Jac_inv = Jac.T * np.linalg.inv(Jac*Jac.T)

        q_dot = Jac_inv*np.matrix(pos_error ).T

        return np.squeeze(np.array(q_dot.T))

    def coordinate_convert(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution])

    def angle_normalize(self,x):
        #Normalizes the angle between pi and -pi
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def increase_brightness(self, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] = value + v[v <= lim]

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

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
#         with open("angle.txt", "w") as file:
#             file.write("Start\n")
        # Uncomment to have gravity act in the z-axis
        # self.env.world.setGravity((0,0,-9.81))
                #Calculate the initial thresholds

        for _ in range(100000):
            # The change in time between iterations can be found in the self.env.dt variable
            dt = self.env.dt
            # self.env.render returns 2 RGB arrays of the robot, one for the xy-plane, and one for the xz-plane
            arrxy,arrxz = self.env.render('rgb-array')

#             angle = self.detect_joint_angles(arrxy, arrxz)

#             print(angle[0], angle[1], angle[2], angle[3])

#             jointAngles = np.array([3, -3, -3, 3])

#             self.env.step((np.zeros(4), np.zeros(4), jointAngles, np.zeros(4)))
            detectedJointAngles = self.detect_joint_angles(arrxy, arrxz)
            jointAngles = self.IK(arrxy, arrxz)
            detectedJointVels = self.angle_normalize(detectedJointAngles - prev_JAs)/dt
            print(jointAngles)
            
            prev_JAs = detectedJointAngles

            #jointAngles = np.array([3, -3, -3, 3])

            self.env.step((jointAngles, detectedJointVels, np.zeros(4), np.zeros(4)))
            # self.env.step((np.zeros(4),np.zeros(4),np.zeros(4), np.zeros(4)))
            # The step method will send the control input to the robot, the parameters are as follows: (Current Joint Angles/Error, Current Joint Velocities, Desired Joint Angles, Torque input)

#main method
def main():
    cv2.useOptimized()
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()


# In[ ]:




