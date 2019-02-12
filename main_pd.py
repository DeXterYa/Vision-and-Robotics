#!/usr/bin/env python2.7
import gym
import reacher3D.Reacher
import numpy as np
import cv2
import math
import scipy as sp
import collections
import time
class MainReacher():
    def __init__(self):
        self.env = gym.make('3DReacherMy-v0')
        self.env.reset()



    def detect_red(self, image):

        mask = cv2.inRange(image, (image[0][0][0]-5, 0, 0), (image[0][0][0]+5, 0, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return self.coordinate_convert(np.array([cx, cy]))

    def detect_green(self, image):

        mask = cv2.inRange(image, (0, image[0][0][0]-5, 0), (0, image[0][0][0]+5, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return self.coordinate_convert(np.array([cx, cy]))

    def detect_blue(self, image):

        mask = cv2.inRange(image, (0, 0, image[0][0][0]-5), (0, 0, image[0][0][0]+5))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return self.coordinate_convert(np.array([cx, cy]))

    def detect_darkblue(self, image, colour):



        mask = cv2.inRange(image, (0, 0, 127.5*image[0][0][0]/255-1), (0, 0, 127.5*image[0][0][0]/255+1))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return self.coordinate_convert(np.array([cx, cy]))

    def detect_valid(self, image):
        # In this method you can focus on detecting the center of the blue circle
        # Isolate the blue colour in the image as a binary image
        r, g, b = cv2.split(image)
        # cv2.imwrite(filename="inhenced0.jpg", img= HSV2)
        (minVal, maxValg, minLoc, maxLoc) = cv2.minMaxLoc(g)
        (minVal, maxValb, minLoc, maxLoc) = cv2.minMaxLoc(b)
        (minVal, maxValr, minLoc, maxLoc) = cv2.minMaxLoc(r)

        # with open("angle.txt", "a") as file:
        #     file.write(str(minVal)+"\n")
        #     file.write(str(maxVal)+"\n\n")
        lower = 170.0
        upper = 180.0
        mask = cv2.inRange(image, (lower * image[0][0][0] / 255, lower * image[0][0][0] / 255, lower * image[0][0][0] / 255),
                           (upper * image[0][0][0] /255, upper * image[0][0][0] /255, upper * image[0][0][0] / 255))
        mask_origin = mask.copy()
        # This applies a dilate that basically makes the binary region larger (the more iterations the larger it becomes)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        cv2.imwrite(filename="(in)valid.jpg", img=mask)
        # Obtain the moments of the binary image
        img = mask
        a, contours, b = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        print len(contours)
        centres = []
        output = img.copy()
        for i in range(len(contours)):
            moments = cv2.moments(contours[i])
            centres.append((int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])))
            # cv2.circle(mask_origin, centres[-1], 3, (0, 0, 0), -1)
        print centres

        count1 = 0
        count2 = 0

        target_1 = mask_origin[(centres[0][1]-5):(centres[0][1]+5),(centres[0][0]-5):(centres[0][0]+5)]
        for i in range(0, 10):
            for j in range(0, 10):
                if target_1[i][j] >= 250:
                    count1 += 1

        print count1

        target_1 = mask_origin[(centres[1][1] - 5):(centres[1][1] + 5), (centres[1][0] - 5):(centres[1][0] + 5)]
        for i in range(0, 10):
            for j in range(0, 10):
                if target_1[i][j] >= 250:
                    count2 += 1

        print count2

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

        # cv2.imwrite('output.png', mask_origin)
        # M = cv2.moments(mask)
        # #Calculate pixel coordinates for the center of the blob
        # cx = int(M['m10']/M['m00'])
        # cy = int(M['m01']/M['m00'])
        # #Convert pixel coordinates to world coordinates
        # return self.coordinate_convert(np.array([cx,cy]))


    def detect_joint_angles(self, image_xy, image_xz, darkblue):
        jointPos4 = self.detect_darkblue(image_xz, darkblue)
        jointpos43 = self.detect_darkblue(image_xy, darkblue)
        jointPos34 = self.detect_blue(image_xz)
        jointPos3 = self.detect_blue(image_xy)
        jointPos2 = self.detect_green(image_xy)
        jointPos21 = self.detect_green(image_xz)
        jointPos12 = self.detect_red(image_xy)
        jointPos1 = self.detect_red(image_xz)

        detecttarget = self.detect_valid(image_xy)
        detecttarget2 = self.detect_valid(image_xz)
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


    def Jacobian(self, arrxy, arrxz, colour, angles):

        red_xy = self.detect_red(arrxy)
        red_xz = self.detect_red(arrxz)

        green_xy = self.detect_green(arrxy)
        green_xz = self.detect_green(arrxz)

        blue_xy = self.detect_blue(arrxy)
        blue_xz = self.detect_blue(arrxz)

        ee_xy = self.detect_darkblue(arrxy, colour)
        ee_xz = self.detect_darkblue(arrxz, colour)


        jacobian = np.zeros((6, 4))
        z_vector = np.array([0, 0, 1])
        y_vector = np.array([0, -1, 0])

        j1_pos = np.zeros((3,1))
        j2_pos = np.array([red_xy[0], red_xy[1], red_xz[1]]).reshape(3, 1)
        j3_pos = np.array([green_xy[0], green_xy[1], green_xz[1]]).reshape(3, 1)
        j4_pos = np.array([blue_xy[0], blue_xy[1], blue_xz[1]]).reshape(3, 1)
        ee_pos = np.array([ee_xy[0], ee_xy[1], ee_xz[1]]).reshape(3, 1)


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

        print y_vector


        pos_3D[0: 3] = (ee_pos - j4_pos).T

        jacobian[0:3, 3] = np.cross(y_vector, pos_3D)
        jacobian[3:6, 3] = y_vector

        return jacobian


    def IK(self, arrxy, arrxz, colour, angles):
        ee_xy = self.detect_darkblue(arrxy, colour)
        ee_xz = self.detect_darkblue(arrxz, colour)
        ee_pos = np.array([ee_xy[0], ee_xy[1], ee_xz[1]]).reshape(1, 3)

        detecttarget_xy = self.detect_valid(arrxy)
        detecttarget_xz = self.detect_valid(arrxz)

        desired_position = np.array([detecttarget_xy[0], detecttarget_xy[1], detecttarget_xz[1]]).reshape(1, 3)

        #print desired_position

        #print ee_pos

        pos_error = desired_position - ee_pos

        Jac = np.matrix(self.Jacobian(arrxy, arrxz, colour, angles), dtype='float')[0:3,:]

        #print Jac

        if np.linalg.matrix_rank(Jac, 0.4) < 3:
            Jac_inv = Jac.T

        else:
            Jac_inv = Jac.T * np.linalg.inv(Jac*Jac.T)

        #Jac_inv = Jac.T * np.linalg.inv(Jac * Jac.T)

        #print Jac_inv

        q_dot = Jac_inv*np.matrix(pos_error, dtype='float' ).T

        return np.squeeze(np.array(q_dot.T))


    def ts_pd_control(self, curr_ee_pos, curr_ee_vel, desired_ee_pos):
        P = np.array([80, 80, 80])
        D = np.array([30, 30, 30])


        P_error = np.matrix(desired_ee_pos - curr_ee_pos).T
        D_error = np.zeros(shape=(3, 1)) - np.matrix(curr_ee_vel).T
        
        PD_error = np.diag(P)*P_error + np.diag(D)*D_error

        return PD_error

    def grav(self, arrxy, arrxz, colour, angles):

        g = 9.81
        m = 1



        joint_2xy = self.detect_red(arrxy)
        joint_2xz = self.detect_red(arrxz)
        joint_2 = np.array([joint_2xy[0], joint_2xy[1], joint_2xz[1]]).reshape(1, 3)
        p1_1 =  np.array([joint_2xy[0], joint_2xy[1], joint_2xz[1]]).reshape(1, 3)/2


        joint_3xy = self.detect_green(arrxy)
        joint_3xz = self.detect_green(arrxz)
        joint_3 = np.array([joint_3xy[0], joint_3xy[1], joint_3xz[1]]).reshape(1, 3)
        p1_2 =  (joint_2 + joint_3).reshape(1, 3)/2

        joint_4xy = self.detect_blue(arrxy)
        joint_4xz = self.detect_blue(arrxz)
        joint_4 = np.array([joint_4xy[0], joint_4xy[1], joint_4xz[1]]).reshape(1, 3)
        p1_3 = (joint_3 + joint_4).reshape(1, 3)/2

        joint_5xy = self.detect_darkblue(arrxy, colour)
        joint_5xz = self.detect_darkblue(arrxz, colour)
        joint_5 = np.array([joint_5xy[0], joint_5xy[1], joint_5xz[1]]).reshape(1, 3)
        p1_4 = (joint_4 + joint_5).reshape(1, 3)/2


        translation_1 = joint_2

        rotation_1 = np.zeros((3, 3))
        rotation_1 = np.array([[math.cos(angles[0]), 0, -math.sin(angles[0])], [0, 1, 0],
                      [math.sin(angles[0]), 0, math.cos(angles[0])]])

        p2_2 = np.dot(rotation_1.T, (p1_2 - translation_1).T).reshape(1, 3)
        p2_3 = np.dot(rotation_1.T, (p1_3 - translation_1).T).reshape(1, 3)
        p2_4 = np.dot(rotation_1.T, (p1_4 - translation_1).T).reshape(1, 3)

        rotation_2 = np.zeros((3, 3))
        rotation_2 = np.array([[math.cos(angles[1]), -math.sin(angles[1]), 0], [math.sin(angles[1]), math.cos(angles[1]), 0],
                      [0, 0, 1]])

        translation_2 = joint_3
        p3_3 = np.dot(np.dot(rotation_1, rotation_2).T, (p1_3 - translation_2).T).reshape(1, 3)
        p3_4 = np.dot(np.dot(rotation_1, rotation_2).T, (p1_4 - translation_2).T).reshape(1, 3)


        rotation_3 = np.zeros((3, 3))
        rotation_3 = np.array([[math.cos(angles[2]), -math.sin(angles[2]), 0], [math.sin(angles[2]), math.cos(angles[2]), 0],
                      [0, 0, 1]])
        translation_3 = joint_4

        rotation_all = np.dot(np.dot(rotation_1, rotation_2), rotation_3)
        p4_4 = np.dot(rotation_all.T, (p1_4 - translation_3).T).reshape(1, 3)

        F_gra = np.array([0, 0, 9.81]).reshape(1, 3)

        torque_4 = np.cross(p4_4, F_gra)
        torque_3 = np.cross(p3_3, F_gra) + np.cross(p3_4, F_gra)
        torque_2 = np.cross(p2_2, F_gra) + np.cross(p2_3, F_gra) + np.cross(p2_4, F_gra)
        torque_1 = np.cross(p1_1, F_gra) + np.cross(p1_2, F_gra) + np.cross(p1_3, F_gra) + np.cross(p1_4, F_gra)


        torque_1 = 7/2 * g *math.cos(angles[0]) + 5/2 * g * math.cos(angles[1]) * math.cos(angles[0]) - 3/2 * g * math.cos(angles[1] + angles[2]) * math.sin(angles[0]) + 1/2 * g * math.sin(angles[3]) *math.cos(angles[0])
        torque_2 = - 5/2 * g * math.sin(angles[1]) * math.sin(angles[0]) - 3/2 * g * math.sin(angles[1] + angles[2]) * math.cos(angles[0])
        torque_3 = - 3/2 * g * math.sin(angles[1] + angles[2]) * math.cos(angles[0])
        torque_4 = 1/2 * g * math.cos(angles[3]) *math.sin(angles[0])


        # print p1_1
        # print p1_2
        # print p1_3
        # print p1_4

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
        self.env.controlMode="TORQUE"
        #Run 100000 iterations
        prev_JAs = np.zeros(4)
        prev_jvs = collections.deque(np.zeros(4), 1)

        # Uncomment to have gravity act in the z-axis
        self.env.world.setGravity((0, 0, -9.81))

        colour = 255.0
        darkblue = 127.5
        lightvalue = 1.0

        prevEePos = np.zeros(shape=(1, 3))
        self.env.enable_gravity(True)

        for _ in range(1000000):
            # The change in time between iterations can be found in the self.env.dt variable
            dt = self.env.dt
            # self.env.render returns 2 RGB arrays of the robot, one for the xy-plane, and one for the xz-plane
            arrxy,arrxz = self.env.render('rgb-array')


            lightvalue = (arrxy[0][0][0]+0.0)/colour

            colour = arrxy[0][0][0]+0.0
            darkblue = darkblue * lightvalue

            angle = self.detect_joint_angles(arrxy, arrxz, darkblue)

            ee_xy = self.detect_darkblue(arrxy, darkblue)
            ee_xz = self.detect_darkblue(arrxz, darkblue)
            ee_pos = np.array([ee_xy[0], ee_xy[1], ee_xz[1]])

            ee_tar_xy = self.detect_valid(arrxy)
            ee_tar_xz = self.detect_valid(arrxz)
            ee_tar = np.array([ee_tar_xy[0], ee_tar_xy[1], ee_tar_xz[1]])

            ee_vel = (ee_pos - prevEePos)/dt

            prevEePos = ee_pos
            J = self.Jacobian(arrxy, arrxz, darkblue, angle)[0:3, :]

            ee_desired_force = self.ts_pd_control(ee_pos, ee_vel, ee_tar)
            grav_opposite_torques = self.grav(arrxy, arrxz, darkblue,angle)
            torques = J.T*ee_desired_force + grav_opposite_torques






            #jointAngles = np.array([3, -3, -3, 3])

            self.env.step((np.zeros(4), np.zeros(4), np.zeros(4), torques))
            # self.env.step((np.zeros(4),np.zeros(4),np.zeros(4), np.zeros(4)))
            # The step method will send the control input to the robot, the parameters are as follows: (Current Joint Angles/Error, Current Joint Velocities, Desired Joint Angles, Torque input)

#main method
def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()
