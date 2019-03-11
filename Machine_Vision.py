# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:47:06 2019

@author: Samuel Gibbs
"""

import cv2  
import cv2.aruco as aruco

import math      
import numpy as np

class map_capture():
    
    def __init__(self,camera_option,frame_width,frame_height):
        self.video = cv2.VideoCapture(camera_option)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1);
        self.video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        ret, self.aruco_frame = self.video.read()
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width) # set the resolution - 640,480
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        
    def get_frame_resolution(self):
        width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
        
        return (width,height)
        
    def get_new_frame(self):
        #ok, frame = self.video.read()
        frame = self.aruco_frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        #cv2.imshow("CostMap",thresh)
       
        return ((thresh.flatten()/2.55).astype(int))
    
    def __get_aruco_coor_angle(self,corners,i):
         #Camera calibration parameters
        cameraMatrix = np.array([[1.3953673275755928e+03, 0, 9.9285445205853750e+02], [0,1.3880458574466945e+03, 5.3905119245877574e+02],[ 0., 0., 1.]])
        distCoeffs = np.array([5.7392039180004371e-02, -3.4983260309560962e-02,-2.5933903577082485e-03, 3.4269688895033714e-03,-1.8891849772162170e-01 ])
        
        #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
        rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, cameraMatrix, distCoeffs) 

        aruco.drawAxis(self.aruco_frame, cameraMatrix, distCoeffs, rvec[0], tvec[0], 0.1) #Draw Axis
        aruco.drawDetectedMarkers(self.aruco_frame, corners) #Draw A square around the markers
        aruco_x_coor = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4
        aruco_y_coor = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4
 
        
        #Find conversion factor between coordinates and mm
        aruco_dimensions = 80
        
        conversion_factor = (math.sqrt((abs(corners[i][0][0][0] - corners[i][0][1][0]))**2+(abs(corners[i][0][0][1] - corners[i][0][1][1]))**2))/aruco_dimensions
     
        rotM = np.zeros(shape=(3,3))
        cv2.Rodrigues(rvec[i-1  ], rotM, jacobian = 0)
        R = rotM
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
        singular = sy < 1e-6
     
        if  not singular :
            angle = math.atan2(R[1,0], R[0,0])
        else :
            angle = 0
     
        angle = (-angle)

        return (aruco_x_coor,aruco_y_coor,angle,conversion_factor)

    def __get_object_center_from_aruco(self,aruco_x_coor,aruco_y_coor,angle,distance_x,distance_y,distance_offset,conversion_factor):

        distance_aruco_to_platform_centre = math.sqrt((((distance_x/2)-distance_offset)*conversion_factor)**2 + (((distance_y/2)-distance_offset)*conversion_factor)**2)
        angle_offset = math.atan(((distance_y/2)*conversion_factor)/((distance_x/2)*conversion_factor)) - (math.pi)/2
        
        platform_center_x = int(aruco_x_coor + distance_aruco_to_platform_centre*math.cos(angle-angle_offset))
        platform_center_y = int(aruco_y_coor - distance_aruco_to_platform_centre*math.sin(angle-angle_offset))
        
        return(platform_center_x,platform_center_y)
    
    def __mask_object_with_aruco_coor(self,x0,y0,angle,conversion_factor,object_height,object_width):
        #Draw rotated rectangle
        angle = -angle
        
        b = math.cos(angle) * 0.5
        a = math.sin(angle) * 0.5
        pt0 = (int(x0 - a * object_height - b * object_width), int(y0 + b * object_height - a * object_width))
        pt1 = (int(x0 + a * object_height - b * object_width), int(y0 - b * object_height - a * object_width))
        pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
        pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))
    
        cv2.line(self.aruco_frame, pt0, pt1, (255, 255, 255), 1)
        cv2.line(self.aruco_frame, pt1, pt2, (255, 255, 255), 1)
        cv2.line(self.aruco_frame, pt2, pt3, (255, 255, 255), 1)
        cv2.line(self.aruco_frame, pt3, pt0, (255, 255, 255), 1)
        
        rect_corners = np.array([[pt0],[pt1],[pt2],[pt3]])
        
        #cv2.fillPoly(self.aruco_frame,[rect_corners],(0,0,0))
        
    def __detect_blocks_in_roi(self,x0,y0,angle,conversion_factor,object_height,object_width):
        
        angle = -angle
        b = math.cos(angle) * 0.5
        a = math.sin(angle) * 0.5
        pt0 = (int(x0 - a * object_height - b * object_width), int(y0 + b * object_height - a * object_width))
        pt1 = (int(x0 + a * object_height - b * object_width), int(y0 - b * object_height - a * object_width))
        pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
        pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))
        
        mask = np.array([[pt0],[pt1],[pt2],[pt3]], dtype=np.int32) 
        #Create a new array filled with zeros, size equal to size of the image to be filtered
        image2 = np.zeros((480, 640), np.int8)
        
        cv2.fillPoly(image2, [mask],255)
        
        maskimage2 = cv2.inRange(image2, 1, 255)
        out = cv2.bitwise_and(self.aruco_frame, self.aruco_frame, mask=maskimage2)
        
        cv2.imshow("test",out)
    
    def get_transform(self):
        
        #aruco width and height
        
        ret, self.aruco_frame = self.video.read()
 
        gray = cv2.cvtColor(self.aruco_frame, cv2.COLOR_BGR2GRAY)
        
        T = cv2.getTrackbarPos('T','Thresh')
        retval, gray = cv2.threshold(gray,T,255,cv2.THRESH_BINARY)  
    
        cv2.imshow("Thresh",gray)
    
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250) 
        parameters =  aruco.DetectorParameters_create()
     
        #lists of ids and the corners beloning to each ids
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        self.aruco_frame = aruco.drawDetectedMarkers(self.aruco_frame, corners,ids,(255,255,0))
        
        
        #If an aruco marker is found
        if np.all(ids != None):
            
            #For all aruco markers found
            for i in range(0,int(ids.size)):
                
                
                aruco_x_coor,aruco_y_coor,angle,conversion_factor = self.__get_aruco_coor_angle(corners,i)
                
                #Distance from the center of the aruco marker to the centre of the object in mm
                distance_x = 206
                distance_y = 460
                
                #distance from the center of the aruco marker to the bottom left corner of the platform
                distance_offset = 50
                
                station_center_x,station_center_y = self.__get_object_center_from_aruco(aruco_x_coor,aruco_y_coor,angle,distance_x,distance_y,distance_offset,conversion_factor)
                
                
                station_height = 200*conversion_factor
                station_width = 200*conversion_factor
                self.__detect_blocks_in_roi(station_center_x,station_center_y,angle,conversion_factor,station_height,station_width)
            
                platform_center_x,platform_center_y = self.__get_object_center_from_aruco(aruco_x_coor,aruco_y_coor,angle,distance_x,distance_y,distance_offset,conversion_factor)
                platform_height = 450*conversion_factor
                platform_width = 500*conversion_factor
                self.__mask_object_with_aruco_coor(platform_center_x,platform_center_y,angle,conversion_factor,platform_height,platform_width)
                
                
                found = 1
                
                transform_dict = {
                        "state" : found,
                        "x" : platform_center_x,
                        "y" : platform_center_y,
                        "angle" : angle
                        }
                
                return (transform_dict)
        else:
            found = 0
            transform_dict = {
                        "state" : found,
                        "x" : 0,
                        "y" : 0,
                        "angle" : 0
                        }
            return (transform_dict)
            

                
        
    def show_frame(self):
        #cv2.imshow('costmap',self.thresh)
        cv2.imshow('Aruco',self.aruco_frame)
        
        
    def stop(self):
        cv2.destroyAllWindows()
        self.video.release()
        
    def nothing(self,x):
        pass
        
if __name__ == '__main__':
    map = map_capture(1,640,480)
    # create trackbars for color change
    cv2.namedWindow('Thresh')
    cv2.createTrackbar('T','Thresh',248,255,map.nothing)
    
    while 1:
        
        map.get_transform()
        map.get_new_frame()
        map.show_frame()
        k = cv2.waitKey(1) & 0xff
        #Press escape to close program and take a picture
        if k == 27 :
            map.stop()
            break