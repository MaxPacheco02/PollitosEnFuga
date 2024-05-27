#!/usr/bin/env python3

import numpy as np
import math
from numpy import linalg
from matplotlib import pyplot as plt
import time
import cv2
from segmentation_functions import *
from sklearn.cluster import KMeans

ax = [0, 1911, 0, 1077]
shape = (ax[1], ax[3])


vid_capture = cv2.VideoCapture('/home/max/yolo_training/ducks.mp4')

while(vid_capture.isOpened()):
  ret, frame = vid_capture.read()
  if ret == True:
    w = len(frame[0])
    h = len(frame)
    
    frame_org = np.copy(frame)

    frame = frame[::, (int)(w/3)::]

    ax_cut = [0, 1911, (int)(w/3), 1077]

    frame_rgb = np.copy(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Define our parameters for Canny
    low_threshold = 100
    high_threshold = 100
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1
    theta = np.pi/180
    threshold = 100
    min_line_length = 0
    max_line_gap = 10000
    line_frame = np.copy(frame_rgb) #creating an image copy to draw lines on
    # Run Hough on the edge-detected image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    # Iterate over the output "lines" and draw lines on the image copy

    lines_reg = []
    inter_reg = []
    last_inter = []
    ortho_lines = []
    line_ang_coords = []

    if lines is not None:
        # Remove repeated lines
        ang_treshold = 0.3
        dist_treshold = 20
        for line in lines:
            l = line[0]
            rept_line = False
            for lin in lines_reg:
                if(is_line_similar(lin, l, ang_treshold, dist_treshold)):
                    rept_line = True
            if not rept_line:
                lines_reg.append(l)
                a = get_ang(l)
                line_ang_coords.append([math.cos(a), math.sin(a)])

        # Remove non ortho lines
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(line_ang_coords)
        for i in range(len(lines_reg) - 1):
            is_ortho = True
            for j in range(i+1, len(lines_reg)):
                [x, y] = intersect(lines_reg[i], lines_reg[j])
                if x >= ax[0] and x <= ax[1] and y >= ax[2] and y <= ax[3]:
                    if kmeans.labels_[i] == kmeans.labels_[j]:
                        is_ortho = False
            if is_ortho:
                ortho_lines.append(lines_reg[i])

        for i in ortho_lines:
            line_ext = [extend_line(i, ax)]
            for x1,y1,x2,y2 in line_ext:
                cv2.line(line_frame,(x1,y1),(x2,y2),(255,0,0),2)


        # for x1,y1,x2,y2 in [lines_reg[i]]:
        #     cv2.line(line_frame,(x1,y1),(x2,y2),(255,0,255),3)                       
        # for x1,y1,x2,y2 in [lines_reg[j]]:
        #     cv2.line(line_frame,(x1,y1),(x2,y2),(255,0,255),3)         

        # Draw points of intersection
        for i in range(len(ortho_lines) - 1):
            for j in range(i+1, len(ortho_lines)):
                [x, y] = intersect(ortho_lines[i], ortho_lines[j])
                if x >= ax[0] and x <= ax[1] and y >= ax[2] and y <= ax[3]:
                    cv2.circle(line_frame,(int(x),int(y)),1,(0,0,255),10)
                    cv2.circle(frame_org,(int(x)+(int)(w/3),int(y)),1,(0,0,255),10)
                    # cv2.putText(line_frame, str(x), (int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    print(len(line_frame), len(line_frame[0]))
    cv2.imshow('LineFrame',line_frame)
    # cv2.imshow('Frame',frame_org)
    # time.sleep(0.1)
    key = cv2.waitKey(1)
     
    if key == ord('q'):
      break
  else:
    break
 
vid_capture.release()
cv2.destroyAllWindows()
