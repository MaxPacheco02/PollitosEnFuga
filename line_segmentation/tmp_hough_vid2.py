#!/usr/bin/env python3

import numpy as np
import math
from numpy import linalg
from matplotlib import pyplot as plt
import time
import cv2
from segmentation_functions import *
from sklearn.cluster import KMeans

# ax = [0, 1911, 0, 1077]
# ax = [0, 566, 0, 480]
ax = [0, 523, 0, 432]
ax_2 = [-ax[1]/3, ax[1]*4/3, -ax[3]/3, ax[3]*4/3]
shape = (ax[1], ax[3])


vid_capture = cv2.VideoCapture('/home/max/yolo_training/ducks.mp4')

mtx = [[802.36862544,0.,513.77791238],
        [0.,806.20532073,380.37832786],
        [0.,0.,1.]]

new_mtx = [[843.86861459,0.,501.28525653],
            [0.,819.28126788,379.12698376],
            [0.,0.,1.]]

distortion = [[3.43565672e-01,-2.61438931e+00,-3.22813383e-03,7.12044010e-04,6.24731503e+00]]

mtx_np = np.array(mtx)
new_mtx_np = np.array(new_mtx)
distortion_np = np.array(distortion)

frame_count = 0
skew_line_start = 240
skew_line_end = 750
while(vid_capture.isOpened()):
  ret, frame = vid_capture.read()
  if ret == True:
    frame_count += 1
    w = len(frame[0])
    h = len(frame)
    
    frame_org = np.copy(frame)

    frame = cv2.undistort(frame, mtx_np, distortion_np, None, new_mtx_np)


    frame = frame[(int)(h/20):(int)(19*h/20), (int)(w/3):(int)(19*w/20)]

    print(frame_count, len(frame), len(frame[0]))


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
    point_reg = []

    if lines is not None:
        # Remove repeated lines
        print('a')
        print(lines)
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
                line_ext = [extend_line(l, ax)]
                for x1,y1,x2,y2 in line_ext:
                    cv2.line(line_frame,(x1,y1),(x2,y2),(255,0,0),2)
        if frame_count > skew_line_start and frame_count < skew_line_end and len(lines_reg) > 1:
            # Keep only orthogonal lines
            clust = get_clusters(lines_reg)
            h_lines = []
            v_lines = []
            for i in range(len(clust)):
                if clust[i]:
                    v_lines.append(lines_reg[i])
                else:
                    h_lines.append(lines_reg[i])
            
            v_inter_count = [0] * len(v_lines)
            h_inter_count = [0] * len(h_lines)

            for i in range(len(h_lines) - 1):
                for j in range(i+1, len(h_lines)):
                    inter_inside = False
                    [x, y] = [int(j) for j in intersect(h_lines[i], h_lines[j])]
                    if x >= ax_2[0] and x <= ax_2[1] and y >= ax_2[2] and y <= ax_2[3]:
                        inter_inside = True
                        h_inter_count[i] += 1
                        h_inter_count[j] += 1
            for i in range(len(v_lines) - 1):
                for j in range(i+1, len(v_lines)):
                    inter_inside = False
                    [x, y] = [int(j) for j in intersect(v_lines[i], v_lines[j])]
                    if x >= ax_2[0] and x <= ax_2[1] and y >= ax_2[2] and y <= ax_2[3]:
                        inter_inside = True
                        v_inter_count[i] += 1
                        v_inter_count[j] += 1

            max_index_v = 0
            max_index_h = 0
            if max(v_inter_count) != 0:
                max_index_v = v_lines[v_inter_count.index(max(v_inter_count))]
                # print(max(v_inter_count))
                # print(v_inter_count)

            if max(h_inter_count) != 0:
                max_index_h = h_lines[h_inter_count.index(max(h_inter_count))]
                # print(max(h_inter_count))
                # print(h_inter_count)

            if max(v_inter_count) > max(h_inter_count):
                max_index_inter = max_index_v
            else:
                max_index_inter = max_index_h
            if max(v_inter_count) != 0 or max(h_inter_count) != 0:
                for x1,y1,x2,y2 in [max_index_inter]:
                    cv2.line(line_frame,(x1,y1),(x2,y2),(255,255,255),3)

        # Draw points of intersection
        for i in range(len(lines_reg) - 1):
            for j in range(i+1, len(lines_reg)):
                [x, y] = intersect(lines_reg[i], lines_reg[j])
                if x >= ax[0] and x <= ax[1] and y >= ax[2] and y <= ax[3]:
                    cv2.circle(line_frame,(int(x),int(y)),1,(0,0,255),10)
                    cv2.circle(frame_org,(int(x)+(int)(w/3),int(y)),1,(0,0,255),10)
                    # cv2.putText(line_frame, str(x), (int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        

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
