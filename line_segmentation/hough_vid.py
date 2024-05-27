#!/usr/bin/env python3

import numpy as np
import math
from numpy import linalg
from matplotlib import pyplot as plt
import time
import cv2
from segmentation_functions import *
from sklearn.cluster import KMeans

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
yolo = YOLO('/home/max/yolo_training/runs/detect/train_ducks/weights/best.pt')  # pretrained YOLOv8n model

# ax = [0, 1911, 0, 1077]
# ax = [0, 566, 0, 480]
ax = [0, 523, 0, 432]
ax_2 = [-ax[1]/3, ax[1]*4/3, -ax[3]/3, ax[3]*4/3]
shape = (ax[1], ax[3])

last_h_ang = 2.
last_h_ang = 0.
last_hv_lines = [[]]
last_first_midpoints = [[-1,-1],[-1,-1]]
offset_accum = [0.] * 2

vid_capture = cv2.VideoCapture('/home/max/yolo_training/ducks.mp4')
# vid_capture = cv2.VideoCapture('/home/max/yolo_training/line_segmentation/tiles.mp4')

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

last_point_reg = []
point_coords_dict = {
    1: [0,0]
}

duck_shape = (1000, 1000)

while(vid_capture.isOpened()):
    duck_frame = np.full((duck_shape[0], duck_shape[1], 3) , (0, 0, 0), np.uint8)

    cv2.line(duck_frame,(500,0),(500,1000),(255,255,255),1)
    cv2.line(duck_frame,(0,500),(1000,500),(255,255,255),1)
    ret, frame = vid_capture.read()
    if ret == True:
        frame_count += 1
        w = len(frame[0])
        h = len(frame)

        frame_org = np.copy(frame)

        # frame = cv2.undistort(frame, mtx_np, distortion_np, None, new_mtx_np)

        # frame = frame[(int)(h/20):(int)(19*h/20), (int)(w/3):(int)(19*w/20)]

        frame = frame[::, (int)(w/3):]
        # frame = cv2.resize(frame, shape) # just for tiles!

        # print(frame_count, len(frame), len(frame[0]))

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
        point_reg = []

        if frame_count > -1:
            if lines is not None:
                # Remove repeated lines
                # print('a')
                # print(lines)
                ang_treshold = 0.3
                dist_treshold = 30
                for line in lines:
                    l = line[0]
                    rept_line = False
                    for lin in lines_reg:
                        if(is_line_similar(lin, l, ang_treshold, dist_treshold)):
                            rept_line = True
                    if not rept_line:
                        lines_reg.append(extend_line(l, ax))

                if len(lines_reg) > 1:
                    clust = get_clusters(lines_reg)
                    h_lines = []
                    v_lines = []
                    hv_lines = [h_lines, v_lines]
                    for i in range(len(clust)):
                        hv_lines[clust[i]].append(lines_reg[i])
                    if not h_line_similar2(last_hv_lines[0], hv_lines):
                        hv_lines = hv_lines[-1:] + hv_lines[:-1]
                    # if not h_line_similar(last_h_ang, hv_lines):
                    #     hv_lines = hv_lines[-1:] + hv_lines[:-1]
                    # last_h_ang = get_ang(hv_lines[0][0])

                    for i in range(2):
                        for j in hv_lines[i]:
                            for x1,y1,x2,y2 in [j]:
                                cv2.line(line_frame,(x1,y1),(x2,y2),(255*(1-i),255*i,0),3)
                            
                    # Sort (horizontal and vertical) lines by position 
                    hv_count = 0
                    hv_ave_lines = []
                    hv_idxs = []
                    for i in range(2):
                        ave_line, line_idxs = line_sort(hv_lines[i], last_first_midpoints[i], ax)
                        last_first_midpoints[i] = midpoint(hv_lines[i][line_idxs.index(0)])
                        hv_ave_lines.append(ave_line)
                        hv_idxs.append(line_idxs)
                        half_idx = line_idxs.index((len(hv_lines[i]) - 1) // 2)
                    hv_lines = sorted_lines(hv_lines, hv_idxs).copy()
                        # for j in range(len(hv_lines[i])):
                        #     for x1,y1,x2,y2 in [hv_lines[i][j]]:
                        #         cv2.line(line_frame,(x1,y1),(x2,y2),(255,0,0),2)
                        #         md = midpoint(hv_lines[i][j])
                                # cv2.putText(line_frame, str(line_idxs[j]), (int(md[0]), int(md[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, hv_count * 255), 2, cv2.LINE_AA)
                        # hv_count += 1
                    
                    # Offset current lines to match past's
                    if len(last_hv_lines[0]) > 1 and len(last_hv_lines[1]) > 1 and len(hv_lines[0]) > 1 and len(hv_lines[1]) > 1:
                        # print(last_hv_lines)
                        for i in range(2):
                            offset_accum[i] -= find_offset(hv_lines[i], last_hv_lines[i])

                    # Draw points of intersections
                    keypoint_count = 0
                    point_reg = []
                    for i in range(len(hv_lines[0])):
                        for j in range(len(hv_lines[1])):
                            l1 = hv_lines[0][i]
                            l2 = hv_lines[1][j]
                            [x, y] = [int(k) for k in intersect(l1, l2)]
                            if x >= ax[0] and x <= ax[1] and y >= ax[2] and y <= ax[3]:
                                keypoint_count += 1
                                new_point = Point(keypoint_count, x, y, [i + int(offset_accum[0]), j + int(offset_accum[1])])
                                point_reg.append(new_point)
                                
                                cv2.circle(line_frame,(x,y),1,(0,0,255),10)
                    # for i in range(len(hv_idxs[0])):
                    #     for j in range(len(hv_idxs[1])):
                    #         l1 = hv_lines[0][hv_idxs[0].index(i)].copy()
                    #         l2 = hv_lines[1][hv_idxs[1].index(j)].copy()
                    #         [x, y] = [int(k) for k in intersect(l1, l2)]
                    #         if x >= ax[0] and x <= ax[1] and y >= ax[2] and y <= ax[3]:
                    #             keypoint_count += 1
                    #             new_point = Point(keypoint_count, x, y, [i, j])
                    #             point_reg.append(new_point)
                                
                    #             cv2.circle(line_frame,(x,y),1,(0,0,255),10)
                                # cv2.putText(line_frame, str(keypoint_count), (int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                    
                    # for k in range(len(point_reg) - 1):
                    #     for l in range(k + 1, len(point_reg)):
                    #         does_share, n = share_lines(point_reg[k], point_reg[l])
                    #         if does_share:
                    #             if n:
                    #                 point_vertical_relation(point_reg[k], point_reg[l])
                    #             else:
                    #                 point_horizontal_relation(point_reg[k], point_reg[l])

                    # # Find transform
                    # found_rot = False
                    # if len(last_point_reg) > 1:
                    #     for i in range(len(last_point_reg)):
                    #         for j in range(len(point_reg)):
                    #             if not found_rot:
                    #                 shares_adj, rotation_rel = share_adj([i,j],[last_point_reg,point_reg])
                    #                 if same_point(last_point_reg[i], point_reg[j]) and shares_adj:
                    #                     found_rot = True
                    #                     trans_rel = get_trans_rel(last_point_reg[i], point_reg[j])
                    #                     sp_coords = [point_reg[j].x, point_reg[j].y]
                    #                     cv2.circle(line_frame,(sp_coords[0],sp_coords[1]),10,(255,0,255),10)

                        # Transform new frame
                        # if found_rot:
                            # print("\nrotation relation", rotation_rel)
                            # print("\translation relation", trans_rel)
                            # for i in point_reg:
                            #     point_rotate(i, rotation_rel)
                                # point_translate(i, trans_rel)

                    last_hv_lines = hv_lines.copy()

                    # Find points between which a duck is
                    results = yolo(frame_rgb, stream=True)  # generator of Results objects

                    # Process results generator
                    duck_poses = []
                    for result in results:
                        boxes = result.boxes  # Boxes object for bounding box outputs
                        if boxes is not None:
                            duck_poses.append(tensor_to_midpoint(boxes.xyxy.tolist()))
                            for box in boxes.xyxy.tolist(): 
                                xB = int(box[2])
                                xA = int(box[0])
                                yB = int(box[3])
                                yA = int(box[1])
                                
                                cv2.rectangle(line_frame, (xA, yA), (xB, yB), (0, 0, 255), 4)

                    for duck_pose in duck_poses[0]:
                        print(duck_pose)
                        duck_irl_pose = [-1] * 2
                        for i in range(2):
                            for j in range(len(hv_lines[i])-1):
                                if duck_irl_pose[i] == -1:
                                    p_base = dist(duck_pose,hv_lines[i][j])
                                    p_top = dist(duck_pose,hv_lines[i][j+1])
                                    if get_sign(p_base) != get_sign(p_top):
                                        duck_irl_pose[i] = j + offset_accum[i] + abs(p_base)/(abs(p_base) + abs(p_top))
                        print(duck_irl_pose)
                        # print(duck_irl_pose[0]*10+duck_shape[0]/2,duck_irl_pose[1]*10+duck_shape[1]/2)
                        duck_circle = [duck_irl_pose[0]*10+duck_shape[0]/2, duck_irl_pose[1]*10+duck_shape[1]/2]
                        cv2.circle(duck_frame,(int(duck_circle[0]), int(duck_circle[1])),1,(0,255,255),5)
                        # for j in range(len(hv_lines[1])-1):

                    
                    for i in point_reg:
                        # cv2.circle(line_frame,(duck_pose[0],duck_pose[1]),2,(0,255,255),20)
                        cv2.putText(line_frame, str(i.lines[0])+','+str(i.lines[1]), (int(i.x), int(i.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                        # cv2.putText(line_frame, str(i.id), (int(i.x) + 20, int(i.y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                        last_point_reg = point_reg.copy()

        # print(len(line_frame), len(line_frame[0]))
        cv2.imshow('LineFrame',line_frame)
        cv2.imshow('DuckFrame',duck_frame)
        cv2.imshow('Frame',frame_org)
        # time.sleep(.5)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
    else:
        break

vid_capture.release()
cv2.destroyAllWindows()
