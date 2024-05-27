#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import math
from numpy import linalg
import time
import cv2
from segmentation_functions import *
from sklearn.cluster import KMeans

ax = [0, 523, 0, 432]
ax_2 = [-ax[1]/3, ax[1]*4/3, -ax[3]/3, ax[3]*4/3]
shape = (ax[3], ax[1])

lines = [[39,431,353,14],
[56,430,502,0],
[201,0,503,431],
[13,352,319,0],
[0,258,194,0],
[40,420,353,4],
[89,390,440,51],
[97,12,329,431],
[17,355,326,0],
[36,430,359,1],
[81,404,470,28],
[220,427,521,191],
[38,416,349,3],
[47,422,353,16],
[67,412,423,68],
[1,264,190,12],
[38,418,320,44],
[1,152,113,430],
[42,420,338,27],
[99,9,326,419],
[15,30,208,427],
[0,369,321,0],
[96,381,431,58],
[99,17,327,429],
[16,355,322,3],
[0,143,96,382],
[212,5,404,279],
[49,430,470,24],
[14,31,208,429],
[10,18,178,362],
[203,1,474,388],
[434,67,522,161],
[1,154,113,431]]


lines_reg = []
line_ang_coords = []
point_reg = []
ortho_lines = []

# Create empty frame
line_frame = np.full((shape[0], shape[1], 3) , (0, 0, 0), np.uint8)

if lines is not None:
    # Remove repeated lines
    ang_treshold = 0.2
    dist_treshold = 20
    for line in lines:
        l = line
        rept_line = False
        for lin in lines_reg:
            if(is_line_similar(lin, l, ang_treshold, dist_treshold)):
                rept_line = True
        if not rept_line:
            lines_reg.append(l)
            line_ext = [extend_line(l, ax)]
            for x1,y1,x2,y2 in line_ext:
                cv2.line(line_frame,(x1,y1),(x2,y2),(255,0,0),2)

    # Keep only orthogonal lines
    # if frame_count > skew_line_start and frame_count < skew_line_end and len(lines_reg) > 1:
    if len(lines_reg) > 1:
        clust = get_clusters(lines_reg)
        h_lines = []
        v_lines = []
        hv_lines = [h_lines, v_lines]
        for i in range(len(clust)):
            hv_lines[clust[i]].append(lines_reg[i])
        
        # for k in hv_lines:
        #     for i in range(len(k) - 1):
        #         for j in range(i+1, len(k)):
        #             intersects = False
        #             [x, y] = [int(j) for j in intersect(k[i], k[j])]
        #             if x >= ax[0] and x <= ax[1] and y >= ax[2] and y <= ax[3]:
        #                 intersects = True
        #             if not intersects:
        #                 ortho_lines.append(k[i])
    
        # Sort (horizontal and vertical) lines by position 
        hv_count = 0
        hv_ave_lines = []
        hv_idxs = []
        for i in hv_lines:
            ave_line, line_idxs = line_sort(i)
            hv_ave_lines.append(ave_line)
            hv_idxs.append(line_idxs)
            half_idx = line_idxs.index((len(i) - 1) // 2)
            # for j in range(len(i)):
            #     for x1,y1,x2,y2 in [i[j]]:
            #         cv2.line(line_frame,(x1,y1),(x2,y2),(255,0,0),2)
            #         md = midpoint(i[j])
            #         cv2.putText(line_frame, str(line_idxs[j]), (int(md[0]), int(md[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255 * hv_count, 255), 2, cv2.LINE_AA)
            # hv_count += 1

        # Draw points of intersection
        # print(hv_lines)
        # print(hv_idxs)
        keypoint_count = 0
        tmp_point_reg = []
        for i in range(len(hv_idxs[0])):
            for j in range(len(hv_idxs[1])):
                l1 = h_lines[hv_idxs[0].index(i)]
                l2 = v_lines[hv_idxs[1].index(j)]
                [x, y] = [int(k) for k in intersect(l1, l2)]
                if x >= ax[0] and x <= ax[1] and y >= ax[2] and y <= ax[3]:
                    keypoint_count += 1
                    new_point = Point(keypoint_count, x, y, [l1, l2])
                    tmp_point_reg.append(new_point)
                    
                    cv2.circle(line_frame,(x,y),1,(0,0,255),10)
                    cv2.putText(line_frame, str(keypoint_count), (int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        
        for k in range(len(tmp_point_reg) - 1):
            for l in range(k + 1, len(tmp_point_reg)):
                does_share, n = share_lines(tmp_point_reg[k], tmp_point_reg[l])
                if does_share:
                    if n:
                        [tmp_point_reg[k], tmp_point_reg[l]] = point_vertical_relation(tmp_point_reg[k], tmp_point_reg[l])
                    else:
                        [tmp_point_reg[k], tmp_point_reg[l]] = point_horizontal_relation(tmp_point_reg[k], tmp_point_reg[l])
        print("je"*10)
        for i in tmp_point_reg:
            i.display()

cv2.imwrite('image_alpha.png', line_frame)

# plt.axis(ax)
# plt.show()
