#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import math
from numpy import linalg
import cv2

def get_ang(line):
    return math.atan2(line[2] - line[0], line[3] - line[1])

def angle_diff(ang1, ang2):
    dif = (ang1 - ang2 + math.pi) % (2*math.pi) - math.pi
    return dif + 2 * math.pi if dif < -math.pi else dif

def dist(p, l):
    p1 = np.array([l[0], l[1]])
    p2 = np.array([l[2], l[3]])
    p3 = np.array(p)
    return abs(np.cross(p2-p1, p1-p3)/linalg.norm(p2-p1))

def sim_line(l1, l2, ang_tres, dist_tres):
    a1 = get_ang(l1)
    a2 = get_ang(l2)
    ang_d = abs(angle_diff(a1, a2))
    ang_d = min(ang_d, math.pi - ang_d)

    d1 = dist([l1[0],l1[1]], l2)
    d2 = dist([l1[2],l1[3]], l2)
    d3 = dist([l2[0],l2[1]], l1)
    d4 = dist([l2[2],l2[3]], l1)
    d_min = min(d1, d2, d3, d4)

    if(ang_d < ang_tres and d_min < dist_tres):
        return True
    return False

def extend(line, lim):
    for x1,y1,x2,y2 in [line]:
        m = (y2 - y1) / (x2 - x1)

        if(m == -math.inf or m == math.inf):
            y1 = lim[2]
            y2 = lim[3]
            return [int(x1),int(y1),int(x2),int(y2)]
        
        while(x1-1 > lim[0] and y1-m > lim[2]):
            x1 = x1 - 1
            y1 = y1 - m
        while(x2+1 < lim[1] and y2+m < lim[3]):
            x2 = x2 + 1
            y2 = y2 + m

        return [x1,y1,x2,y2]
    
def inter(l1, l2):
    m1 = (l1[3] - l1[1]) / (l1[2] - l1[0])
    b1 = l1[1] - m1 * l1[0]
    m2 = (l2[3] - l2[1]) / (l2[2] - l2[0])
    b2 = l2[1] - m2 * l2[0]
    x = (b2 - b1) - (m1 - m2)
    y = m1 * x + b1
    return [x,y]

ang_t = 0.3
dist_t = 50
ax = [0, 1911, 0, 1077]

# Read in the image
image = cv2.imread('/home/max/yolo_training/ducks_img3.png')
# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Define our parameters for Canny
low_threshold = 0
high_threshold = 100
edges = cv2.Canny(gray, low_threshold, high_threshold)
plt.imshow(edges, cmap='gray')
# plt.show()

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1
theta = np.pi/180
threshold = 100
min_line_length = 10
max_line_gap = 100
line_image = np.copy(image) #creating an image copy to draw lines on
# Run Hough on the edge-detected image
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
# Iterate over the output "lines" and draw lines on the image copy

lines_reg = []
# for line in lines:
#     for x1,y1,x2,y2 in line:
#         cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),2)
for line in lines:
    l = line[0]
    # print(l)
    # print(len(l))
    # print(type(l))
    rept_line = False
    for lin in lines_reg:
        if(sim_line(lin, l, ang_t, dist_t)):
            rept_line = True
    if not rept_line:
        lines_reg.append(l)
        line_ext = [[int(ex) for ex in extend(l, ax)]]
        for x1,y1,x2,y2 in line_ext:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)
# plt.imshow(line_image)
# plt.show()

for i in range(len(lines_reg) - 1):
    for j in range(i+1, len(lines_reg)):
        [x, y] = inter(lines_reg[i], lines_reg[j])
        if x >= ax[0] and x <= ax[1] and y >= ax[2] and y <= ax[3]:
            cv2.circle(line_image,(int(x),int(y)),5,(0,255,255),5)
        
plt.imshow(line_image)
plt.show()