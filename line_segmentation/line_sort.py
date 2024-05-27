#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import math
from numpy import linalg
import time
import cv2
from segmentation_functions import *
from sklearn.cluster import KMeans

ax = [0, 1000, 0, 500]
ax_2 = [-ax[1]/3, ax[1]*4/3, -ax[3]/3, ax[3]*4/3]
shape = (ax[3], ax[1])

lines = [
[200,500,300,400],
[0,0,1000,100],
[800,200,0,100],
]

distss = line_sort(lines)

# Create empty frame
line_frame = np.full((shape[0], shape[1], 3) , (0, 0, 0), np.uint8)

for i in range(len(lines)):
  for x1,y1,x2,y2 in [lines[i]]:
      cv2.line(line_frame,(x1,y1),(x2,y2),(255,0,0),2)
      md = midpoint(lines[i])
      cv2.putText(line_frame, str(distss[i]), (int(md[0]), int(md[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imwrite('image_alpha.png', line_frame)
