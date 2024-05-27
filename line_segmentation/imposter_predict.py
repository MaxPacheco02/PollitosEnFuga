#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import math
from numpy import linalg
import time
import cv2
from segmentation_functions import *
from sklearn.cluster import KMeans
from scipy import stats

ax = [0, 523, 0, 432]
ax_2 = [-ax[1]/3, ax[1]*4/3, -ax[3]/3, ax[3]*4/3]
shape = (ax[3], ax[1])

# Create empty frame
line_frame = np.full((shape[0], shape[1], 3) , (0, 0, 0), np.uint8)

lines = [
[39, 431, 353, 14],
[56, 430, 502, 0],
[201, 0, 503, 431],
[13, 352, 319, 0],
[0, 258, 194, 0],
[97, 12, 329, 431],
[220, 427, 521, 191],
[1, 152, 113, 430],
[15, 30, 208, 427],
[434, 67, 522, 161]
]

def myfunc(x):
  return slope * x + intercept

# sort the lines 'in order', from left to right
# calculate 
# for line in lines:
#     for x1,y1,x2,y2 in [line]:
#         cv2.line(line_frame,(x1,y1),(x2,y2),(255,255,255),3)

# let's pretend we have the angles of the lines sorted out in order...

y = [50, 51, 52.0, 52.1, 53]
print(y)

for i in range(len(y)):
    tmp = y.copy()
    tmp.remove(y[i])
    x = range(len(tmp))

    # print(x,tmp)

    slope, intercept, r, p, std_err = stats.linregress(x, tmp)

    print(r)

# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.show()

# cv2.imwrite('image_alpha.png', line_frame)