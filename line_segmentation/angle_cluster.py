#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import math
from numpy import linalg
import time
import cv2
from segmentation_functions import *
from sklearn.cluster import KMeans

angles = [1.4663817088823943, 1.4308998411136689, -0.5239586404160237, 
          1.3090828243936916, 1.4663978569276719, -0.41776147379187023, 
          1.37894087646879, -0.2791938465462227, -0.6090871589077729]

line_ang_coords = []
for i in angles:
    line_ang_coords.append([math.cos(i), math.sin(i)])

kmeans = KMeans(n_clusters=2)
kmeans.fit(line_ang_coords)

arr = np.array(line_ang_coords)

print(kmeans.labels_)
print(arr)
plt.scatter(arr[:,0], arr[:,1], c=kmeans.labels_)
plt.show()