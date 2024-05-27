#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import cv2

def plot_line(lin):
    plt.plot([lin[0],lin[2]], [lin[1],lin[3]],'-')

a = [1379, 380, 1875, 715]
b = [1199,267,1404, 405]

plot_line(a)
plot_line(b)
plt.show()