#!/usr/bin/env python3

import numpy as np
import math
from numpy import linalg
from matplotlib import pyplot as plt
import time
import cv2
from segmentation_functions import *
from sklearn.cluster import KMeans

last_point_list = []

def share_ls(p1, p2):
    for i in range(2):
        for j in range(2):
            if p1.lines[i] == p2.lines[j]:
                return True, i
    return False, -1

def set_fb_relation(l, i, j):
    if l[i].fwd == -1 and l[j].back == -1:
        l[i].fwd = l[j].id
        l[j].back = l[i].id

def set_lr_relation(l, i, j):
    if l[i].left == -1 and l[j].right == -1:
        l[i].left = l[j].id
        l[j].right = l[i].id

def set_hv_relations(p_list):
    for i in range(len(p_list) - 1):
        for j in range(i + 1, len(p_list)):
            does_share, ax = share_ls(p_list[i], p_list[j])
            if does_share:
                if ax:
                    set_fb_relation(p_list, i, j)
                else:
                    set_lr_relation(p_list, i, j)

# Fill point list
# list = [
#     [1,6],
#     [1,7],
#     [2,6],
#     [2,7],
#     [2,8],
#     [3,6],
#     [3,7],
#     [3,8],
#     [3,9],
#     [4,6],
#     [4,7],
#     [4,8],
#     [4,9],
#     [4,10],
#     [5,6],
#     [5,7],
#     [5,8],
#     [5,9],
#     [5,10]]

# 'old'
# list = [
#     [0,3],
#     [0,4],
#     [0,5],
#     [1,3],
#     [1,4],
#     [1,5],
#     [2,3],
#     [2,4],
#     [2,5],
# ]
# # Fill list of points
# point_list = []
# point_list2 = []
# for i in range(len(list)):
#     tmp_point = Point(i+1, list[i][0], list[i][1], list[i])
#     tmp_point2 = Point(i+1, list[i][0] + 0.1, list[i][1] + 0.1, list[i])
#     point_list.append(tmp_point)
#     point_list2.append(tmp_point2)

# 'tmp'
list1 = [
    [0,2],
    [0,3],
    [1,2],
    [1,3]
]
list2 = [
    [0,2],
    [0,3]
    [1,2],
    [1,3],
]
# Fill list of points
point_list = []
point_list2 = []
for i in range(len(list1)):
    tmp_point = Point(i+1, list1[i][0], list1[i][1] - 2, list1[i])
    tmp_point2 = Point(i+10, list2[i][0] + 0.05, list2[i][1] - 2 + 0.1, list2[i])
    point_list.append(tmp_point)
    point_list2.append(tmp_point2)

# 'tmp'
       
# Build relations for all the elements
set_hv_relations(point_list)
set_hv_relations(point_list2)

print('\n#1')
for i in point_list:
    i.display()

# Find transform
found_rot = False
if len(point_list) > 1:
    for i in range(len(point_list)):
        for j in range(len(point_list2)):
            if not found_rot:
                shares_adj, rotation_rel = share_adj([i,j],[point_list,point_list2])
                if same_point(point_list[i], point_list2[j]) and shares_adj:
                    found_rot = True
                    trans_rel = get_trans_rel(point_list[i], point_list2[j])

# Transform new frame
print("\nrotation relation", rotation_rel)
if found_rot:
    for i in point_list2:
        point_rotate(i, rotation_rel)
        point_translate(i, trans_rel)

# Adapt new point list to the last one
# for all points in last list
    # for all points in current list
        # if same coords and sharing a vertical and horizontal id:
            # find the coords of adjacent points
            # adapt current list's orientation to match last's
            # translate current list's common point to match last's coords
            # translate whole current list
            # update last_point_list
print('\n#2 POST')
for i in point_list2:
    i.display()
