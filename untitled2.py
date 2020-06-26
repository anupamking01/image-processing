#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:43:10 2019

@author: aman
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('messi.jpg',0)
plt.imshow(img, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()