from functools import partial
import pandas as pd
import numpy as np
import os
from BCS_functions import BCS_functions
import cv2
from glob import glob

from BCS_RR import find_centroid_wrapper

GAUSSIAN_NOISE_MEAN = 0
GAUSSIAN_NOISE_STDV = 0.1   # 5% of the max value
NOISE_ITERATIONS = 100

if __name__ == "__main__":
    # img_folder = r"C:\Users\qzheng\OneDrive - NREL\BCS Comparison\CENER\data\raw_input\raw_input\CAT\03_22_2023\images"
    # img_folder = r"C:\Users\qzheng\OneDrive - NREL\BCS Comparison\CENER\data\raw_input\raw_input\CAT\04_03_2021\images\Images_1"
    img_folder = r"C:\Users\qzheng\OneDrive - NREL\BCS Comparison\CENER\data\raw_input\raw_input\CAT\11_06_2023\images"
    img_paths = glob(os.path.join(img_folder, "*.tif"))

    for img_path in img_paths:
        find_centroid_wrapper(img_path, visualization=True)