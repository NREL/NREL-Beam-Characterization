import pandas as pd
import numpy as np
import os
from BCS_functions import BCS_functions

def find_centroid_wrapper(img_path: str, visualization:bool = False) -> np.ndarray:
    """
    Wrapper function to find the centroid of the image
    """
    img, img_centroid = BCS_functions.load_image(img_path)
    corners = BCS_functions.find_corner_candidates(img)
    valid_corners = BCS_functions.valid_intersections(corners, img.shape)
    rectified_img = BCS_functions.rectify_and_crop(img_centroid, valid_corners)
    rectified_img_filtered = BCS_functions.low_pass_filter(rectified_img, keep_ratio=0.02)
    rectified_img_gamma_filtered = BCS_functions.gamma_correction(rectified_img_filtered, 5)

    centroid_location = BCS_functions.find_centroid(rectified_img_gamma_filtered)

    if visualization:
        import matplotlib.pyplot as plt
        plt.imshow(rectified_img_gamma_filtered, cmap='gray')
        plt.scatter(centroid_location[0], centroid_location[1], c='r', s=100)
        plt.show()

    return centroid_location
if __name__ == "__main__":
    root_folder = r"C:\Users\qzheng\OneDrive - NREL\BCS Comparison"
    rr_xlsx_path = os.path.join(root_folder, "BCS_RR_Library_v1.xlsx")
    rr_test_images = pd.read_excel(rr_xlsx_path, sheet_name="TestImages")
    # valid entried are those with non-empty image path
    valid_entries = rr_test_images.dropna(subset=["ImagePath"])
    
    for index, row in valid_entries.iterrows():
        img_path = os.path.join(root_folder, row["ImagePath"])
        centroid_location = find_centroid_wrapper(img_path, True)

        # 500 is the center pixel location since the orignial image is scaled to 1000x1000
        err_x_px = centroid_location[0] - 500
        err_y_px = centroid_location[1] - 500
        px_to_m_ratio = row["TargetW"] / 1000
        