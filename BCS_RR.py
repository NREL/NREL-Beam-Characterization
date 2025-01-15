import pandas as pd
import numpy as np
import os
from BCS_functions import BCS_functions
import cv2

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


def tracking_error_finder(heliostat_position: np.ndarray, target_position: np.ndarray, center_measurements: np.ndarray):
    ideal_vector = target_position - heliostat_position
    measured_vector = ideal_vector.copy()
    measured_vector[0] += center_measurements[0]    # x measruement off
    measured_vector[2] += center_measurements[1]    # z measurement off
    cos_theta = np.dot(ideal_vector, measured_vector) / (np.linalg.norm(ideal_vector) * np.linalg.norm(measured_vector))
    theta = np.arccos(cos_theta)

    return theta   
    

    

if __name__ == "__main__":
    visualization = True
    root_folder = r"C:\Users\qzheng\OneDrive - NREL\BCS Comparison"
    rr_xlsx_path = os.path.join(root_folder, "BCS_RR_Library_v1.xlsx")
    rr_test_images = pd.read_excel(rr_xlsx_path, sheet_name="TestImages")
    # valid entried are those with non-empty image path
    valid_entries = rr_test_images.dropna(subset=["ImagePath"])
    
    for index, row in valid_entries.iterrows():
        img_path = os.path.join(root_folder, row["ImagePath"], row["ImageName"])

        # PSA data does not need corner finding
        if row["Source"] == "PSA":
            _, img = BCS_functions.load_image(img_path)
            img = cv2.resize(img, (1000, 1000))
            
            img_gamma = BCS_functions.gamma_correction(img, 5)
            centroid_location = BCS_functions.find_centroid(img_gamma)

            if visualization:
                import matplotlib.pyplot as plt
                plt.imshow(img_gamma, cmap='gray')
                plt.scatter(centroid_location[0], centroid_location[1], c='r', s=50)
                plt.title(row["ImageName"])
                plt.show()

            else:
                centroid_location = find_centroid_wrapper(img_path, visualization=visualization)

        # 500 is the center pixel location since the orignial image is scaled to 1000x1000
        err_x_px = centroid_location[0] - 500
        err_y_px = centroid_location[1] - 500
        px_to_m_ratio = row["TargetW"] / 1000

        heliostat_pos = np.array([row["HeliostatX"], row["HeliostatY"], row["HeliostatZ"]])
        target_pos = np.array([row["TargetX"], row["TargetY"], row["TargetZ"]])
        center_measurements = np.array([err_x_px * px_to_m_ratio, err_y_px * px_to_m_ratio])

        tracking_error = tracking_error_finder(heliostat_pos, target_pos, center_measurements)
        print(f"Traking error for {row['ImageName']}: {tracking_error*1000:.2f} mrad")
        