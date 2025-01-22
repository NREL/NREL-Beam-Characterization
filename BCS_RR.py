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
    # rectified_img_filtered = cv2.medianBlur(rectified_img, 35)
    rectified_img_gamma_filtered = BCS_functions.gamma_correction(rectified_img_filtered, 7)

    centroid_location = BCS_functions.find_centroid(rectified_img_gamma_filtered)

    if visualization:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2)
        axs[1].imshow(rectified_img, cmap='gray')
        axs[1].set_title("Original image")
        axs[1].axis('off')
        axs[0].imshow(rectified_img_gamma_filtered, cmap='gray')
        axs[0].scatter(centroid_location[0], centroid_location[1], c='r', s=20)
        axs[0].set_title("Gamma corrected image")
        axs[0].axis('off')
        plt.show()

    return centroid_location


def tracking_error_finder(heliostat_position: np.ndarray, target_position: np.ndarray, center_measurements: np.ndarray):
    """
    Relies on the dot product to find 
    """
    
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
        if os.path.exists(img_path) is False:
            print(f"Image {img_path} does not exist")
            continue

        # PSA data does not need corner finding
        if row["Source"] == "PSA":
            _, img = BCS_functions.load_image(img_path)
            img = cv2.resize(img, (1000, 1000))
            
            img_gamma = BCS_functions.gamma_correction(img, 2)
            centroid_location = BCS_functions.find_centroid(img_gamma)

            if visualization:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(img_gamma, cmap='gray')
                axs[0].scatter(centroid_location[0], centroid_location[1], c='r', s=20)
                axs[0].set_title("Gamma corrected image")
                axs[0].axis('off')
                axs[1].imshow(img, cmap='gray')
                axs[1].scatter(centroid_location[0], centroid_location[1], c='r', s=20)
                axs[1].set_title("Original image")
                axs[1].axis('off')

                plt.show()
        # Sandia data does not need borner finding, the excel sheet does not have targer width either
        elif row["Source"] == "Sandia":
            _, img = BCS_functions.load_image(img_path)
            img = cv2.resize(img, (1000, 1000))
            img_low_pass = BCS_functions.low_pass_filter(img, keep_ratio=0.1)
            # img_low_pass = cv2.medianBlur(img, 35)
            img_gamma = BCS_functions.gamma_correction(img_low_pass, 2)
            centroid_location = BCS_functions.find_centroid(img_gamma)

             # TODO: Change this number, just made up since there's no data
            row["TargetW"] = 12

            if visualization:
                import matplotlib.pyplot as plt
                # 2 subplot
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(img_gamma, cmap='gray')
                axs[0].scatter(centroid_location[0], centroid_location[1], c='r', s=20)
                axs[0].set_title("Gamma corrected image")
                axs[0].axis('off')
                axs[1].imshow(img, cmap='gray')
                axs[1].scatter(centroid_location[0], centroid_location[1], c='r', s=20)
                axs[1].set_title("Original image")
                axs[1].axis('off')

                plt.show()

        elif row["Source"] == "CENER":
            # CENER data uses latitude and longtiude to find the centroid
            # Here I manually converted the data into distances and hard coded 
            row["HeliostatX"] = 0
            row["HeliostatY"] = 0
            row["HeliostatZ"] = 0

            row["TargetX"] = 0
            row["TargetY"] = 201
            row["TargetZ"] = 9.975
            centroid_location = find_centroid_wrapper(img_path, visualization=visualization)

        else:
            raise ValueError(f"Unknown source {row['Source']}")
        
        # 500 is the center pixel location since the orignial image is scaled to 1000x1000
        err_x_px = centroid_location[0] - 500
        err_y_px = centroid_location[1] - 500
        px_to_m_ratio = row["TargetW"] / 1000

        heliostat_pos = np.array([row["HeliostatX"], row["HeliostatY"], row["HeliostatZ"]])
        target_pos = np.array([row["TargetX"], row["TargetY"], row["TargetZ"]])
        center_measurements = np.array([err_x_px * px_to_m_ratio, err_y_px * px_to_m_ratio])

        tracking_error = tracking_error_finder(heliostat_pos, target_pos, center_measurements)
        print(f"Traking error for {row['ImageName']}: {tracking_error*1000:.2f} mrad")
        