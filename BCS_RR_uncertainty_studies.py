from functools import partial
import pandas as pd
import numpy as np
import os
from BCS_functions import BCS_functions
import cv2

# Uncertainty have many sources
# 1. Corners
# 2. Camera position
# 3. Target position
# 4. Optical errors
# 5. Resolution of cameras

# img_filter_function = partial(BCS_functions.low_pass_filter, keep_ratio=0.02)     # Low fass filtering
img_filter_function = partial(cv2.medianBlur, ksize=35)   

def centroid_location_with_corner_uncertianties(img_path: str, source_name: str, uncertainties: dict, visualiztion: bool = False):
    """ A wrapper function to calculate the tracking error with uncertainties
    Args:
        img_path: str: path to the image
        source_name: str: name of the source
        uncertainties: dict: dictionary of uncertainties, with names of uncertainties as they key

    """

    img_, img_for_centroid_calculation_ = BCS_functions.load_image(img_path)
    img = img_.copy()

    # Add noises to corners
    num_samples = 100
    corner_variance = uncertainties['Corner stdv']
    np.random.seed(42)
    mean = 0
    corner_1_noise = np.random.normal(mean, corner_variance, (num_samples, 2))
    corner_2_noise = np.random.normal(mean, corner_variance, (num_samples, 2))
    corner_3_noise = np.random.normal(mean, corner_variance, (num_samples, 2))
    corner_4_noise = np.random.normal(mean, corner_variance, (num_samples, 2))
    
    if source_name == "CENER":
        corners = BCS_functions.find_corner_candidates(img)
        valid_corners = BCS_functions.valid_intersections(corners, img.shape)   # valid corners is a list of 4 tuples for corners
    else:
        # For other sources, the corners are simply the 4 corners of the image
        valid_corners = [(0, 0), (img.shape[1], 0), (img.shape[1], img.shape[0]), (0, img.shape[0])]

    centroid_location_buffer = []
    for i in range(num_samples):
        noisy_corners = valid_corners.copy()
        noisy_corners[0] += corner_1_noise[i]
        noisy_corners[1] += corner_2_noise[i]
        noisy_corners[2] += corner_3_noise[i]
        noisy_corners[3] += corner_4_noise[i]

        noisy_corners[0] = tuple(noisy_corners[0].astype(int))
        noisy_corners[1] = tuple(noisy_corners[1].astype(int))
        noisy_corners[2] = tuple(noisy_corners[2].astype(int))
        noisy_corners[3] = tuple(noisy_corners[3].astype(int))
        rectified_img = BCS_functions.rectify_and_crop(img_for_centroid_calculation_, noisy_corners)
        rectified_img_filtered = img_filter_function(rectified_img)
        rectified_img_gamma_filtered = BCS_functions.gamma_correction(rectified_img_filtered, 7)
        centroid_location = BCS_functions.find_centroid(rectified_img_gamma_filtered)
        centroid_location_buffer.append(centroid_location)

    centroid_location_buffer = np.array(centroid_location_buffer)
    return centroid_location_buffer

def tracking_error_finder(heliostat_position, target_position, centroid_buffer):
    ideal_vector = target_position - heliostat_position

    horizontal_angular_error_buffer = []
    vertical_angular_error_buffer = []
    combined_angular_error_buffer = []

    for i in range(centroid_buffer.shape[0]):
        measured_vector = ideal_vector.copy()
        measured_vector[0] += centroid_buffer[i, 0]
        measured_vector[1] += centroid_buffer[i, 1]

        dot_product = np.dot(ideal_vector, measured_vector)
        norm_ideal = np.linalg.norm(ideal_vector)
        norm_measured = np.linalg.norm(measured_vector)
        cos_theta = dot_product / (norm_ideal * norm_measured)
        theta = np.arccos(cos_theta)
        combined_angular_error_buffer.append(theta)

        # For calculation of X-only errors
        measured_vector_x_only = ideal_vector.copy()
        measured_vector_x_only[0] += centroid_buffer[i, 0]
        norm_measured_x_only = np.linalg.norm(measured_vector_x_only)
        dot_product = np.dot(ideal_vector,measured_vector_x_only)
        cos_theta = dot_product / (norm_ideal * norm_measured_x_only)
        theta = np.arccos(cos_theta)
        # Add sign to error
        if measured_vector_x_only[0] > ideal_vector[0]:
            theta = theta
        else:
            theta = -theta
        horizontal_angular_error_buffer.append(theta)

        # For calculation of Y-only errors
        measured_vector_z_only = ideal_vector.copy()
        measured_vector_z_only[2] += centroid_buffer[i, 1]
        norm_measured_z_only = np.linalg.norm(measured_vector_z_only)
        dot_product = np.dot(ideal_vector, measured_vector_z_only)
        cos_theta = dot_product / (norm_ideal * norm_measured_z_only)
        theta = np.arccos(cos_theta)
        if measured_vector_z_only[2] > ideal_vector[2]:
            theta = theta
        else:
            theta = -theta
        vertical_angular_error_buffer.append(theta)

    average_combined_tracking_error = np.mean(combined_angular_error_buffer)
    average_horizontal_tracking_error = np.mean(horizontal_angular_error_buffer)
    average_vertical_tracking_error = np.mean(vertical_angular_error_buffer)

    combined_error_uncertainty = np.std(combined_angular_error_buffer)
    horizontal_error_uncertainty = np.std(horizontal_angular_error_buffer)
    vertical_error_uncertainty = np.std(vertical_angular_error_buffer)

    return (average_combined_tracking_error, combined_error_uncertainty), (average_horizontal_tracking_error, horizontal_error_uncertainty), (average_vertical_tracking_error, vertical_error_uncertainty)
    

def tracking_error_with_uncertainties(heliostat_position: np.ndarray, target_position: np.ndarray, center_measurements: np.ndarray):
    """
    Produces the tracking error with uncertainties, uncertainty sources are comming from the center measurements

    Args:
        heliostat_position: The location of the heliostat
        target_position: The location of the target
        center_measurements: The measured center of the target with respect to the heliostat. This is a 3D array, the first dimension is the number of measurements

    Returns:
        Three tuples:
        First tuple (combined tracking error, uncertainty)
        Second tuple (horizontal tracking error, uncertainty)
        Third tuple (vertical tracking error, uncertainty)
    """
    
    ideal_vector = target_position - heliostat_position

    horizontal_angular_error_buffer = []
    vertical_angular_error_buffer = []
    combined_angular_error_buffer = []

    for i in range(center_measurements.shape[0]):
        measured_vector = ideal_vector.copy()
        measured_vector[0] += center_measurements[i, 0]    # x measruement off
        measured_vector[2] += center_measurements[i, 1]    # z measurement off

        dot_product = np.dot(ideal_vector, measured_vector)
        norm_ideal = np.linalg.norm(ideal_vector)
        norm_measured = np.linalg.norm(measured_vector)
        cos_theta = dot_product / (norm_ideal * norm_measured)
        theta = np.arccos(cos_theta)
        combined_angular_error_buffer.append(theta)

        # For calculation of X-only errors
        measured_vector_x_only = ideal_vector.copy()
        measured_vector_x_only[0] += center_measurements[i, 0]
        norm_measured_x_only = np.linalg.norm(measured_vector_x_only)
        dot_product = np.dot(ideal_vector,measured_vector_x_only)
        cos_theta = dot_product / (norm_ideal * norm_measured_x_only)
        theta = np.arccos(cos_theta)
        # Add sign to error
        if measured_vector_x_only[0] > ideal_vector[0]:
            theta = theta
        else:
            theta = -theta
        horizontal_angular_error_buffer.append(theta)

        # For calculation of Y-only errors
        measured_vector_z_only = ideal_vector.copy()
        measured_vector_z_only[2] += center_measurements[i, 1]
        norm_measured_z_only = np.linalg.norm(measured_vector_z_only)
        dot_product = np.dot(ideal_vector, measured_vector_z_only)
        cos_theta = dot_product / (norm_ideal * norm_measured_z_only)
        theta = np.arccos(cos_theta)
        if measured_vector_z_only[2] > ideal_vector[2]:
            theta = theta
        else:
            theta = -theta
        vertical_angular_error_buffer.append(theta)

    average_combined_tracking_error = np.mean(combined_angular_error_buffer)
    average_horizontal_tracking_error = np.mean(horizontal_angular_error_buffer)
    average_vertical_tracking_error = np.mean(vertical_angular_error_buffer)

    combined_error_uncertainty = np.std(combined_angular_error_buffer)
    horizontal_error_uncertainty = np.std(horizontal_angular_error_buffer)
    vertical_error_uncertainty = np.std(vertical_angular_error_buffer)

    return (average_combined_tracking_error, combined_error_uncertainty), (average_horizontal_tracking_error, horizontal_error_uncertainty), (average_vertical_tracking_error, vertical_error_uncertainty)
  
        

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

        centroid_location_with_noisy_corners = centroid_location_with_corner_uncertianties(img_path, row["Source"], uncertainties={"Corner stdv": 10})
        if row["Source"] == "Sandia":
            row["TargetW"] = 12
        elif row["Source"] == "CENER":
            row["HeliostatX"] = 0
            row["HeliostatY"] = 0
            row["HeliostatZ"] = 0

            row["TargetX"] = 0
            row["TargetY"] = 201
            row["TargetZ"] = 9.975

        error_x_px = centroid_location_with_noisy_corners[:, 0] - 500
        error_y_px = centroid_location_with_noisy_corners[:, 1] - 500
        px_to_m_ratio = row["TargetW"] / 1000


        heliostat_pos = np.array([row["HeliostatX"], row["HeliostatY"], row["HeliostatZ"]])
        target_pos = np.array([row["TargetX"], row["TargetY"], row["TargetZ"]])
        combined_tracking_error_distribution, horizontal_tracking_error_distribution, vertical_tracking_error_distribution \
              = tracking_error_with_uncertainties(heliostat_pos, target_pos, centroid_location_with_noisy_corners)
        