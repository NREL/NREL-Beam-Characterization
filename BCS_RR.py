from functools import partial
import pandas as pd
import numpy as np
import os
from BCS_functions import BCS_functions
import cv2
import matplotlib.pyplot as plt

GAUSSIAN_NOISE_MEAN = 0
GAUSSIAN_NOISE_STDV = 0.05   # 5% of the max value
NOISE_ITERATIONS = 10

# img_filter_function = partial(BCS_functions.low_pass_filter, keep_ratio=0.02)     # Low fass filtering
img_filter_function = partial(cv2.medianBlur, ksize=35)         # Median blue is used for median filter 

def find_centroid_wrapper(img_path: str, source_name: str, visualization:bool = False) -> np.ndarray:
    """
    Wrapper function to find the centroid of the image, works for any images that may require a corner finding functionality
    
    Args:
        img_path: The path to the image
        source_name: The name of the source, current are limited to "PSA", "Sandia", and "CENER". CENER data requires image corpping
        visualization: Whether to visualize the centroid finding process

    Returns:
        The centroid location and the buffer of the centroid location
    
    """
    img_, img_centroid_ = BCS_functions.load_image(img_path)
    centroid_buffer = []

    # The iteration put random noises on the image to get uncertainties
    for i in range(NOISE_ITERATIONS):
        img = img_.copy()
        noise = np.zeros(img.shape, np.uint8)
        cv2.randn(noise, GAUSSIAN_NOISE_MEAN, 2)
        img = cv2.add(img, noise)

        if source_name == "CENER":
            corners = BCS_functions.find_corner_candidates(img)
            valid_corners = BCS_functions.valid_intersections(corners, img.shape)
            rectified_img = BCS_functions.rectify_and_crop(img_centroid_, valid_corners)
            original_target_region_for_show = rectified_img.copy()  # only for visualization
            rectified_img_filtered = img_filter_function(rectified_img) # for removing patterns on the target
            img_to_process = rectified_img_filtered
        else:
            img_to_process = img
            original_target_region_for_show = img_to_process.copy()

        noise_2 = np.zeros(img_to_process.shape, np.uint8)
        cv2.randn(noise_2, GAUSSIAN_NOISE_MEAN, GAUSSIAN_NOISE_STDV * np.max(img_to_process))
        img_to_process = cv2.add(img_to_process, noise_2)
        img_to_process_gamma_filtered = BCS_functions.gamma_correction(img_to_process, 7)
        

        centroid_location = BCS_functions.find_centroid(img_to_process_gamma_filtered)
        centroid_buffer.append(centroid_location)
    centroid_location = np.mean(centroid_buffer, axis=0)
    centroid_location_stdv = np.std(centroid_buffer, axis=0)


    if visualization:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2)
        axs[1].imshow(original_target_region_for_show, cmap='gray')
        axs[1].set_title("Original image")
        axs[1].axis('off')
        axs[0].imshow(img_to_process_gamma_filtered, cmap='gray')
        axs[0].scatter(centroid_location[0], centroid_location[1], c='r', s=20)
        axs[0].set_title("Gamma corrected image")
        axs[0].axis('off')

        # supertitle showing the standard deviation of the centroid
        plt.suptitle(f"Centroid location: {centroid_location}, \nStdv: {centroid_location_stdv}")
        plt.show()

    return centroid_location, centroid_buffer

def tracking_error_finder(heliostat_position: np.ndarray, target_position: np.ndarray, center_measurements: np.ndarray, center_measurement_uncertainties: np.ndarray = None):
    """
    Finds the tracking error of the heliostat with respect to the target

    Args:
        heliostat_position: The location of the heliostat
        target_position: The location of the target
        center_measurements: The measured center of the target with respect to the heliostat. This is 2D array
        center_measurement_uncertainties: The uncertainties of the center measurements. This is 2D array

    Returns:
        The tracking error in radians, this is a single number
    """
    
    ideal_vector = target_position - heliostat_position
    measured_vector = ideal_vector.copy()
    measured_vector[0] += center_measurements[0]    # x measruement off
    measured_vector[2] += center_measurements[1]    # z measurement off
    cos_theta = np.dot(ideal_vector, measured_vector) / (np.linalg.norm(ideal_vector) * np.linalg.norm(measured_vector))
    theta = np.arccos(cos_theta)

    return theta 
 

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

        # The if statement here are for finding of the centroid location,
        # some data needs to be treated differencetly because they need corner cropping while others do not. 
        if row["Source"] == "PSA":
            _, img_original = BCS_functions.load_image(img_path)

            centroid_location, centroid_location_buffer = find_centroid_wrapper(img_path, source_name=row["Source"], visualization=visualization)
                
        # Sandia data does not need border finding, the excel sheet does not have targer width either
        elif row["Source"] == "Sandia":
            _, img_original = BCS_functions.load_image(img_path)
            centroid_location, centroid_location_buffer = \
            find_centroid_wrapper(img_path, source_name=row["Source"], visualization=visualization)

             # TODO: Change this number, just made up since there's no data
            row["TargetW"] = 12

        elif row["Source"] == "CENER":
            # CENER data uses latitude and longtiude to find the centroid
            # Here I manually converted the data into distances and hard coded 
            row["HeliostatX"] = 0
            row["HeliostatY"] = 0
            row["HeliostatZ"] = 0

            row["TargetX"] = 0
            row["TargetY"] = 201
            row["TargetZ"] = 9.975
            centroid_location, centroid_location_buffer = \
            find_centroid_wrapper(img_path, source_name=row["Source"], visualization=visualization)

        else:
            raise ValueError(f"Unknown source {row['Source']}")
        
        # 500 is the center pixel location since the orignial image is scaled to 1000x1000
        err_x_px = centroid_location[0] - 500
        err_y_px = centroid_location[1] - 500
        px_to_m_ratio = row["TargetW"] / 1000

        # TODO: Needs to unify the naming convention for each axis, current there will be a confusion about Z and Y location
        heliostat_pos = np.array([row["HeliostatX"], row["HeliostatY"], row["HeliostatZ"]])
        target_pos = np.array([row["TargetX"], row["TargetY"], row["TargetZ"]])
        center_measurements = np.array([err_x_px * px_to_m_ratio, err_y_px * px_to_m_ratio])
        tracking_error = tracking_error_finder(heliostat_pos, target_pos, center_measurements)
        print(f"Tracking error for {row['ImageName']}: {tracking_error*1000:.2f} mrad")

        centroid_location_buffer = np.array(centroid_location_buffer)
        err_x_px_buffer = centroid_location_buffer[:, 0] - 500
        err_y_px_buffer = centroid_location_buffer[:, 1] - 500
        center_measurements_buffer = np.array([err_x_px_buffer * px_to_m_ratio, err_y_px_buffer * px_to_m_ratio])
        combined_tracking_error_distribution, horizontal_tracking_error_distribution, vertical_tracking_error_distribution \
              = tracking_error_with_uncertainties(heliostat_pos, target_pos, center_measurements_buffer)
        print(f"Source: {row['Source']}")
        print(f"Combined tracking error for {row['ImageName']}: {combined_tracking_error_distribution[0]*1000:.2f} mrad, uncertainty {combined_tracking_error_distribution[1]*1000:.2f} mrad")
        print(f"Horizontal tracking error for {row['ImageName']}: {horizontal_tracking_error_distribution[0]*1000:.2f} mrad, uncertainty {horizontal_tracking_error_distribution[1]*1000:.2f} mrad")
        print(f"Vertical tracking error for {row['ImageName']}: {vertical_tracking_error_distribution[0]*1000:.2f} mrad, uncertainty {vertical_tracking_error_distribution[1]*1000:.2f} mrad")
        print()