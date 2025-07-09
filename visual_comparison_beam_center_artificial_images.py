import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread
import pandas as pd
from skimage.filters import threshold_yen as threshold_yen_
from skimage.measure import label, regionprops
import cv2
from glob import glob

artificial_image_folder = r"C:\Users\qzheng\OneDrive - NREL\BCS Comparison\ArtificialImages"
subfolders = [f.path for f in os.scandir(artificial_image_folder) if f.is_dir()]
corner_file_name = "corner_locations.csv"
corner_file_location = os.path.join("Artificial_image_processing", corner_file_name)
resized_image_width = 500
resized_image_height = 500


def crop_and_rectify(corner_file:str, img_name:str, width:int=1000, height:int=1000):
    corner_df = pd.read_csv(corner_file, index_col=0)
    img_base_name = os.path.basename(img_name)
    img = imread(img_name)
    heliostat_field = "CENER" if "cener" in img_name else "STJ"
    corner_data = corner_df.loc[f"{heliostat_field}_{img_base_name}"]
    upper_left_x = int(corner_data['ULX'])
    upper_left_y = int(corner_data['ULY'])
    upper_right_x = int(corner_data['URX'])
    upper_right_y = int(corner_data['URY'])
    lower_left_x = int(corner_data['LLX'])
    lower_left_y = int(corner_data['LLY'])
    lower_right_x = int(corner_data['LRX'])
    lower_right_y = int(corner_data['LRY'])

    # perspective transform the locations within the corners to a rectangle described by width and height
    src_points = np.array([[upper_left_x, upper_left_y],
                           [upper_right_x, upper_right_y],
                           [lower_left_x, lower_left_y],
                           [lower_right_x, lower_right_y]], dtype=np.float32)
    
    dst_points = np.array([[0, 0],
                           [width, 0],
                           [0, height],
                           [height, width]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    rectified_img = cv2.warpPerspective(img, M, (width, height))
    return rectified_img

def get_inverse_perspective_transform(corner_file, img_name, width, height):
    corner_df = pd.read_csv(corner_file, index_col=0)
    heliostat_field = "CENER" if "cener" in img_name else "STJ"
    img_base_name = os.path.basename(img_name)
    corner_data = corner_df.loc[f"{heliostat_field}_{img_base_name}"]
    upper_left_x = int(corner_data['ULX'])
    upper_left_y = int(corner_data['ULY'])
    upper_right_x = int(corner_data['URX'])
    upper_right_y = int(corner_data['URY'])
    lower_left_x = int(corner_data['LLX'])
    lower_left_y = int(corner_data['LLY'])
    lower_right_x = int(corner_data['LRX'])
    lower_right_y = int(corner_data['LRY'])

    # perspective transform the locations within the corners to a rectangle described by width and height
    src_points = np.array([[upper_left_x, upper_left_y],
                           [upper_right_x, upper_right_y],
                           [lower_left_x, lower_left_y],
                           [lower_right_x, lower_right_y]], dtype=np.float32)
    
    dst_points = np.array([[0, 0],
                           [width, 0],
                           [0, height],
                           [height, width]], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return np.linalg.inv(M)


def find_theshold_with_energy(img_input, energy_level):
    sorted_data = np.sort(img_input.flatten())[::-1]
    cumulative_sum = np.cumsum(sorted_data)
    total_sum = cumulative_sum[-1]
    threshold_index = np.searchsorted(cumulative_sum, total_sum * energy_level / 100.0)
    threshold_value = sorted_data[threshold_index]
    return threshold_value

def find_centroid(img_input, bright_spot_threshold):
    regionprops_list = regionprops(label(img_input > bright_spot_threshold), 
                                            intensity_image=img_input)
    sorted_regionprops = sorted(regionprops_list, key=lambda x: x.area, reverse=True)

    beam_center_pixel = sorted_regionprops[0].centroid
    return beam_center_pixel

def apply_inverse_transform(points, inverse_M):
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = inverse_M @ points_homogeneous.T
    transformed_points /= transformed_points[2, :]  # Normalize by the third coordinate
    return transformed_points[:2, :].T


def main():
    for subfolder in subfolders:
        image_files = glob(os.path.join(subfolder, "*.png"))
        for image_file in image_files:
            _image = imread(image_file)
            rectified_img = crop_and_rectify(corner_file_location, image_file, width=resized_image_width, height=resized_image_height)
            
            hist, bin_edges = np.histogram(rectified_img.flatten(), bins=256, range=(0, 255))
            peak_idx = np.argmax(hist)
            peak_bin_right = bin_edges[peak_idx + 1]
            background_threshold = peak_bin_right + 5
            rectified_image_float = rectified_img.astype(np.float32)
            rectified_image_float -= background_threshold
            # clip the values to be in the range [0, 255]
            rectified_image_float = np.clip(rectified_image_float, 0, 255)
            # convert back to uint8
            rectified_image_uint8 = rectified_image_float.astype(np.uint8)
            threshold_yen = threshold_yen_(rectified_image_uint8)
            threshold_90 = find_theshold_with_energy(rectified_image_uint8, 90)
            threshold_60 = find_theshold_with_energy(rectified_image_uint8, 60)

            img_threshold_90_binary = rectified_image_uint8 > threshold_90
            img_threshold_60_binary = rectified_image_uint8 > threshold_60
            img_threshold_yen_binary = rectified_image_uint8 > threshold_yen
            img_threhsold_90 = rectified_image_uint8.copy()
            img_threhsold_90[~img_threshold_90_binary] = 0
            img_threhsold_90_center = find_centroid(rectified_image_uint8, threshold_90)
            img_threhsold_60 = rectified_image_uint8.copy()
            img_threhsold_60[~img_threshold_60_binary] = 0
            img_threhsold_60_center = find_centroid(rectified_image_uint8, threshold_60)
            img_threshold_yen = rectified_image_uint8.copy()
            img_threshold_yen[~img_threshold_yen_binary] = 0
            img_threshold_yen_center = find_centroid(rectified_image_uint8, threshold_yen)

            img_max_val = np.max(rectified_image_uint8)
            (y_max, x_max) = np.where(rectified_image_uint8 == img_max_val)
            y_max = y_max[0]
            x_max = x_max[0]

            inverse_M = get_inverse_perspective_transform(corner_file=corner_file_location, img_name=image_file,
                                               width=resized_image_width, height=resized_image_height)

            centers = np.array([
                (img_threhsold_90_center[1], img_threhsold_90_center[0]),
                (img_threhsold_60_center[1], img_threhsold_60_center[0]),
                (img_threshold_yen_center[1], img_threshold_yen_center[0]),
                (x_max, y_max)  # x and y are swapped in center coordinates
            ])

            transformed_centers = apply_inverse_transform(centers, inverse_M)

            column_names = ["Scaled 90% Energy", "Scaled 60% Energy", "Scaled Yen", "Scaled Peak Flux",
                "Original 90% Energy", "Original 60% Energy", "Original Yen", "Original Peak Flux"]

            row_names = ["X", "Y"]

            # create the dataframe
            df_centers = pd.DataFrame(index=row_names, columns=column_names)
            # fill the dataframe with the transformed centers, round to integers for all
            df_centers.loc["X", "Scaled 90% Energy"] = np.round(centers[0, 0]).astype(int)
            df_centers.loc["Y", "Scaled 90% Energy"] = np.round(centers[0, 1]).astype(int)
            df_centers.loc["X", "Scaled 60% Energy"] = np.round(centers[1, 0]).astype(int)
            df_centers.loc["Y", "Scaled 60% Energy"] = np.round(centers[1, 1]).astype(int)
            df_centers.loc["X", "Scaled Yen"] = np.round(centers[2, 0]).astype(int)
            df_centers.loc["Y", "Scaled Yen"] = np.round(centers[2, 1]).astype(int)
            df_centers.loc["X", "Scaled Peak Flux"] = np.round(centers[3, 0]).astype(int)
            df_centers.loc["Y", "Scaled Peak Flux"] = np.round(centers[3, 1]).astype(int)

            df_centers.loc["X", "Original 90% Energy"] = np.round(transformed_centers[0, 0]).astype(int)
            df_centers.loc["Y", "Original 90% Energy"] = np.round(transformed_centers[0, 1]).astype(int)
            df_centers.loc["X", "Original 60% Energy"] = np.round(transformed_centers[1, 0]).astype(int)
            df_centers.loc["Y", "Original 60% Energy"] = np.round(transformed_centers[1, 1]).astype(int)
            df_centers.loc["X", "Original Yen"] = np.round(transformed_centers[2, 0]).astype(int)
            df_centers.loc["Y", "Original Yen"] = np.round(transformed_centers[2, 1]).astype(int)
            df_centers.loc["X", "Original Peak Flux"] = np.round(transformed_centers[3, 0]).astype(int)
            df_centers.loc["Y", "Original Peak Flux"] = np.round(transformed_centers[3, 1]).astype(int)

            # save the dataframe to csv
            csv_saving_folder = os.path.join("Artificial_image_processing", "beam_centers", os.path.basename(subfolder))
            csv_file_name = os.path.basename(image_file).replace(".png", ".csv")
            csv_file_path = os.path.join(csv_saving_folder, csv_file_name)
            os.makedirs(csv_saving_folder, exist_ok=True)
            df_centers.to_csv(csv_file_path)

            # Also save the visualization plot
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.imshow(_image, cmap="gray", vmin=0, vmax=255)
            plt.scatter(transformed_centers[0, 0], transformed_centers[0, 1],
                        label="90% Energy", color="red", alpha=0.5)
            plt.scatter(transformed_centers[1, 0], transformed_centers[1, 1],
                        label="60% Energy", color="blue", alpha=0.5)
            plt.scatter(transformed_centers[2, 0], transformed_centers[2, 1],
                        label="Yen", color="green", alpha=0.5)
            plt.scatter(transformed_centers[3, 0], transformed_centers[3, 1],
                        label="Peak Flux point", color="orange", alpha=0.5)
            plt.title("Transformed Beam Centers in Original Image")
            plt.xlabel("X (px)")
            plt.ylabel("Y (px)")
            plt.legend()
            # turn off axis
            plt.axis("off")
            plt.xticks([])
            plt.yticks([])  
            # save the figure
            fig_file_path = os.path.join(csv_saving_folder, os.path.basename(image_file).replace(".png", "_visualization.png"))
            fig.savefig(fig_file_path, bbox_inches='tight', dpi=300)
            # close the figure
            plt.close(fig)

def summarizing_results():
    column_names = ["90% Energy X", "90% Energy Y", "60% Energy X", "60% Energy Y",
                    "Yen X", "Yen Y", "Peak Flux X", "Peak Flux Y"]

    result_folder = os.path.join("Artificial_image_processing", "beam_centers")
    result_folders = [f.path for f in os.scandir(result_folder) if f.is_dir()]
    result_summary_dict = {}
    for result_subfolder in result_folders:
        current_folder_name = os.path.basename(result_subfolder)
        csv_files = glob(os.path.join(result_subfolder, "*.csv"))

        for csv_file in csv_files:
            key = os.path.basename(csv_file).replace(".csv", "")
            key = f"{current_folder_name}_{key}"
            df = pd.read_csv(csv_file, index_col=0)
            result_summary_dict[key] = [df.loc["X", "Scaled 90% Energy"],
                                        df.loc["Y", "Scaled 90% Energy"],
                                        df.loc["X", "Scaled 60% Energy"],
                                        df.loc["Y", "Scaled 60% Energy"],
                                        df.loc["X", "Scaled Yen"],
                                        df.loc["Y", "Scaled Yen"],
                                        df.loc["X", "Scaled Peak Flux"],
                                        df.loc["Y", "Scaled Peak Flux"]]
    result_summary_df = pd.DataFrame.from_dict(result_summary_dict, orient='index', columns=column_names)
    
        
            

if __name__ == "__main__":
    # main()
    # print("Processing completed.")
    summarizing_results()
