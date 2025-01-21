import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
import os
import cv2
from skimage.transform import hough_line, hough_line_peaks
from skimage import img_as_float

class BCS_functions:
    @staticmethod
    def load_image(img_path: str) -> np.ndarray:
        """
        Load and process image for target edge detection

        returns to images, one for edge detection, another one for centroid finding
        """
        _, imageExtension = os.path.splitext(img_path)
        img = imread(img_path)
        if len(img.shape) == 3 and img.shape[0] > 3:
            img = img[0, :, :]
        elif len(img.shape) == 3:
            img = rgb2gray(img)
        else:
            img = img

        # convert to 8 bit if image is 16 bit
        if img.dtype == np.uint16:
            img = (img / 65535.0 * 255).astype(np.uint8)

        alpha = 2
        beta = 50
        im = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)

        return im, img
    
    @staticmethod
    def find_corner_candidates(img: np.ndarray) -> list:
        """
        Finds intersection candidates of the edges in the image
        The candidates needs to be further processed to pick the actual corners
        
        Input:
        img: np.ndarray: image array

        Output:
        list: list of tuples, each tuple contains the x and y coordinates of the intersection
        """

        # preprocessings
        kernel = np.ones((3,3),np.uint8)
        blur = cv2.blur(img, (3,3))
        erodeI = 3
        dilateI = 3
        imerode = cv2.erode(blur,kernel,iterations = erodeI)
        im_dilate = cv2.dilate(imerode,kernel,iterations = dilateI)
        bin_img = cv2.adaptiveThreshold(imerode, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
        bin_img = ~bin_img

        # Hough transform
        mintestangle = -89.03
        maxtestangle = 90.2

        tested_angles = np.linspace((mintestangle*np.pi)/180, (maxtestangle*np.pi)/180, 100) 
        h, theta, d = hough_line(bin_img, tested_angles)

        hpeaks, angles, dists = hough_line_peaks(h,theta,d, min_distance=150, min_angle=2, threshold=150, num_peaks=4)

        lines = np.zeros((4,4)) #save all points
        slopes = np.zeros((4,1)) #calc slopes
        y_ints = np.zeros((4,1)) #calc y-intercepts

        for i in range(len(hpeaks)):
            x0 = 1
            x1 = bin_img.shape[1] 
            y0 = (dists[i]-x0*np.cos(angles[i]))/np.sin(angles[i]) 
            y1 = (dists[i]-x1*np.cos(angles[i]))/np.sin(angles[i])
            lines[i,:] = [x0, x1, y0, y1] #save points
            slopes[i] = (y1-y0)/(x1-x0) #save slope
            y_ints[i] = -slopes[i]*x0+y0 #save y-intercept


        def intersector(slp1, int1, slp2, int2):
            x_int = abs((int2-int1))/abs((slp1-slp2))
            y_int = slp1*x_int+int1
            return (x_int[0], y_int[0])
        
        int_1 = intersector(slopes[0], y_ints[0], slopes[1], y_ints[1])
        int_2 = intersector(slopes[0], y_ints[0], slopes[2], y_ints[2])
        int_3 = intersector(slopes[0], y_ints[0], slopes[3], y_ints[3])
        int_4 = intersector(slopes[1], y_ints[1], slopes[2], y_ints[2])
        int_5 = intersector(slopes[1], y_ints[1], slopes[3], y_ints[3])
        int_6 = intersector(slopes[2], y_ints[2], slopes[3], y_ints[3])

        return [int_1, int_2, int_3, int_4, int_5, int_6]
    
    @staticmethod
    def valid_intersections(intersection_candidate: list, img_shape: tuple) -> list:
        """
        Finds the actual intersection points from the candidate
        return the valid intersection in the order of:
        top left, top right, bottom left, bottom right

        Input:
        intersection_candidate: list: list of tuples, each tuple contains the x and y coordinates of the intersection
        img_shape: tuple: shape of the image

        Output:
        list: list of tuples, each tuple contains the x and y coordinates of the intersection
        """
        def organize_positions(positions):
            # Sort by y-coordinate (top to bottom)
            positions_sorted = sorted(positions, key=lambda pos: pos[1])
            
            # Split into top and bottom halves
            top_positions = positions_sorted[:2]
            bottom_positions = positions_sorted[2:]
            
            # Sort top positions by x-coordinate (left to right)
            top_left = min(top_positions, key=lambda pos: pos[0])
            top_right = max(top_positions, key=lambda pos: pos[0])
            
            # Sort bottom positions by x-coordinate (left to right)
            bottom_left = min(bottom_positions, key=lambda pos: pos[0])
            bottom_right = max(bottom_positions, key=lambda pos: pos[0])
            
            return [top_left, top_right, bottom_left, bottom_right]

        valid_ints = []
        for i in intersection_candidate:
            if i[0] > 0 and i[0] < img_shape[1] and i[1] > 0 and i[1] < img_shape[0]:
                valid_ints.append(i)
        return organize_positions(valid_ints)
    
    @staticmethod
    def rectify_and_crop(unprocessed_im: np.ndarray, corners: list) -> np.ndarray:
        img_output_size = 1000
        dst_points = np.float32([[0, 0], [img_output_size, 0], [0, img_output_size], [img_output_size, img_output_size]])
        src_points = np.float32(corners)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed_im = cv2.warpPerspective(unprocessed_im, M, (img_output_size, img_output_size))
        croppedIm = transformed_im[0:img_output_size, 0:img_output_size]

        return croppedIm
    
    @staticmethod
    def low_pass_filter(image, keep_ratio):
        # Convert the image to grayscale if it is not already
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get the dimensions of the image
        rows, cols = image.shape
        
        # Perform the 2D FFT
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        
        # Create a mask with the same dimensions as the image
        mask = np.zeros((rows, cols), np.uint8)
        
        # Calculate the center of the image
        crow, ccol = rows // 2 , cols // 2
        
        # Determine the size of the low-pass filter
        keep_rows = int(rows * keep_ratio / 2)
        keep_cols = int(cols * keep_ratio / 2)
        
        # Apply the mask to keep the low frequency components
        mask[crow-keep_rows:crow+keep_rows, ccol-keep_cols:ccol+keep_cols] = 1
        
        # Apply the mask to the shifted FFT
        fshift = fshift * mask
        
        # Perform the inverse FFT
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        img_back = (img_back / np.max(img_back) * 255).astype(np.uint8)
        
        return img_back
    
    
        
    @staticmethod
    def find_centroid(img):
        img = img_as_float(img)
        M = np.sum(img)
        y_indices, x_indices = np.indices(img.shape)
        X_c = np.sum(x_indices * img) / M
        Y_c = np.sum(y_indices * img) / M
        return X_c, Y_c
    
    @staticmethod
    def gamma_correction(img, gamma_val):
        img = np.power(img.astype(np.float64), gamma_val)
        img = (img / np.max(img) * 255).astype(np.uint8)

        return img
        
if __name__ == "__main__":
    img_path = r"Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/CrescentDunes\Image4.bmp"
    img, img_centroid = BCS_functions.load_image(img_path)
    corners = BCS_functions.find_corner_candidates(img)
    valid_corners = BCS_functions.valid_intersections(corners, img.shape)
    rectified_img = BCS_functions.rectify_and_crop(img_centroid, valid_corners)
    # rectified_img_filtered = BCS_functions.low_pass_filter(rectified_img, keep_ratio=0.02)
    rectified_img_filtered = cv2.medianBlur(rectified_img, 35)
    rectified_img_gamma_filtered = BCS_functions.gamma_correction(rectified_img_filtered, 5)
    centroid_location = BCS_functions.find_centroid(rectified_img_gamma_filtered)

    # matplotlib show rectified image
    import matplotlib.pyplot as plt
    plt.imshow(rectified_img_gamma_filtered, cmap='gray')
    plt.scatter(centroid_location[0], centroid_location[1], c='r', s=100)
    plt.show()

    print(centroid_location)




