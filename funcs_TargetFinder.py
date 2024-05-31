import numpy as np 
import matplotlib.pyplot as plt
import csv 
import cv2 
import skimage
from skimage.feature import canny
from skimage.exposure import rescale_intensity
from skimage.transform import hough_line, hough_line_peaks
from skimage.io import imread
from statistics import mean



#%%
def cropimage_KS(img,corners):
    
    int_TL = corners[0]; int_TR = corners[1]; int_LR = corners[2]; int_LL = corners[3]
    x_val = [x[0] for x in corners]
    y_val = [x[1] for x in corners]
    # plt.plot(x_val,y_val, 'bo')
    

    #finding the center of the target given the coordinates of the edges 
    x_midpt = 0.5*(int_TL[0]+int_LR[0])
    y_midpt = 0.5*(int_TL[1] + int_LR[1])
    # plt.plot(x_midpt,y_midpt, "o")
    target_mid = (x_midpt, y_midpt)
    # print("Center of the Target: ",target_mid)
    
    
    #finding pixels across target
    Px = 0.5*(abs(int_TL[0] - int_TR[0]) + abs(int_LL[0] - int_LR[0]))
    Py = 0.5*(abs(int_TL[1] - int_LL[1]) + abs(int_TR[1] - int_LR[1]))
    
    
    # plt.axis('off')
    #plt.title("Detected Lines")
    #plt.savefig('/content/drive/MyDrive/NREL/ProcessedIms/DetectedEdges/CrescentDunes/' + "Image" + str(fileNum) + 'Detected', dpi = 300, bbox_inches='tight', pad_inches=0)
    # plt.show()

    #%% Crop image to just the target bounds
    x_rect_left = int(mean((int_TL[0],int_LL[0])))
    x_rect_right = int(mean((int_TR[0],int_LR[0])))
    y_rect_upper = int(mean((int_TL[1],int_TR[1])))
    y_rect_lower = int(mean((int_LL[1],int_LR[1])))
    
    croppedIm = img[y_rect_upper:y_rect_lower, x_rect_left:x_rect_right]
    # Redefine the target midpoint in cropped space
    target_mid_cropped = (x_midpt-x_rect_left,y_midpt-y_rect_upper)
    
    # Original crop + resize code below
    #croppedIm = img[int(int_TL[1]):int(int_LL[1]), int(int_LL[0]):int(int_LR[0])]
    #croppedIm = cv2.resize(croppedIm, (1936,1456))
    # plt.figure()
    # plt.axis("off"); plt.title('Crop, KS method')
    # plt.imshow(croppedIm, cmap = plt.cm.gray)
    return croppedIm


#%%
def cropimage_DT(img,in_corners):
    def cornersdt(pts): #Locate corners of a rectangular contour and arrange them in clockwise order starting at top left
        #Create a list of four points, corresponding to four corners of the target. CW starting top-left
        points_array = np.zeros((4,2),dtype = int)
        
        sums_vec = np.sum(pts,axis = 1) #Sum of x and y coordinate of each point of function argument
        points_array[0] = pts[np.argmin(sums_vec)] #Lowest sum corresponds to top-left corner
        points_array[2] = pts[np.argmax(sums_vec)] #Highest sum corresponds to bottom-right corner
        
        diffs_vec = np.diff(pts,axis = 1)
        points_array[1] = pts[np.argmin(diffs_vec)] #Most-negative difference corresponds to top-right corner
        points_array[3] = pts[np.argmax(diffs_vec)] #Most-positive difference corresponds to bottom-left corner
        
        return points_array
    
    def transform_target (input_image, pts): # Perform an image transform to render the target image rectangular (compensate for camera angle)
        (T_L, T_R, B_R, B_L) = pts
        # Calculate the length of each side of the original target rectangle
        d_upper = np.sqrt((T_R[0] - T_L[0])**2 + (T_R[1] - T_L[1])**2)
        d_lower = np.sqrt((B_R[0] - B_L[0])**2 + (B_R[1] - B_L[1])**2)
        d_left = np.sqrt((T_L[0] - B_L[0])**2 + (T_L[1] - B_L[1])**2)
        d_right = np.sqrt((T_R[0] - B_R[0])**2 + (T_R[1] - B_R[1])**2)
        
        #Calculate output image's dimensions based on maximum w and h of input shape
        output_width = max((int(d_upper),int(d_lower)))
        output_height = max((int(d_left),int(d_right)))
        
        #Define the output image's size using the previously-calculated width and height values
        output_perimeter = np.array([
            [0,0],
            [output_width - 1,0],
            [output_width - 1, output_height - 1],
            [0,output_height - 1]], dtype = "float32")
        
        #Perform the transform
        transform_array = cv2.getPerspectiveTransform(pts,output_perimeter) #Outputs a 3 x 3 transform matrix
        output_image = cv2.warpPerspective(input_image, transform_array,(output_width,output_height)) #Applies transform matrix to original image
        return output_image
    
    corners = cornersdt(in_corners)
    
    
    BCI_trans = transform_target(img,corners.astype(np.float32)) # Perform perspective correction and cropping to make the ORIGINAL target image perfectly rectangular and stripped of background        
    trans_crop = BCI_trans
    # plt.figure(); plt.imshow(trans_crop,cmap = plt.cm.gray); plt.axis('off'); plt.title('Crop, DT method')
    return trans_crop
    # trans_crop = BCI_trans[idd][val_crop:-val_crop,val_crop:-val_crop] # Crop a border around the image
    # macro_plot2(trans_crop[idd],"channel " + str(idd+1) + " orig.",trans_fn,221) # Plot result in new plot
    
