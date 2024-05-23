import numpy as np 
import matplotlib.pyplot as plt
import csv 
import cv2 
from skimage.feature import canny
from skimage.exposure import rescale_intensity
from skimage.transform import hough_line, hough_line_peaks
from skimage.io import imread
from statistics import mean



def readimage_KS(imageFile):
    # Reading in selected file, seperating color bands, displaying image, and printing image shape
    img = imread(imageFile, as_gray = True)
    # plt.figure(); plt.imshow(img, cmap = plt.cm.gray)
    img_intensityscaled = rescale_intensity(img,in_range='image',out_range=(0,255)).astype(np.uint8)
    # plt.figure(); plt.imshow(img_intensityscaled, cmap = plt.cm.gray)
    #splitting color bands
    # hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    # lower_blue = np.array([40,80,0])
    # upper_blue = np.array([222,222,255])
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # result = cv2.bitwise_and(img,img, mask=mask)
    # r= cv2.split(result)
    # g = cv2.split(result)
    # b = cv2.split(result)
    # r= cv2.split(img)
    # g = cv2.split(img)
    # b = cv2.split(img)
    
    # w = img.shape
    # h=img.shape
    # c=img.shape
    # img.dtype
    # print("Image width: ",w[1]," pixels; Image height: ",w[0]," pixels")
    return img_intensityscaled



def findcorners_KS(img,fileNum):
#%% Normalizing image, improving image saturation, filtering, and binarizing
    '''
    The variables alpha and beta have to be adjusted with different images.
    
    If the image is well saturated: alpha=1 & beta=1
    if the image is slightly dark: alpha=1 & beta=80
    if the image is still dark: alpha=2 & beta=1
    if the image is still dark: alpha=2 & beta=80
    
    Because of these adjustments, I printed out successful values for alpha & beta in a csv file under Github>DataFiles. This is something that will need to be automated in the future. Maybe with image intensity sorting that assigns particular alpha and beta values depending on the overall image intensities.
    '''
    # # Apply saturation weights
    # alpha = 1
    # beta = 50    
    # im = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
    # plt.figure(); plt.imshow(im, cmap='gray')
    
    im=cv2.equalizeHist(img)
    # im = cv2.fastNlMeansDenoising(im,5,7,21)
    plt.figure(); plt.imshow(im,cmap='gray')
    
    # Adjust and binarize image
    kernel = np.ones((3,3),np.uint8)
    blur = cv2.blur(im, (3,3));
    erodeI = 3
    dilateI = 3
    imerode = cv2.erode(blur,kernel,iterations = erodeI)
    im_dilate = cv2.dilate(imerode,kernel,iterations = dilateI)
    bin_img = cv2.adaptiveThreshold(im_dilate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    bin_img = ~bin_img
    plt.figure()
    plt.axis('off')
    plt.imshow(bin_img, cmap='gray')
    
    #%% Apply Hough transform
    
    '''
    The tested angles are set to look for lines that are close but not quite 
    vertical or horizontal. This is important for intersection the Hough lines 
    because exactly vertical lines have undefined slopes. 
    Minimum distance is set to 150 to find unique houghpeaks in a certain space. 
    The figure below is the hough peaks in accumulator space (r-theta): theta on the x-axis 
    and a distance value (r) on the y-axis. Since we are looking for two peaks at theta values 
    of nearly 0, and two peaks for theta values of nearly 90, it would be helpful 
    to search for this number of peaks in a restricted range of theta values 
    instead of hoping the algorithm finds them correctly. 
    In most cases, they are found correctly, but not always.
    '''
    mintestangle = -89.03
    maxtestangle = 90.2
    
    tested_angles = np.linspace((mintestangle*np.pi)/180, (maxtestangle*np.pi)/180, 100) 
    #tested_angles = np.linspace((-np.pi)/2, (np.pi)/2, 100)
    
    # brightness = np.sum(im)/(255*np.prod(np.shape(im)))
    # print(brightness)
    # minimum_brightness = 0.5

        
    # bin_img = canny(im,2,1,25)
    plt.figure(); plt.imshow(bin_img,cmap='gray')
    
    h, theta, d = hough_line(bin_img, tested_angles)
    
    
    hpeaks, angles, dists = hough_line_peaks(h,theta,d, min_distance=150, min_angle=2, threshold=150, num_peaks=4)
    #print(hpeaks.shape)
    #print(angles.shape)
    #print(dists.shape)
    #print(np.rad2deg(angles))
    #print(dists)
    
    plt.figure()
    plt.imshow(np.log(h+1), aspect='auto', extent=[np.rad2deg(theta[0]), np.rad2deg(theta[-1]), d[-1], d[0]], cmap='gray')
    plt.plot(np.rad2deg(angles), dists, 'bo')
    #plt.axis('off')
    
    plt.show()
    
    #%% Extract Hough lines and find intersection points
    '''
    The Hough Transormation orders lines based on the most amount of votes, 
    so the order of the lines can change without being noticed. 
    means that some sorting had to be done in order to make sure that 
    we were actually intersecting lines that had an intersection point (and not parallel lines). 
    I did the sorting by finding all of the possiple intersections. 
    There are 6 possible intersection points with 2 of them not being good, 
    and these are very large positive numbers (since the intersection points occur 
    are very large either positive or negative pixel values). 
    With this knowledge, we sorted them based on their value and discarded 
    the first two intersections.
    
    Since the origin (0,0) is the top left corner, the largest pixel value will be the bottom right. The smallest pixel value is the top left.
    '''
    
    plt.figure()
    plt.imshow(bin_img, cmap=plt.cm.gray)
    
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
      line = (x0,x1),(y0,y1)
      #print(line)
      plt.plot((x0,x1),(y0,y1), '-r')
    
    
    plt.xlim((0,bin_img.shape[1]))
    plt.ylim((bin_img.shape[0],0))
    
    def intersector(slp1, int1, slp2, int2):
      #print("Slope A: ", slp1, "Slope B :", slp2 )
     
      x_int = abs((int2-int1))/abs((slp1-slp2))
      y_int = slp1*x_int+int1
      return (x_int[0], y_int[0])
    
    int_1 = intersector(slopes[0], y_ints[0], slopes[1], y_ints[1])
    int_2 = intersector(slopes[0], y_ints[0], slopes[2], y_ints[2])
    int_3 = intersector(slopes[0], y_ints[0], slopes[3], y_ints[3])
    int_4 = intersector(slopes[1], y_ints[1], slopes[2], y_ints[2])
    int_5 = intersector(slopes[1], y_ints[1], slopes[3], y_ints[3])
    int_6 = intersector(slopes[2], y_ints[2], slopes[3], y_ints[3])
    
    #sort intersections
    #There's always going to be two that don't work, these two are always going to be the first two since they are large positive numbers
    def get_max(sub):
      return max(abs(sub))
    
    def get_prod(sub):
        return abs(sub[0]*sub[1])
    test_list = [int_1, int_2, int_3, int_4, int_5, int_6]
    # test_prods = [tt[0] * tt[1] for tt in test_list]
    def cornersdt(pts): #Locate corners of a rectangular contour and arrange them in clockwise order starting at top left
        #Create a list of four points, corresponding to four corners of the target. CW starting top-left
        points_array = [None]*4
        
        sums_vec = np.sum(pts,axis = 1) #Sum of x and y coordinate of each point of function argument
        points_array[0] = pts[np.argmin(sums_vec)] #Lowest sum corresponds to top-left corner
        points_array[2] = pts[np.argmax(sums_vec)] #Highest sum corresponds to bottom-right corner
        
        diffs_vec = np.diff(pts,axis = 1)
        points_array[1] = pts[np.argmin(diffs_vec)] #Most-negative difference corresponds to top-right corner
        points_array[3] = pts[np.argmax(diffs_vec)] #Most-positive difference corresponds to bottom-left corner
        return points_array
    
    
    test_list.sort(key = get_prod, reverse = True)
    cwcorners = cornersdt(test_list[2:])
    # print("Sorted Tuples: " + str(test_list))
    
    return cwcorners

    # print("Top Left Corner: ", int_TL)
    # print("Top Right Corner: ", int_TR)
    # print("Lower Right Corner: ",int_LR)
    # print("Lower Left Corner: ",int_LL)
    #%% Save corner locations on the Y-drive.
    # f = open('Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/DataFiles/CrescentDunes/TargetEdges/TargetEdges' + "File" + str(fileNum), 'w')
    # theWriter = csv.writer(f)
    # theWriter.writerow(['Image','Alpa', 'Beta','Erosions','Dilations','Top Left Corner','Top Right Corner','Lower Left Corner','Lower Right Corner'])
    # theWriter.writerow([fileNum,alpha,beta,erodeI,dilateI,int_TL,int_TR,int_LL,int_LR])

#%%
def cropimage_KS(img,corners):
    
    int_TL = corners[0]; int_TR = corners[1]; int_LR = corners[2]; int_LL = corners[3]
    x_val = [x[0] for x in corners]
    y_val = [x[1] for x in corners]
    plt.plot(x_val,y_val, 'bo')
    

    #finding the center of the target given the coordinates of the edges 
    x_midpt = 0.5*(int_TL[0]+int_LR[0])
    y_midpt = 0.5*(int_TL[1] + int_LR[1])
    plt.plot(x_midpt,y_midpt, "o")
    target_mid = (x_midpt, y_midpt)
    # print("Center of the Target: ",target_mid)
    
    
    #finding pixels across target
    Px = 0.5*(abs(int_TL[0] - int_TR[0]) + abs(int_LL[0] - int_LR[0]))
    Py = 0.5*(abs(int_TL[1] - int_LL[1]) + abs(int_TR[1] - int_LR[1]))
    
    
    plt.axis('off')
    #plt.title("Detected Lines")
    #plt.savefig('/content/drive/MyDrive/NREL/ProcessedIms/DetectedEdges/CrescentDunes/' + "Image" + str(fileNum) + 'Detected', dpi = 300, bbox_inches='tight', pad_inches=0)
    plt.show()

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
    plt.figure()
    plt.axis("off")
    plt.imshow(croppedIm, cmap = plt.cm.gray)
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
    plt.figure; plt.imshow(trans_crop,cmap = plt.cm.gray)
    return trans_crop
    # trans_crop = BCI_trans[idd][val_crop:-val_crop,val_crop:-val_crop] # Crop a border around the image
    # macro_plot2(trans_crop[idd],"channel " + str(idd+1) + " orig.",trans_fn,221) # Plot result in new plot
    
