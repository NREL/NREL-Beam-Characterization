import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import csv 
import cv2 
import skimage
from skimage.feature import canny
from skimage.exposure import rescale_intensity
from skimage.transform import hough_line, hough_line_peaks
from skimage.io import imread
from statistics import mean


def readimage_KS(imageFile):
    def splitcolorbands(img): # unused function currently
        #splitting color bands
        hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        lower_blue = np.array([40,80,0])
        upper_blue = np.array([222,222,255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(img,img, mask=mask)
        r= cv2.split(result)
        g = cv2.split(result)
        b = cv2.split(result)
        r= cv2.split(img)
        g = cv2.split(img)
        b = cv2.split(img)
        
        w = img.shape
        h=img.shape
        c=img.shape
        img.dtype
        print("Image width: ",w[1]," pixels; Image height: ",w[0]," pixels")
        
        
    # Reading in selected file, seperating color bands, displaying image, and printing image shape
    img = cv2.imread(imageFile, 0)
    # maxhere = np.iinfo(img.dtype).max
    # # Assume we're setting this to a uint8
    # maxnew = np.floor(np.max(img)/maxhere*255)
    maxnew = 255
    img_intensityscaled = rescale_intensity(img,in_range='image',out_range=(0,maxnew)).astype(np.uint8) 
    
    
    imgout = img_intensityscaled
    
    plt.figure(); plt.imshow(imgout, cmap = plt.cm.gray); plt.axis('off')
    plt.title('Image after import and dtype conversion')
    
    return imgout


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
    
    # # Perform histogram equalization, which will also brighten dark images
    # # Two types of equalization are performed below; pick one.
    # # The first is standard equalization, the second is CLAHE (adaptive)
    # im=cv2.equalizeHist(img)
    # plt.figure(); plt.imshow(im,cmap='gray')
    
    # clahe = cv2.createCLAHE()
    # im = clahe.apply(img)
    # plt.figure(); plt.imshow(im, cmap='gray')
    
    def macro_morph(img,kernel): #Apply morphology to an input image and return the morphologically-modified image
        BCI_morph = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
        BCI_morph = cv2.morphologyEx(BCI_morph,cv2.MORPH_CLOSE,kernel)
        # BCI_morph = cv2.morphologyEx(BCI_morph,cv2.MORPH_ERODE,kernel)
        return BCI_morph
    
    def macro_contour(img): # Identify the largest image contour and return it as an 2-dimensional array
        mac_raw, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # Find all contours
        mac_loc = np.argmax([np.size(mmac,axis = 0) for mmac in mac_raw]) # Identify the ID of the longest contour in each channel
        mac_perim = np.squeeze(mac_raw[mac_loc]) # Set up the longest contour as a 2D array of points
        return mac_perim
    
    def corners(pts): #Locate corners of a rectangular contour and arrange them in clockwise order starting at top left
        #Create a list of four points, corresponding to four corners of the target. CW starting top-left
        points_array = np.zeros((4,2),dtype = int)
        
        sums_vec = np.sum(pts,axis = 1) #Sum of x and y coordinate of each point of function argument
        points_array[0] = pts[np.argmin(sums_vec)] #Lowest sum corresponds to top-left corner
        points_array[2] = pts[np.argmax(sums_vec)] #Highest sum corresponds to bottom-right corner
        
        diffs_vec = np.diff(pts,axis = 1)
        points_array[1] = pts[np.argmin(diffs_vec)] #Most-negative difference corresponds to top-right corner
        points_array[3] = pts[np.argmax(diffs_vec)] #Most-positive difference corresponds to bottom-left corner
        
        return points_array

    
    # Adjust and binarize image
    def localBinarize(inputimage):
        kernel = np.ones((3,3),np.uint8)
        blur = cv2.blur(inputimage, (3,3));
        erodeI = 3
        dilateI = 3
        imerode = cv2.erode(blur,kernel,iterations = erodeI)
        im_dilate = cv2.dilate(imerode,kernel,iterations = dilateI)
        thresmax = np.iinfo(inputimage.dtype).max # Maximum pixel brightness value based on image datatype
        bin_img = cv2.adaptiveThreshold(im_dilate, thresmax, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
        bin_img = ~bin_img
        # plt.figure()
        # plt.axis('off')
        # plt.imshow(bin_img, cmap='gray')
        return bin_img 
    
    def bilat(inputimage):
        kernel = np.ones((7,7))/49
        twofilter = cv2.filter2D(inputimage,ddepth=-1,kernel=kernel)
        return cv2.bilateralFilter(twofilter,11,81,81)
    
    def getPerimeter(inputPoints):
        #Takes a list of points (each item in the list is an x-y pair) and produces the perimeter of the resulting closed shape
        endPoints = inputPoints[1:]
        endPoints.append(inputPoints[0])
        # print(endPoints)
        numSegments = len(inputPoints)
        segmentLengths = [None]*numSegments
        for idp, pp in enumerate(inputPoints):
            x1 = pp[0]; y1 = pp[1]
            x2 = endPoints[idp][0]; y2 = endPoints[idp][1]
            segmentLengths[idp] = ((x2-x1)**2 + (y2-y1)**2)**0.5
        return np.sum(segmentLengths)
    
    img_step1 = localBinarize(img)
    img_step2 = skimage.morphology.skeletonize(cv2.morphologyEx(img_step1,cv2.MORPH_OPEN,np.ones((3,3))))
    # img_step2 = cv2.Canny(img_step1,90,255)
    
    img_step3 = canny(bilat(img),sigma=1)
    # plt.figure(); plt.imshow(bin_img,cmap='gray')
    #%% Apply Hough transform
    def macro_hough(bin_img,plotTitle):
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
        # mintestangle = -89.03
        # maxtestangle = 90.2
        # tested_angles = np.linspace((mintestangle*np.pi)/180, (maxtestangle*np.pi)/180, 100) 
        
        # h, theta, d = hough_line(bin_img, tested_angles)
        h, theta, d = hough_line(bin_img)
        
        npeaks = 4
        hpeaks, angles, dists = hough_line_peaks(h,theta,d, min_distance=150, num_peaks=npeaks) # min_angle=2, threshold=150,
        #print(hpeaks.shape)
        #print(angles.shape)
        #print(dists.shape)
        #print(np.rad2deg(angles))
        #print(dists)
        
        # plt.figure()
        # plt.imshow(np.log(h+1), aspect='auto', extent=[np.rad2deg(theta[0]), np.rad2deg(theta[-1]), d[-1], d[0]], cmap='gray')
        # plt.plot(np.rad2deg(angles), dists, 'bo')
        # #plt.axis('off')
        
        # plt.show()
        
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
        
        lines = np.zeros((npeaks,4)) #save all points
        slopes = np.zeros((npeaks,1)) #calc slopes
        y_ints = np.zeros((npeaks,1)) #calc y-intercepts
        
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
        plt.axis('off'); plt.title(plotTitle); 
        
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
        cwcorners = cornersdt(test_list[-4:])
        # print("Sorted Tuples: " + str(test_list))
        
        plt.show(); 
        return cwcorners
    
    # imcorn0 = macro_hough(img_step0, '0. Tuckerization')
    # Attempt finding Hough lines for three different preprocessing variants
    imcorn1 = macro_hough(img_step1, '1. Binarize')
    imcorn2 = macro_hough(img_step2, '2. Binarize then skeletonize')
    imcorn3 = macro_hough(img_step3, '3. Canny edge detect')
    
    corns = [imcorn1,imcorn2,imcorn3]
    perimeters = np.zeros((len(corns),1))
    
    # Get target perimeter length where all four intersection points are within image bounds.
    for idc,cc in enumerate(corns):
        if np.min(np.array(cc)) > 0: perimeters[idc] = getPerimeter(cc)
        
    # Find minimum non-zero perimeter
    funcout = np.min(perimeters[np.nonzero(perimeters)])
    cornloc = np.where(perimeters==funcout)[0][0]
    
    # If the minimum non-zero perimeter is within 2% of the third preprocessing method,
    # it is preferred to just use this method because it is assumed correct.
    errbound = 0.01
    if (1-errbound)*perimeters[-1] < perimeters[cornloc] < (1+errbound)*perimeters[-1]:
        cornloc = len(perimeters)-1
    chosencorns = corns[cornloc]
    chosenvariant = cornloc + 1
    # chosenvariant = int(input("Enter 1, 2, or 3 depending on which target edges are correct: ",))
    # if chosenvariant==1:
    #     funcout = imcorn1
    #     print('Variant 1 chosen')
    # elif chosenvariant == 2:
    #     funcout = imcorn2
    #     print('Variant 2 chosen')
    # else: 
    #     funcout = imcorn3
    #     print('Variant 3 chosen')
    return [chosencorns, chosenvariant]
    
    # print("Top Left Corner: ", int_TL)
    # print("Top Right Corner: ", int_TR)
    # print("Lower Right Corner: ",int_LR)
    # print("Lower Left Corner: ",int_LL)
    #%% Save corner locations on the Y-drive.
    # f = open('Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/DataFiles/CrescentDunes/TargetEdges/TargetEdges' + "File" + str(fileNum), 'w')
    # theWriter = csv.writer(f)
    # theWriter.writerow(['Image','Alpa', 'Beta','Erosions','Dilations','Top Left Corner','Top Right Corner','Lower Left Corner','Lower Right Corner'])
    # theWriter.writerow([fileNum,alpha,beta,erodeI,dilateI,int_TL,int_TR,int_LL,int_LR])
