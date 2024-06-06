import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Polygon
import cv2 
import csv
from math import sqrt

def findcenter_KS(croppedIm, fileNum, params):
    #%% Accentuate shape of beam on target
    kernel = np.ones((params['kernelsize'],params['kernelsize']),np.uint8) # default = 3
    blur = cv2.blur(croppedIm, (params['blursize'],params['blursize'])) # default = 3
    erodeI = params['erodeI'] # default = 5
    dilateI = params['dilateI'] # default = 15
    imerode = cv2.erode(blur,kernel,iterations = erodeI)
    im_dilate = cv2.dilate(imerode,kernel,iterations = dilateI)

    imedge = cv2.Canny(im_dilate, params['cannylower'],params['cannyupper']) # defaults: 8, 19
    thresmax = np.iinfo(croppedIm.dtype).max
    img_th = cv2.adaptiveThreshold(imedge, thresmax, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    #img_th=~img_th  
    #finding contours is easier when the contours themselves are in white. The contour I am looking for is the innermost contour that is in white. 


    #morphlogy to remove unwanted noise
    img_th = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    # plt.imshow(img_th, cmap='gray'); plt.axis('off')


    #%% Find contours
    '''
    RETR_CCOMP is a good method for finding both exterior and interior contours 
    and distinguishing them.

    CHAIN_APPROX_SIMPLE is good for finding less duplicate and more unique contours
    '''

    contours, heirarchy = cv2.findContours(img_th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #img_th=~img_th
    src_copy =img_th.copy()
    src_copy = cv2.cvtColor(src_copy, cv2.COLOR_BGR2RGB)

    for i, cont in enumerate(contours):
      if heirarchy[0][i][3] == -1:
        src_copy = cv2.drawContours(src_copy, cont, -1, (0,255,0), 5)
      else:
        src_copy = cv2.drawContours(src_copy, cont, -1, (0,0,255),5)

    # print("Number of Contours: {}".format(len(contours)))
    #sorted_contours = sorted(contour1, key=cv2.contourArea, reverse= True)
    #for i, cont in enumerate(sorted_contours[:3],1):
     # cv2.drawContours(img, contour1, -1, (0,255,0), 3)
      #cv2.putText(im, str(i), (cont[0,0,1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 4)


    # plt.figure(figsize=[5,5])
    # plt.imshow(src_copy);plt.title("Beam Contour");plt.axis('off')

    #%% Filtering the contours, and fitting elipses to them
    '''
    The x and y components of the ellipse centroid are found along with the 
    ellipse eccentricity and are saved in a csv file under Github>DataFiles.
    '''
    #R = distance heliostat to target in meters
    # W = width of the target in meters
    # H = height of the target in meters



    from numpy.ma.core import arctan
    def eccentricity_from_ellipse(contour):
        """Calculates the eccentricity fitting an ellipse from a contour"""

        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)

        a = ma / 2
        b = MA / 2

        ecc = np.sqrt(a ** 2 - b ** 2) / a
        return ecc 

    cnts, hiers = cv2.findContours(img_th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
   
    
    src_copy =img_th.copy()
    src_copy = cv2.cvtColor(src_copy, cv2.COLOR_BGR2RGB)

    # Pixel areas. These may need adjustment or an equation-based designation based
    # on camera resolutioin to keep from having to fiddle with them. 
    max_area = np.size(croppedIm)/2
    min_area = max_area / 100


    for i, cont in enumerate(cnts):
      if heirarchy[0][i][3] == -1:
        src_copy = cv2.drawContours(src_copy, cont, -1, (0,255,0), 5)
      else:
        src_copy = cv2.drawContours(src_copy, cont, -1, (0,0,255),5)
    
    area = np.zeros(len(cnts)); isarea = area.copy()
    # phi_alt = area.copy(); phi_elv = area.copy()
    Centroid = [None]*len(cnts)
    for ic,cnt in enumerate(cnts): 
      area[ic] = cv2.contourArea(cnt)
      if min_area <= area[ic] <= max_area:
        # Find and store centroid
        isarea[ic]=1
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(src_copy, ellipse, (255,0,0), 10, cv2.LINE_AA)
        M = cv2.moments(cnt)
        # cX = int(M["m10"]/M["m00"])
        # cY = int(M["m01"]/M["m00"])
        # cv2.circle(src_copy, (cX,cY), 7, (255,0,0), -1)
        cX = M["m10"]/M["m00"]
        cY = M["m01"]/M["m00"]
        Centroid[ic]=(cX,cY)
 
        
        eccentricity = eccentricity_from_ellipse(cnt)
        # f1 = open('Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/DataFiles/CrescentDunes/BeamContour/BeamContour' + "File" + str(fileNum), 'w')
        # theWriter1 = csv.writer(f1)
        # theWriter1.writerow(['Centroids','Eccentricity'])
        # theWriter1.writerow([(cX,cY), eccentricity])   
    area = area*isarea
    if len(cnts) == 0 or np.sum(area)==0: # If we didn't find any contours
        return []
    else:
        # Find the smallest ellipse, which corresponds to the bright inner beam area
        # Alternatively, finding the largest ellipse would likely give an intensity-weighted centroid result.
        area_inner_loc = np.argmax(area*[area==np.min(area[np.nonzero(area)])])
        
        # plt.figure(figsize=[5,5])
        # plt.axis('off')
        # plt.imshow(src_copy)
    
        # plt.savefig('Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/DataFiles/CrescentDunes/ProcessedIms/BeamDetection/' + "Image" + str(fileNum), bbox_inches='tight', pad_inches=0)
         
        return Centroid[area_inner_loc]

    
def findcenter_DT(croppedIm, fileNum, params):
    def macro_scale(img): 
        # This function scales the image such that its darkest spot is 0 and its lightest is 255
        gray_min = img.min()
        gray_max = img.max()
        im_scaled = ((img - gray_min)*(255/(gray_max-gray_min))).astype(np.uint8)
        return im_scaled    
    
    def macro_contour(img):
        # This function identifies the largest image contour and return it as an 2-dimensional array
        mac_raw, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # Find all contours
        mac_loc = np.argmax([np.size(mmac,axis = 0) for mmac in mac_raw]) # Identify the ID of the longest contour in each channel
        mac_perim = np.squeeze(mac_raw[mac_loc]) # Set up the longest contour as a 2D array of points
        return mac_perim
    
    def macro_plot2(img,header,fn,axn): #(image variable, plot title, figure number, 3-digit subplot ID)
        plt.figure(fn,dpi=600); plt.subplot(axn)
        ax = plt.gca(); fig = plt.gcf()
        implot = ax.imshow(img,cmap='gray',interpolation='none')
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(header,fontsize=6)
        # plt.tight_layout()
        return fig,ax
    
    # With the transformed target-only picture, identify the beam area on the target
    # Blur the image.
    imsize = np.shape(croppedIm)
    # cs = np.floor(imsize[0] * imsize[1] / 50000).astype(int)
    # d = np.floor(cs / 8).astype(int)
    cs = params['bilateralcs']; d = params['bilaterald'] # defaults: 51, 9
    trans_blur = cv2.bilateralFilter(croppedIm,d,cs,cs)
    
    # Scale the image. 
    trans_scaled = macro_scale(trans_blur) 
        
    # Threshold the image. The background should be mostly blank
    #Start with Otsu's method
    val_otsu, trans_otsu = cv2.threshold(trans_scaled,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    val_thresh = int(val_otsu*params['otsumultiplier'])
    val_thresh, trans_thresh = cv2.threshold(trans_scaled,val_thresh,255,cv2.THRESH_BINARY)
    
    # Normalize and multiply the thresheld image by the original cropped image
    # Trans_fin is an output image where all pixels outside the beam area are black, but the
    # beam area itself remains in accurate original photograph intensity.
    trans_fin = (trans_thresh/255 * croppedIm).astype(np.uint8)
    
    # Take the image moments
    fin_mom = cv2.moments(trans_fin, binaryImage=False)
    fin_area = fin_mom["m00"]
    
    # Calculate intensity-weighted centroid using image moments
    cX = fin_mom["m10"] / fin_mom["m00"]
    cY = fin_mom["m01"] / fin_mom["m00"]
    circ = Circle((int(cX),int(cY)),radius=5,color='black');
    
    
    # Calculate the 90% flux area by iteratively eroding the image and comparing its area moment to original moment
    erode_temp = trans_fin
    kernel = np.ones((3,3),np.uint8) # Kernel size for iterative erosion
    fluxpoints = [0.9,0.75,0.5,0.25,0.1]
    idfxp = 0; idfxpmax = len(fluxpoints); storage_contours= [None]*idfxpmax
    
    for pp in range(1,min((np.size(erode_temp,axis=0),np.size(erode_temp,axis=1)))):
        erode_temp = cv2.erode(erode_temp,kernel) # Perform erosion
        erode_mom = cv2.moments(erode_temp,binaryImage=False) # Calculate area moment
        erode_fluxratio = erode_mom["m00"]/fin_mom["m00"] #Compare area moments
                
        if erode_fluxratio < fluxpoints[idfxp]:
            thiscontour = macro_contour(erode_temp)
            storage_contours[idfxp] = thiscontour
            idfxp = idfxp+1
       
        if idfxp == idfxpmax: break
    else:
        print("Error locating all flux areas.")
        exit()
        
    # # Plot the output from the last code block
    # plt.figure();    
    # f_fig, f_ax = macro_plot2(trans_fin,"Flux map",plt.gcf().number,111)
    
    # #Add patches to show 90% region and centroid # 'cyan',
    # cpalette = ['darkturquoise','slateblue','springgreen','mediumseagreen',
    #             'yellowgreen','yellow','orange','red','brown']
    
    # for id_xx, xx in enumerate(storage_contours):
    #     patch_flx = Polygon(xx,closed=True,facecolor=cpalette[id_xx],edgecolor='none')
    #     f_ax.add_patch(patch_flx)
   
    # f_ax.add_patch(circ) # Plot centroid on image
    
    # plt.tight_layout()
    
    return (cX,cY)
