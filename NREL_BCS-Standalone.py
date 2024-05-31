#%% Module imports and setup
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import glob
from math import sqrt
import cv2
import skimage as sk


# print("Imports complete.")

# Deliver inputs to the script
R = 200 # Heliostat-target distance, meters
W = 12 # Target width, meters
H = 12 # Target height, meters
# input_locdirectory = "Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/CrescentDunes"
input_locdirectory = r'C:\Users\DTSVANKI\OneDrive - NREL\BCS Comparison\CENER\data\raw_input\raw_input\CAT\04_03_2021\images\Images_2'
input_filetype = "tif"

'''
Choosing image files to process
   
'''
imageFilename = input_locdirectory + "/*." + input_filetype
imageFiles = sorted(glob.glob(imageFilename))

for ii in range(len(imageFiles)):
  print(ii+1, ",", imageFiles[ii])
fileNum = int(input("Enter file number: "))
imageFile = imageFiles[fileNum-1]

print("Selected file: ",imageFile)

#%% Process image using imported functions
from funcs_CornerFinder import * 
from funcs_TargetFinder import  *
from funcs_CentroidFinder import *

img = readimage_KS(imageFile)
corners,submethod = findcorners_KS(img,fileNum)
print("Method ",submethod," chosen.")
#%%

croppedImDT = cropimage_DT(img,corners)
croppedImKS = cropimage_KS(img,corners)

def BCSresults(croppedIm,Centroid):
    target_mid_cropped = np.flipud(np.shape(croppedIm))/2
    dx = abs(Centroid[0]-target_mid_cropped[0])
    dy = abs(Centroid[1]-target_mid_cropped[1])
    dr = sqrt((dx)**2 +(dy)**2)
    
    
    #finding the pixel extent
    PEx = W/np.shape(croppedIm)[1]
    PEy = H/np.shape(croppedIm)[0]
    
    #finding Phi altitude and elevation in mrad
    # Values are multiplied by 0.5 to reflect heliostat deviation rather than beam deviation
    phi_el = 0.5*np.arctan(((dy*PEy)/R))*1000
    phi_az = 0.5*np.arctan(((dx*PEx)/R))*1000
    
    # area_inner_loc = np.where(area==np.min(area[np.nonzero(area)]))[0]
    # print("Azimuth tracking error: ", phi_az," mrad") 
    # print("Elevation tracking error: ", phi_el, " mrad")
    
    return [phi_az,phi_el]


list_ims = [croppedImDT,croppedImDT]
numcentroids = 2
list_centroids = [None]*numcentroids
list_centroids[0] = findcenter_KS(list_ims[0],fileNum)
list_centroids[1] = findcenter_DT(list_ims[1],fileNum)


# Export results
 
list_trackdevs = [None]*numcentroids*2
for idlc,lc in enumerate(list_centroids):
    trackdev = BCSresults(list_ims[idlc],lc)
    list_trackdevs[2*idlc] = trackdev[0] # azimuth
    list_trackdevs[2*idlc+1] = trackdev[1] # elevation
list_trackdevs.insert(0,submethod)
list_trackdevs.insert(0,imageFile)
BCSoutput = pd.DataFrame(data=[list_trackdevs],columns=('fileloc','cornermethod',
                                                        'az, KS','el, KS',
                                                      'az, DT','el,DT',))
BCSoutput.to_csv('CENERtestresults.csv',mode='a', index=False, header=False)
