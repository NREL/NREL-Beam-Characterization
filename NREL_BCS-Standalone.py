#%% Module imports and setup
import numpy as np 
import matplotlib.pyplot as plt
import glob
from math import sqrt
# print("Imports complete.")

#%% Deliver inputs to the script
R = 120 # Heliostat-target distance, meters
W = 2.0 # Target width, meters
H = 2.2 # Target height, meters
input_locdirectory = "Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/CrescentDunes"
input_filetype = "bmp"

'''
Choosing image files to process
   
'''
imageFilename = input_locdirectory + "/*." + input_filetype
imageFiles = sorted(glob.glob(imageFilename),key=len)

for ii in range(len(imageFiles)):
  print(ii+1, ",", imageFiles[ii])
fileNum = int(input("Enter file number: "))
imageFile = imageFiles[fileNum-1]

print("Selected file: ",imageFile)

#%% Process image using imported functions 
from funcs_TargetFinder import  readimage_KS, findcorners_KS, cropimage_KS, cropimage_DT
from funcs_CentroidFinder import findcenter_KS, findcenter_DT

img = readimage_KS(imageFile)
corners = findcorners_KS(img,fileNum)
croppedIm = cropimage_DT(img,corners)
Centroid = findcenter_DT(croppedIm,fileNum)

  
#finding ∆r
target_mid_cropped = np.flipud(np.shape(croppedIm))/2
dx = abs(Centroid[0]-target_mid_cropped[0])
dy = abs(Centroid[1]-target_mid_cropped[1])
dr = sqrt((dx)**2 +(dy)**2)


#finding the pixel extent
PEx = W/np.shape(croppedIm)[1]
PEy = H/np.shape(croppedIm)[0]

#finding Phi altitude and elevation
phi_el = np.arctan(((dy*PEy)/R))
phi_az = np.arctan(((dx*PEx)/R))

# area_inner_loc = np.where(area==np.min(area[np.nonzero(area)]))[0]
print("Azimuth tracking error: ", phi_az*1000," mrad") 
print("Elevation tracking error: ", phi_el*1000, " mrad")