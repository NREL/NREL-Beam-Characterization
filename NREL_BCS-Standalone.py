#%% Module imports and setup
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import glob
from math import sqrt
import cv2
import skimage as sk
import imutils


# print("Imports complete.")

# Deliver inputs to the script
R = 200 # Heliostat-target distance, meters
W = 12 # Target width, meters
H = 12 # Target height, meters
# input_locdirectory = "Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/CrescentDunes"
input_locdirectory = r'C:\Users\DTSVANKI\OneDrive - NREL\BCS Comparison\CENER\data\raw_input\raw_input\CAT\03_22_2023\images'
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

# Process image using imported functions
from funcs_CornerFinder import * 
from funcs_TargetFinder import  *

img = readimage_KS(imageFile)
# img = imutils.rotate(img,180) # Just for CENER photos that are upside-down)
# plt.imshow(img,cmap=plt.cm.gray)
corners,submethod = findcorners_KS(img,fileNum)
print("Method ",submethod," chosen.")
#%% Process cropped image
from funcs_CentroidFinder import *
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

def tryKS():
    outputKS = pd.DataFrame()
    for kernelKS in range(1,7,2):
        for blurKS in range(3,6,2):
            for erodeKS in range(3,8,1):
                for dilateKS in range(13,18,1):
                    for cannylKS in range(7,10,1):
                        cannyuKS = cannylKS*2 + 3
                        paramsKS = {'kernelsize': kernelKS,
                                    'blursize': blurKS,
                                    'erodeI': erodeKS,
                                    'dilateI': dilateKS,
                                    'cannylower': cannylKS,
                                    'cannyupper': cannyuKS}
                        centroid = findcenter_KS(croppedImDT,fileNum,paramsKS)
                        if centroid == []:
                            dev_az = 0; dev_el = 0
                        else:
                            [dev_az,dev_el] = BCSresults(croppedImDT,centroid)
                        outputKS = pd.concat([outputKS,pd.DataFrame([[dev_az,dev_el,'KS',imageFilename,kernelKS,blurKS,erodeKS,dilateKS,cannylKS,cannyuKS]])])
    return outputKS
# output.to_csv('SensitivityStudy.csv',mode='a', index=False, header=False)

def tryDT():
    outputDT = pd.DataFrame()
    for bilatcs in range(39,64,6):
        for bilatd in range(3,16,3):
            for otsuDT in range(45,105,5):
                paramsDT = {'bilateralcs': bilatcs,
                            'bilaterald': bilatd,
                            'otsumultiplier': otsuDT/100}
                centroid = findcenter_DT(croppedImDT, fileNum,paramsDT)
                [dev_az,dev_el] = BCSresults(croppedImDT,centroid)
                outputDT = pd.concat([outputDT,pd.DataFrame([[dev_az,dev_el,'DT',imageFilename,bilatcs, bilatd, otsuDT/100]])])
    return outputDT

sensitivityKS = tryKS()
sensitivityDT = tryDT()

def macrohist(indata,intitle):
    plt.hist(indata,color='k')
    plt.title(intitle)

# Plot and store results
plt.figure(); fig,axs = plt.subplots(nrows = 2, ncols = 2,sharex=False,sharey=False,layout='constrained');
axs[0,0].hist(sensitivityKS[0], color='k');
axs[0,0].set_title('Azimuth, mrad, inner contour');
axs[0,1].hist(sensitivityKS[1], color='k');
axs[0,1].set_title('Elevation, mrad, inner contour');

axs[1,0].hist(sensitivityDT[0], color='k')
axs[1,0].set_title('Azimuth, mrad, outer contour');
axs[1,1].hist(sensitivityDT[1], color='k');
axs[1,1].set_title('Elevation, mrad, outer contour');

catloc = imageFile.find('CAT')
histimname = 'result_'+imageFile[catloc+4:-4].replace('\\','_')+'.png'
fig.savefig(histimname)
#%% 
# list_ims = [croppedImDT,croppedImDT]
# numcentroids = 2
# list_centroids = [None]*numcentroids
# list_centroids[0] = findcenter_KS(list_ims[0],fileNum)
# list_centroids[1] = findcenter_DT(list_ims[1],fileNum)


# # Export results
 
# list_trackdevs = [None]*numcentroids*2
# for idlc,lc in enumerate(list_centroids):
#     trackdev = BCSresults(list_ims[idlc],lc)
#     list_trackdevs[2*idlc] = trackdev[0] # azimuth
#     list_trackdevs[2*idlc+1] = trackdev[1] # elevation
# list_trackdevs.insert(0,submethod)
# list_trackdevs.insert(0,imageFile)
# BCSoutput = pd.DataFrame(data=[list_trackdevs],columns=('fileloc','cornermethod',
#                                                         'az, KS','el, KS',
#                                                       'az, DT','el,DT',))
# BCSoutput.to_csv('CENERtestresults.csv',mode='a', index=False, header=False)
