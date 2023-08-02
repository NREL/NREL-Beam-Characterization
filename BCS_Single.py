import numpy as np 
import matplotlib.pyplot as plt
from glob import glob 
import glob
import pandas as pd 
import os 
import cv2 
from scipy.io import loadmat 
import argparse 
import imutils 
from skimage import data 
from skimage.feature import blob_dog, blob_log, blob_doh 
from math import sqrt 
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.filters import gaussian 
from skimage import img_as_ubyte
from collections import defaultdict
import sys
from skimage.transform import hough_line, hough_line_peaks
from bisect import bisect 
import pypylon



#imageFiles = sorted(glob.glob("Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/CrescentDunes/*.bmp"),key=len)
#imageFiles = sorted(glob.glob("Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/OTF.07.21.22/otf54/*.jpg"),key=len)
#imageFiles = sorted(glob.glob("Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/OTF.07.21.22/otf75/*.jpg"),key=len)
#imageFiles = sorted(glob.glob("Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/OTF.07.21.22/otf79/*.jpg"),key=len)
#imageFiles = sorted(glob.glob("Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/OTF.07.08.22/*.jpg"),key=len)

from pypylon import pylon
import platform

num_img_to_save = 1
img = pylon.PylonImage()

tlf = pylon.TlFactory.GetInstance()
tl = tlf.CreateTl('BaslerGigE')
cam_info = tl.CreateDeviceInfo()
cam_info.SetIpAddress('192.168.3.3')
cam = pylon.InstantCamera(tlf.CreateDevice(cam_info))

cam.Open()
# load manually set camera settings from in pylon app
pylon.FeaturePersistence.Load("cam.pfs", cam.GetNodeMap(), True)

cam.StartGrabbing()
for i in range(num_img_to_save):
    with cam.RetrieveResult(2000) as result:

        # Calling AttachGrabResultBuffer creates another reference to the
        # grab result buffer. This prevents the buffer's reuse for grabbing.
        img.AttachGrabResultBuffer(result)

            # The JPEG format that is used here supports adjusting the image
            # quality (100 -> best quality, 0 -> poor quality).
        ipo = pylon.ImagePersistenceOptions()
        quality = 100
        ipo.SetQuality(quality)

        filename = "test_target.jpeg"
        img.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)

        # In order to make it possible to reuse the grab result for grabbing
        # again, we have to release the image (effectively emptying the
        # image object).
        img.GetBuffer
        img.Release()

cam.StopGrabbing()
cam.Close()

#reading in selected file
img =imread(filename, as_gray = True)


plt.imshow(img)

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

w = img.shape
h=img.shape
c=img.shape
img.dtype
print(w)


alpha = 2
beta = 50

im = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)

plt.imshow(im, cmap='gray')


#Making Binary Image 
kernel = np.ones((3,3),np.uint8)
blur = cv2.blur(im, (3,3));
erodeI = 3
dilateI = 3
imerode = cv2.erode(blur,kernel,iterations = erodeI)
im_dilate = cv2.dilate(imerode,kernel,iterations = dilateI)
bin_img = cv2.adaptiveThreshold(imerode, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
bin_img = ~bin_img

plt.axis('off')


fileNum = iFile+1

plt.imshow(bin_img, cmap='gray')


from skimage.transform import hough_line, hough_line_peaks


tested_angles = np.linspace((-89.03*np.pi)/180, (90.2*np.pi)/180, 100) 
#tested_angles = np.linspace((-np.pi)/2, (np.pi)/2, 100)
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


from pyparsing.util import lineno
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

#All possible intersections (Some will not yield any intersections)

int_1 = intersector(slopes[0], y_ints[0], slopes[1], y_ints[1])
int_2 = intersector(slopes[0], y_ints[0], slopes[2], y_ints[2])
int_3 = intersector(slopes[0], y_ints[0], slopes[3], y_ints[3])
int_4 = intersector(slopes[1], y_ints[1], slopes[2], y_ints[2])
int_5 = intersector(slopes[1], y_ints[1], slopes[3], y_ints[3])
int_6 = intersector(slopes[2], y_ints[2], slopes[3], y_ints[3])

#sort intersections
#There's always going to be two that don't work, these two are always going to be the first two since they are large positive numbers
def get_max(sub):
  return max(sub)

test_list = [int_1, int_2, int_3, int_4, int_5, int_6]
test_list.sort(key = get_max, reverse = True)
#print("Sorted Tuples: " + str(test_list))

int_TL =  test_list[5]
int_TR =  test_list[3]
int_LL = test_list[4]
int_LR = test_list[2]

print("Top Left Corner: ", int_TL)
print("Top Right Corner: ", int_TR)
print("Lower Right Corner: ",int_LR)
print("Lower Left Corner: ",int_LL)

corners = [(int_TL),(int_TR), (int_LR), (int_LL)]
x_val = [x[0] for x in corners]
y_val = [x[1] for x in corners]
plt.plot(x_val,y_val, 'bo')

#finding the center of the target given the coordinates of the edges 
x_midpt = 0.5*(int_TL[0]+int_LR[0])
y_midpt = 0.5*(int_TL[1] + int_LR[1])
plt.plot(x_midpt,y_midpt, "o")
target_mid = (x_midpt, y_midpt)
print("Center of the Target: ",target_mid)


#finding pixels across target
Px = 0.5*(abs(int_TL[0] - int_TR[0]) + abs(int_LL[0] - int_LR[0]))
Py = 0.5*(abs(int_TL[1] - int_LL[1]) + abs(int_TR[1] - int_LR[1]))


plt.axis('off')
#plt.title("Detected Lines")
#plt.savefig('/content/drive/MyDrive/NREL/ProcessedIms/DetectedEdges/CrescentDunes/' + "Image" + str(fileNum) + 'Detected', dpi = 300, bbox_inches='tight', pad_inches=0)
plt.show()


import csv 
f = open('Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/DataFiles/CrescentDunes/TargetEdges/TargetEdges' + "File" + str(fileNum), 'w')
theWriter = csv.writer(f)
theWriter.writerow(['Image','Alpa', 'Beta','Erosions','Dilations','Top Left Corner','Top Right Corner','Lower Left Corner','Lower Right Corner'])
theWriter.writerow([fileNum,alpha,beta,erodeI,dilateI,int_TL,int_TR,int_LL,int_LR])


croppedIm = img[int(int_TL[1]):int(int_LL[1]), int(int_LL[0]):int(int_LR[0])]
croppedIm = cv2.resize(croppedIm, (1936,1456))
plt.axis("off")
plt.imshow(croppedIm, cmap = plt.cm.gray)


kernel = np.ones((3,3),np.uint8)
blur = cv2.blur(croppedIm, (3,3))
erodeI = 5
dilateI = 15
imerode = cv2.erode(blur,kernel,iterations = erodeI)
im_dilate = cv2.dilate(imerode,kernel,iterations = dilateI)

imedge = cv2.Canny(im_dilate, 8,19)
img_th = cv2.adaptiveThreshold(imedge, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
#img_th=~img_th  
#finding contours is easier when the contours themselves are in white. The contour I am looking for is the innermost contour that is in white. 

plt.axis('off')

#morphlogy to remove unwanted noise
img_th = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
plt.imshow(img_th, cmap='gray')


contours, heirarchy = cv2.findContours(img_th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#img_th=~img_th
src_copy =img_th.copy()
src_copy = cv2.cvtColor(src_copy, cv2.COLOR_BGR2RGB)

for i, cont in enumerate(contours):
  if heirarchy[0][i][3] == -1:
    src_copy = cv2.drawContours(src_copy, cont, -1, (0,255,0), 5)
  else:
    src_copy = cv2.drawContours(src_copy, cont, -1, (0,0,255),5)

print("Numer of Contours: {}".format(len(contours)))
#sorted_contours = sorted(contour1, key=cv2.contourArea, reverse= True)
#for i, cont in enumerate(sorted_contours[:3],1):
 # cv2.drawContours(img, contour1, -1, (0,255,0), 3)
  #cv2.putText(im, str(i), (cont[0,0,1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 4)


plt.figure(figsize=[5,5])
plt.imshow(src_copy);plt.title("Beam Contour");plt.axis('off')


#R = distance heliostat to target in meters
# W = width of the target in meters
# H = height of the target in meters
R = 120
W = 2.0 
H = 2.2


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

max_area = 500000
min_area = 10000


for i, cont in enumerate(cnts):
  if heirarchy[0][i][3] == -1:
    src_copy = cv2.drawContours(src_copy, cont, -1, (0,255,0), 5)
  else:
    src_copy = cv2.drawContours(src_copy, cont, -1, (0,0,255),5)
  
  
for cnt in cnts: 
  area = cv2.contourArea(cnt)
  if min_area <= area <= max_area:
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(src_copy, ellipse, (255,0,0), 10, cv2.LINE_AA)
    M = cv2.moments(cnt)
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])
    cv2.circle(src_copy, (cX,cY), 7, (255,0,0), -1)
    print("Centroid: ",(cX,cY))
    eccentricity = eccentricity_from_ellipse(cnt)
    print("Eccentricity: ",eccentricity)
    f1 = open('Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/DataFiles/CrescentDunes/BeamContour/BeamContour' + "File" + str(fileNum), 'w')
    theWriter1 = csv.writer(f1)
    theWriter1.writerow(['Centroids','Eccentricity'])
    theWriter1.writerow([(cX,cY), eccentricity])
    
    Centroid = (cX,cY)
    print("Centroid: ",(cX,cY))

    #finding ∆r 
    dx = abs(Centroid[0]-target_mid[0])
    dy = abs(Centroid[1]-target_mid[1])
    dr = sqrt((dx)**2 +(dy)**2)
    print("∆x: ",dx)
    print("∆y: ", dy)
    print("∆r" , dr)

    #finding the pixel extent
    PEx = W/Px
    PEy = H/Py
    print("Pixel Extent: ", (PEx,PEy))

    #finding Phi altitude 
    phi_alt = arctan(((dy*PEy)/R))
    print("Altitude tracking error in meters: ",phi_alt) # in meters


    #finding Phi elevation 
    phi_elv = arctan(((dx*PEx)/R))
    print("Tracking error in Elevation: ", phi_elv) # in meters

   
plt.figure(figsize=[5,5])
plt.axis('off')
plt.imshow(src_copy)

plt.savefig('Y:/5700/SolarElectric/PROJECTS/38488_HelioCon_Zhu/BeamCharacterizationSystems/DataFiles/CrescentDunes/ProcessedIms/BeamDetection/' + "Image" + str(fileNum), bbox_inches='tight', pad_inches=0)