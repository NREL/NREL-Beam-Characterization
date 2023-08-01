import argparse
from pypylon import pylon
import time


height, width, n, = 2160, 3840, 1
image_size = height * width

output_path = 'C:/Users/jdeblooi/OneDrive - NREL/BCS_Work/Saved_Photos/Camera_Testing'

img = pylon.PylonImage()

cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
cam.Open()
cam.Height.SetValue(height)
cam.Width.SetValue(width)
cam.StartGrabbing()

for m in range(n):
    with cam.RetrieveResult(2000) as res:
        t1 = time.time()

        img.AttachGrabResultBuffer(res)
        img.Save(pylon.ImageFileFormat_Tiff, output_path + 'test.tif')

        t2 = time.time()
        print('fps: %.2f' % (1 / (t2 - t1)), \
              'bandwidth: %.2f MB/s' % (image_size / 1024**2 / (t2 - t1)))

cam.StopGrabbing()
cam.Close()