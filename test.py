# ===============================================================================
#    This sample illustrates how to grab and process images using the CInstantCamera class.
#    The images are grabbed and processed asynchronously, i.e.,
#    while the application is processing a buffer, the acquisition of the next buffer is done
#    in parallel.
#
#    The CInstantCamera class uses a pool of buffers to retrieve image data
#    from the camera device. Once a buffer is filled and ready,
#    the buffer can be retrieved from the camera object for processing. The buffer
#    and additional image data are collected in a grab result. The grab result is
#    held by a smart pointer after retrieval. The buffer is automatically reused
#    when explicitly released or when the smart pointer object is destroyed.
# ===============================================================================
from pypylon import pylon
from pypylon import genicam
from PIL import Image

import sys

# Number of images to be grabbed.
countOfImagesToGrab = 10

# The exit code of the sample application.
exitCode = 0

try:
    # Create an instant camera object with the camera device found first.
    tlf = pylon.TlFactory.GetInstance()
    tl = tlf.CreateTl('BaslerGigE')
    cam_info = tl.CreateDeviceInfo()
    cam_info.SetIpAddress('192.168.3.3')
    cam = pylon.InstantCamera(tlf.CreateDevice(cam_info))
    cam.Open()

    # Print the model name of the camera.
    print("Using device ", cam.GetDeviceInfo().GetModelName())

    # demonstrate some feature access
    new_width = cam.Width.GetValue() - cam.Width.GetInc()
    if new_width >= cam.Width.GetMin():
        cam.Width.SetValue(new_width)

    # The parameter MaxNumBuffer can be used to control the count of buffers
    # allocated for grabbing. The default value of this parameter is 10.
    cam.MaxNumBuffer = 5
    cam.PixelFormat.SetValue("Mono12")
    # Start the grabbing of c_countOfImagesToGrab images.
    # The camera device is parameterized with a default configuration which
    # sets up free-running continuous acquisition.
    cam.StartGrabbingMax(countOfImagesToGrab)

    # Camera.StopGrabbing() is called automatically by the RetrieveResult() method
    # when c_countOfImagesToGrab images have been retrieved.
    while cam.IsGrabbing():
        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            # Access the image data.
            print("SizeX: ", grabResult.Width)
            print("SizeY: ", grabResult.Height)
            img = grabResult.Array
            print("Gray value of first pixel: ", img[0, 0])
            print(f'Size of image: {img.shape}')
            image = Image.fromarray(img)
            image.show() 

        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        grabResult.Release()
    cam.Close()

except genicam.GenericException as e:
    # Error handling.
    print("An exception occurred.")
    print(e.GetDescription())
    exitCode = 1

sys.exit(exitCode)