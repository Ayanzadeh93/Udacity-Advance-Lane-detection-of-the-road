Advanced Lane Detection
===

![sdc-1-1024x427](https://user-images.githubusercontent.com/25509152/33837471-770be3a2-de9d-11e7-97cc-50c6224bfcc3.png)

- EXECUTIVE SUMMARY

In this Advanced Lane Detection project, we apply computer vision techniques to augment video output with a detected road lane, road radius curvature and road centre offset. The video was supplied by Udacity and captured using the middle camera.

The goals / steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.





===================
you need to install  Python 3 beside with the numpy, opencv and matplotlib  packages

please install following dependecies:

================================================================================================================
Linux/Ubuntu

installing anaconda(python3 version)

if you have install anaconda for python 2.7 you should have install new environment in that version

> conda create --name=yourNewEnvironment python=3 anaconda
> source activate [environmentname]


# pip install pillow
#conda install -c https://conda.anaconda.org/menpo opencv3

after installing check the Cv2 package with this:

>python
>> import cv2

for using  video package please install moviepy

#pip install moviepy



---------------------------------------------------------------------------------------------------------------------
windows (not recommened)

There is not any package for opencv in anaconda (you have to install opencv with the normal mode)


--------------------------------------------------------------------------------------------------------------------

    1-Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    2-Apply a distortion correction to raw images.
    3-Use color transforms, gradients, etc., to create a thresholded binary image.
    4-Apply a perspective transform to rectify binary image ("birds-eye view").
    5-Detect lane pixels and fit to find the lane boundary.
    6-Determine the curvature of the lane and vehicle position with respect to center.
    7-Warp the detected lane boundaries back onto the original image.
    8-Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



