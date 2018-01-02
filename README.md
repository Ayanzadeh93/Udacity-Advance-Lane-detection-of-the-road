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



------------------------------------------------
you need to install  Python 3 beside with the numpy, opencv and matplotlib  packages

please install following dependecies:

------------------------------------------------------------
## Linux/Ubuntu

installing anaconda(python3 version)

if you have install anaconda for python 2.7 you should have install new environment in that version

> conda create --name=yourNewEnvironment python=3 anaconda
>> source activate [environmentname]


#pip install pillow
#conda install -c https://conda.anaconda.org/menpo opencv3

after installing check the Cv2 package with this:

>python
>> import cv2

for using  video package please install moviepy

#pip install moviepy



---------------------------------------------------------------------------------------------------------------------
## windows (not recommened)

There is not any package for opencv in anaconda (you have to install opencv with the normal mode)


--------------------------------------------------------------------------------------------------------------------

## Additional resources for autonomous dribvers


1-Nanodegrees Programs: https://www.udacity.com/nanodegree
Nanodegree Plus (job guarantee): https://www.udacity.com/nanodegree/plus 
UConnect (weekly in-person study sessions):  https://www.udacity.com/uconnect 

2-Courses on Udacity Machine Learning Engineer Nanodegree by Google (Currently Available): https://www.udacity.com/course/machine-learning-engineer-nanodegree-by-google--nd009

3-Artificial Intelligence for Robots (Free Course) https://www.udacity.com/course/artificial-intelligence-for-robotics--cs373

3-Intro to Statistics (Free Course) https://www.udacity.com/course/intro-to-statistics--st101

4-Deep Learning (Free Course) https://www.udacity.com/course/deep-learning--ud730

5-Programming Foundations with Python (Free Course) https://www.udacity.com/course/programming-foundations-with-python--ud036 

6-Introduction to Computer Vision: https://www.udacity.com/course/introduction-to-computer-vision--ud810
Cool topics for self driving car course should cover: 
Deep Learning, Computer Vision, Vehicle Dynamics, Controllers, Localization, Mapping (SLAM), Sensors & Fusion

### Reading Resources Udacity

https://medium.com/udacity/self-driving-car-employers-f24c0013cf1d#.3jlgb1c1i 

https://www.quora.com/Are-Udacity-Nanodegrees-worth-it-for-finding-a-job

http://blog.udacity.com/2015/03/udacity-nanodegree-reviews-your-questions-answered.html

http://blog.udacity.com/2015/03/udacity-nanodegree-reviews-your-questions-answered.html


### News / Resources

http://www.bbc.com/news/technology-36952252

https://techcrunch.com/2016/03/11/gm-buys-self-driving-tech-startup-cruise-as-part-of-a-plan-to-make-driverless-cars/

http://money.cnn.com/2016/04/04/technology/george-hotz-comma-ai-andreessen-horowitz/

https://techcrunch.com/2016/06/30/zoox-raises-200-million-at-1-billion-valuation-for-its-self-driving-cars/

https://www.youtube.com/watch?v=fQmOpxEvpvI

http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

https://www.cbinsights.com/blog/autonomous-driverless-vehicles-corporations-list/ 

http://www.theverge.com/2016/6/6/11866868/comma-ai-george-hotz-interview-self-driving-cars

Trucking Industry: http://ot.to/

Racing Industry: https://blogs.nvidia.com/blog/2016/06/03/autonomous-vehicles/

### Open Source Projects

https://www.reddit.com/r/SelfDrivingCars/comments/4vz3y7/commaai_for_the_people_to_experiment_with_too/

### Datasets

https://www.cityscapes-dataset.com/

http://robotcar-dataset.robots.ox.ac.uk/examples/ 

http://selfracingcars.com/blog/2016/7/26/polysync

http://data.selfracingcars.com/

### Other Resources
Stanford Convolutional Neural Networks for Visual Recognition http://cs231n.github.io/
Deep Learning Framework written in Swift to use on apple devices (written by @amund) http://deeplearningkit.org/
Image segmentation from comma.ai https://commaai.blogspot.de/2016/07/self-coloring-books.html?m=1

Hee Lee, Gim, Friedrich Faundorfer, and Marc Pollefeys. "Motion estimation for self-driving
cars with a generalized camera." ​ Proceedings of the IEEE Conference on Computer Vision and
Pattern ​ ​ Recognition ​ . ​ ​ 2013.

Levinson, Jesse, et al. "Towards fully autonomous driving: Systems and algorithms."
Intelligent ​ ​ Vehicles ​ ​ Symposium ​ ​ (IV), ​ ​ 2011 ​ ​ IEEE ​ . ​ ​ IEEE,​ ​ 2011.


M. Bojarski, D. Del Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. D. Jackel, M.
Monfort, U. Muller, J. Zhang, et al. End to end learning for self-driving cars.arXiv preprint
arXiv:1604.07316,​ ​ 2016

H. Xu, Y. Gao, F. Yu, and T. Darrell. End-to-end learning of driving models from large-scale
video​ ​ datasets.​ ​ arXiv​ ​ preprint​ ​ arXiv:1612.01079,​ ​ 2016.


C. Szegedy et al., “Going deeper with convolutions,” Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit., vol. 07–12–June, pp. 1–9, 2015.


Udacity.Public​ ​ driving​ ​ dataset.​ ​ https://www.udacity.com/self-driving-car,​ ​ 2017.​ ​ [Online;
accessed​ ​ 07-Mar-2017].


Comma.ai.Public accessed​ ​ 07-Mar-2017].driving   dataset.https://github.com/commaai/research,2017.[Online;


Dickmanns and B. Mysliwetz, “Recursive 3-D road and relative ego- state recognition,”
IEEE​ ​ Trans.​ ​ Pattern​ ​ Anal.​ ​ Mach.​ ​ Intell.,​ ​ vol.​ ​ 14,​ ​ no.​ ​ 2,​ ​ pp.​ ​ 199–213,​ ​ Feb.​ ​ 1992.
\bibitem{car accident}
​$ Car​ ​ accidents.​ http://en.wikipedia.org/wiki/car_accident $


Caltech​ ​ Lanes​ ​ datasets​ , ​ ​ http://www.mohamedaly.info/datasets/caltech-lanes


T. Saudi, J. Hijazi, and J. Sulaiman, “Fast lane detection with randomized hough
transform,”​ ​ in​ ​ Proc.​ ​ Symp.​ ​ Inf.​ ​ Technol.,​ ​ 2008,​ ​ vol.​ ​ 4,​ ​ pp.​ ​ 1–5


J. Wang, Y. Wu, Z. Liang, and Y. Xi, “Lane detection based on random hough transform
on​ ​ region​ ​ of​ ​ interesting,”​ ​ in​ ​ Proc.​ ​ IEEE​ ​ Conf.​ ​ Inform.​ ​ Autom.,​ ​ 2010,​ ​ pp.​ ​ 1735–1740.



    2-Apply a distortion correction to raw images.
    3-Use color transforms, gradients, etc., to create a thresholded binary image.
    4-Apply a perspective transform to rectify binary image ("birds-eye view").
    5-Detect lane pixels and fit to find the lane boundary.
    6-Determine the curvature of the lane and vehicle position with respect to center.
    7-Warp the detected lane boundaries back onto the original image.
    8-Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



