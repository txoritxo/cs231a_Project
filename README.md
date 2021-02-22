# cs231a_Project

## Python Implementation
This software implements the camera planar self-calibration method described in the paper "Forget the checkerboard: Practical self-calibration using a planar scene" by D.Herrera et Al. Even though an open source C++ solution is available, it's implementation is oriented towards real-time calibration, making use of hard to follow optimizations and making extensive use of OpenGL GLSL shaders to accelerate the process which makes customization of the code quite a challenge. It has been considered more academic to implement from scratch a more intuitive and customizable solution -although obviously less computationally effcient- for the method described in the paper. Such implementation makes use of openCV and SciKit which enable very easy customizations on feature matching and detection algorithms or non-linear and minimum least square solvers to apply throughout the different steps of the algorithm.
The steps to conduct a planar self-calibration are as follows:
1. Place pictures of a planar structure somewhere within the ./data directory
2. open the file [run_planar_calibration](./python/run_planar_calibration.py) and edit the following:
   * `IMAGE_WIDTH = 800` . All images are resized when loaded, this line specifies the width of the target image, aspect ratio is the same as the original
   * modify the method `qload_images()` so that images are read from the appropriate path and with the right name
3. Execute `run_planar_calibration` which will do the following:
   * Load image files, resize them, and compute feature points using a SIFT detector
   * Conduct feature matching of all images with respect to the first one using a FLANN based matcher
   * Compute the appropriate homographies
   * Filter point correspondences to keep only those Points that are observed by at least `N` images. At this point `N=7`, change the `min_matches` parameter in the method `qfilter_matches`if needed
   * Perform the Projective Bundle Adjustment step of the algorithm after setting initial guess for camera center point and distortion. This step performs a minimum least squares using a cauchy loss function. The final term on the paper's equation (6) that regularizes the center of distortion has not been implemented yet. Apparently it is not completely necessary but maybe we can give it a try. It has to be noted that this step is quite slow, at this point no Jacobian Sparsity information has been introduced, which would speed up this step substantially
   * Perform the homography-based self calibration. This is a tricky step. An initial estimate of the focal length is obtained by applying LSTSQ to a system defined by the paper's equations (15) and (16) in which f^2 is the only variable; the issue is that there's no constraint to set f^2>0 and it may happen that the result is f^2<0, in such case we've taken it's absolute value in order to obtain an initial estimate of the focal length which for the tests done seem to provide reasonable results, **but this might be WRONG and it has no mathematical foundation -yet-**. Then, the actual homography-based self calibration step is conducted obtaining the intrinsic camera parameter matrix **K** and the normal vector to the planar structure used for calibration **n0** 
   * The remaining steps of the algorithm -Metric reconstruction and Metric bundle adjustment- are still WIP.

## C++ binary executables
Even though the open source c++ source code solution is available at this [link](https://github.com/plumonito/planecalib.git), its compilation is quite cumbersome, with several old versions of libraries and a number of modifications and workarounds to makefiles have to be made to get the code to compile. We've included precompiled binary files here, which have some limitations regarding the name and location of the test files. In order to execute the application proceed as follows:
1. To test a video file, its name shall be `Video1.mp4` and should be located in [./data/data4cpp](./data/data4cpp). Open a command line window, change directory to [./cpp](./cpp) and execute `planecalibVideo.exe`
2. To test a sequence of pictures, their names shall be DSCF0000.jpg, DSCF0001.jpg, etc. and shall be located in [./data/data4cpp](./data/data4cpp). Open a command line window, change directory to [./cpp](./cpp) and execute `planecalibPictures.exe`
