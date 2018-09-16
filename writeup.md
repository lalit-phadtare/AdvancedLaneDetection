## Advanced Lane Line Detection

---

[//]: # (Image References)

[image1]: ./test_images/test6.jpg "Input Image"
[image2]: ./output_images/Calib.jpg "Undistorted"
[image3]: ./output_images/test_undist_out7.jpg "Road Transformed"
[image4]: ./output_images/test_bin_out7.jpg "Binary Example"
[image5]: ./output_images/test_warped_out7.jpg "Warp Example"
[image6]: ./output_images/test_lanes_out7.jpg "Fit Visual"
[image7]: ./output_images/test_measured7.jpg "Output"
[video1]: ./project_out_video.mp4 "Video"


---

### Writeup / README


#### 1. Write-up 

This project tries to automatically detect lanes on the road in a video stream using some advanced techniques and measure lane curvature and car position offset from the lane center.
Here is an example of input image and final output image:

Input image:
![alt text][image1]

Output Image:
![alt text][image7]

This project was implemented as a part of [Udacity Self Driving Car Nanodegree coursework](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)

### Camera Calibration

As the first step, camera distortion is done to undistort all input images. This is done in the TunableParameters class. OpenCV's calibrateCamera gives the camera matrix and distortion coefficients for the camera. 

1. A set of 20 chessboard images which are taken using the same camera are used.
2. A 9x6 set of corners on each chessboard images are detected and are called the imgpoints.
3. A objpoints matrix which is a uniform 9x6 grid is created.
4. The calibrateCamera function calculates the camera matrix and distortion coefficients to get the mapping from imgpoints to objpoints.

The code is in `getCameraCalib()`  at advlanedet.py/line 145.

Here is an example of captured chessboard image and it's undistorted version calculated using the camera matrix and distortion coefficients.

![alt text][image2]

### Pipeline (single images)
A pipeline is implemented to test the processing steps over a set of test images and then I move on to use these steps on a video stream.
Here is an example of input frame from the camera.

![alt text][image1]

#### 1. Distortion corrected image

Here is an example of an undistorted image of one of the test images.

![alt text][image3]


#### 2. Binary Image 

Next, a thresholded binary image from the distortion corrected images is calculated. Following approaches were tried:

1. Gaussian smoothing of grayscale version followed by x-gradient, direction and magnitude detection using Sobel filter of size 3.
2. HLS conversion and using L for edge detection and S for yellow and white lane detection.

The final approach I selected was:

1. HSV conversion followed by thresholding the Sobel edge gradient and direction calculations on the V channel. Threshold range for 8 bit normalized  edge is [10, 255] and normalized direction is [0.25*pi 0.49*pi].
2. Threshold the V channel in range [0.7*255 1.0*255]. Values from this  [paper](https://pdfs.semanticscholar.org/88d7/c3fe5bbfe8f275541d344eb0b1c02ccc2779.pdf) were used as starting points.
3. The final binary image is the "AND" or all common high pixels after thresholding from 1 and 2 sub-steps above.

The code is in `getBinary()` at advlanedet.py/line 174.

Here is an example of binary thresholded image:

![alt text][image4]

#### 3. Perspective transform

Perspective transform is calculated in the TunableParameters class at advlanedet.py/line 55. 
For this I decided following set of points in the image:

1. src: Corners of a quadrilateral drawn around the lane region in the front-view camera image.
2. dst: These are the points where I would like the above corners to be mapped so that the image is a top-view image.

This is done in TunableParams class constructor at advlanedet.py/line 16.


```
src = np.float32(
    [[(img_size[0] / 2) - 62, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 62), img_size[1] / 2 + 100]])
	
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
	
```

This resulted in the following source and destination points:


| Source           | Destination   | 
|:----------------:|:-------------:| 
| 578, 460         | 320, 0        | 
| 203.33, 720      | 320, 720      |
| 1126.66, 720     | 960, 720      |
| 702, 460         | 960, 0        |


`getPerspectiveTransform` from OpenCV is used to take above points and calculate a mapping matrix M and its inverse Minv. The perspective transform was found to be working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
The warped image or the top view is as follows:

![alt text][image5]

#### 4. Line fit

To get lane pixels from the image from step 3:

1. Histogram method to detect seeds for lane pixel search is used.
2. Window search method to gather all lane pixels is used. Window size is 50 and number of windows are 9 per image. The minimum number of pixels to be found to call an update on centre of next window is 10.
3. A polynomial of degree 2 is fit through each of the right and left sets of lane pixels detected.

Follow the code from advlanedet.py/line 297 in `fit_polynomial()`

Here is an example of image lane line fit using the histogram and window search method.

![alt text][image6]

#### 5. Unwarp

1. The warped image and lane region is unwarped back using Minv calculated in step 3 to get back the front-view. This is done using the OpenCV warpPerspective() function mainly. The code is at advlanedet.py/line 500.

#### 6. Measurements

These include:
1. Lane curvature for left and right lanes are calculated using the polynomial coefficients calculated in 4. The formula is discussed [here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php)
2. Lane centre is measured by calculating the x intercept on left and right lanes lines and finding it's midpoint. Assuming image middle is camera position I take the difference in real world measurements and declare that as the car offset from center. 
2. Measurement results for each image is labeled on each unwarped frame. The code is in `measureResult()` at advlanedet.py/line 510.

Here is the final annotated image:

![alt text][image7]

---

### Pipeline (video)

The pipeline for images is used as is for video expect at the "Line fit" where it is optimized to use the temporal information. Here is the "Line fit" stage for video:
1. "Line" class objects are used for left and right lane to hold information about lanes detected in last frames.
2. Histogram method to detect seeds for lane pixel search is used if no information of a recently detected line exists.
2. Window search method to gather all lane pixels is used. Window size is 50 and number of windows are 9 per image. The minimum number of pixels to be found to call an update on centre of next window is 10.
3. A polynomial of degree 2 is fit through each of the right and left sets of lane pixels detected.
4. For video, it is further optimized to use the polynomial coefficients from last frame if they were detected as search seeds, skipping the step 1.
5. For video, the coefficients are smoothened over last 10 frames to avoid the lines from jumping around.
6. If there are no lines detected for 25 frames, I hold on to last lines and start from scratch if it is more than 25

The optimized line fit function is in `fit_polynomial_prior()` at advlanedet.py/line 369.
The logic to decide whether to start lane detection with prior or scratch is in videoPipeline() at line advlanedet.py/560.

Here's a [link to my video result](./project_out_video.mp4)

---

### Discussion

1. Top view helps in predicting the lane line polynomial more clearly.
2. Getting a good binary thresholded image for every frame is difficult due to road texture, horizontal patterns on the bridge, and shadows. This gets unwanted pixels in the window.
3. I used 25 fps of camera to calculate most metrics like smoothing and number of undetected frames to let go. Speed of vehicle can also be an important factor here to consider.
4. Extreme curvature of road, road elevations etc. throw off the lane predictions as the points used in perspective warping are hard coded.
5. The same pipeline almost works for challenge_video. It fails when car is under the bridge where lighting is very less to detect the whole yellow lane line on left.
6. I tried using polynomial of degree 3 for the harder challenge thinking that it will detect the double bends but the sudden change from single curve to double bend is still not handled. The code is written to handle degree 2 and 3 polynomial.

