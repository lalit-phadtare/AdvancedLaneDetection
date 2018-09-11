# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 21:57:44 2018

@author: Lalit
"""

import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip
from tempfile import TemporaryFile


class TunableParams():
    '''
    class where we set all tunable params across the pipeline.
    '''
    
    def __init__(self):
        #read in images
        calibInSet = glob.glob('camera_cal/*.jpg')
        timages = glob.glob('test_images/*.jpg')
        
        ret, cameraMatrix, distCoeffs, rvecs, tvecs = getCameraCalib(calibInSet)
        if ret:
            self.cameraMatrix, self.distCoeffs, self.rvecs, self.tvecs = cameraMatrix, distCoeffs, rvecs, tvecs
        
        #store output for undistortion
        calibIn = mpimg.imread(calibInSet[0])
        calibOut = cv2.undistort(calibIn, cameraMatrix, distCoeffs, None, cameraMatrix)
        fig = plt.figure()
        ax = plt.subplot(121)
        ax.imshow(calibIn)
        plt.title('Original/Distortion')
        ax = plt.subplot(122)
        ax.imshow(calibOut)
        plt.title('Unidistorted')
        fig.savefig('output_images/Calib.jpg')
        
        #get perspective matrix
        img_size = (mpimg.imread(timages[0])).shape[0:2][::-1]
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
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
    
        #binary thresholding params
        self.sx_thresh=(1, 255)
        self.s_thresh=(200, 255)
        self.hhsv_thresh=(0.0*255, 0.1*255)
        self.shsv_thresh1=(0.0*255, 0.2*255)
        self.shsv_thresh2=(0.4*255, 1.0*255)
        self.vhsv_thresh=(0.7*255, 1.0*255)
        self.sdir_thresh=(0.25*np.pi, 0.49*np.pi)
        self.sgrad_thresh=(10, 255)
        
        #smoothing for coeff predicition
        self.smoothing = 10
        
        #realworld to pixels conversion
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        #window setting in sliding window search
        self.nwindows = 9
        self.margin = 50
        self.minpix = 10
        
        #gaussian kernel size
        self.gkernel_size = 9 
        
        #degree of polynomial to fit
        self.degree = 2
        


class Line():
    '''
    This class object holds the line data going from previous to next frame
    '''
    
    def __init__(self, smooth_n=3, numcoeffs=3):
        # polynomial degree or number of coeffs.
        self.numcoeffs = numcoeffs
        #no. of frames to smooth over
        self.smoothing = smooth_n
        # was the line detected in the last iteration?
        self.detected = False  
        # recent fit coeffs
        self.recent_coeffs = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #curr x
        self.curr_x = []  
        #curr y
        self.curr_y = []  
        
        
    def addCurrCoeffs(self, coeffs):
        '''
        add the current fit coeffs. This also means that a line was 
        successfully detected.
        '''
        
        self.detected = True
        self.recent_coeffs.extend(coeffs)
        if len(self.recent_coeffs) > self.smoothing*self.numcoeffs:
            for ii in range(self.numcoeffs):
                self.recent_coeffs.pop(0)
                
                
    def getSmoothenedCoeffs(self):
        '''
        get average of line polynomial coeffs upto last 3 frames
        '''
        
        sumarr, count = np.zeros(self.numcoeffs),0
        for ii in range(0, len(self.recent_coeffs)-1, self.numcoeffs):
            sumarr += self.recent_coeffs[ii:ii+self.numcoeffs]
            #a += self.recent_coeffs[ii]
            #b += self.recent_coeffs[ii+1]
            #c += self.recent_coeffs[ii+2]
            count += 1
        self.best_fit = np.divide(sumarr, count)
        return self.best_fit
    
    def addCurrFit(self, x_fit, y_fit):
        '''
        add all the fitted x coordinates of last frame
        '''
        self.curr_x = x_fit
        self.curr_y = y_fit
        
        
def getCameraCalib(images, cr = 9, cc = 6):
    '''
    use the chessboard images to get camera calibration matrix
    '''
    
    objpoints = []
    imgpoints = []
    
    objp = np.zeros((cc*cr,3), np.float32)
    objp[:,:2] = np.mgrid[0:cr, 0:cc].T.reshape(-1,2)
    
    for fname in images:
        
        img = mpimg.imread(fname)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        ret, corners = cv2.findChessboardCorners(gray, (cr, cc), None)
        
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
            
            #img = cv2.drawChessboardCorners(img, (cr,cc), corners, ret)
            #plt.imshow(img)
#        else:
            #print('no points found in ' + fname)


    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return ret, cameraMatrix, distCoeffs, rvecs, tvecs


def getBinary(img,  tparams):
    '''
    threshold the color image using various to get high gradient pixels
    '''
   
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    #hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #h_channel = hls[:,:,0]
    #l_channel = hls[:,:,1]
    #s_channel = hls[:,:,2]
    
    #lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    #llab_channel = lab[:,:,0]
    #alab_channel = lab[:,:,1]
    #blab_channel = lab[:,:,2]
    
    #HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #hhsv_channel = hsv[:,:,0]
    #hhsv_bin = np.zeros_like(hhsv_channel)
    #hhsv_bin[(hhsv_channel >= tparams.hhsv_thresh[0]) & (hhsv_channel <= tparams.hhsv_thresh[1])] = 1
    #shsv_channel = hsv[:,:,1]
    #shsv_bin = np.zeros_like(shsv_channel)
    #shsv_bin[(shsv_channel >= tparams.shsv_thresh1[0]) & (shsv_channel <= tparams.shsv_thresh1[1])] = 1
    #shsv_bin[(shsv_channel >= tparams.shsv_thresh1[0]) & (shsv_channel <= tparams.shsv_thresh1[1])] = 1
    vhsv_channel = hsv[:,:,2]
    vhsv_bin = np.zeros_like(vhsv_channel)
    vhsv_bin[(vhsv_channel >= tparams.vhsv_thresh[0]) & (vhsv_channel <= tparams.vhsv_thresh[1])] = 1
    
    
    ## Define a kernel size and apply Gaussian smoothing
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #blur_gray_channel = cv2.GaussianBlur(gray, (tparams.gkernel_size, tparams.gkernel_size), 0)
    
    ## Define a kernel size and apply Gaussian smoothing
    #blur_l_channel = cv2.GaussianBlur(l_channel, (tparams.gkernel_size, tparams.gkernel_size), 0)
    
    # Sobel x
    sobelx = cv2.Sobel(vhsv_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    sobely = cv2.Sobel(vhsv_channel, cv2.CV_64F, 0, 1) # Take the derivative in y
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    gradmag = np.sqrt(sobelx**2 + sobely**2)
	 # Rescale to 8 bit
    scale_factor = 255/np.max(gradmag)
    gradmag = (gradmag * scale_factor).astype(np.uint8)
    gradmag_binary = np.zeros_like(gradmag)
    gradmag_binary[(gradmag >= tparams.sgrad_thresh[0]) & (gradmag <= tparams.sgrad_thresh[1])] = 1
    
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dirmag_binary =  np.zeros_like(absgraddir)
    dirmag_binary[(absgraddir >= tparams.sdir_thresh[0]) & (absgraddir <= tparams.sdir_thresh[1])] = 1
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= tparams.sx_thresh[0]) & (scaled_sobel <= tparams.sx_thresh[1])] = 1
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    #combined_binary[(gradmag_binary == 1) | ((sxbinary == 1) & (dirmag_binary == 1)) | (vhsv_bin == 1)] = 1
    combined_binary[(vhsv_bin == 1) & ((gradmag_binary == 1) & (dirmag_binary == 1))] = 1
#    
#    mask = np.zeros_like(vhsv_bin)  
#    img_size = vhsv_bin.shape[0:2][::-1]
#    vertices = np.float32(
#            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
#            [((img_size[0] / 6) - 100), img_size[1]-50],
#            [(img_size[0] * 5 / 6) + 100, img_size[1]-50],
#            [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
#    
#    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
#    if len(vhsv_bin.shape) > 2:
#        channel_count = vhsv_bin.shape[2]  # i.e. 3 or 4 depending on your image
#        ignore_mask_color = (255,) * channel_count
#    else:
#        ignore_mask_color = 255
        
    ##filling pixels inside the polygon defined by "vertices" with the fill color    
    #cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), ignore_mask_color)
    
    ##returning the image only where mask pixels are nonzero
    #masked_image = cv2.bitwise_and(combined_binary, mask)
    
    
    return combined_binary


def find_lane_pixels(binary_warped, tparams):
    '''
    find lane pixels by taking a histogram across the image height and assigning peaks as the lane positions.
    '''
    nwindows = tparams.nwindows
    margin = tparams.margin
    minpix = tparams.minpix
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, tparams, save = False, name='temp.jpg'):
    '''
    Find lane pixels and fit a polynomial through it
    '''
    
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, tparams)

    
    try:
        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, tparams.degree)
        right_fit = np.polyfit(righty, rightx, tparams.degree)
        #print(left_fit, right_fit)
    
        # Generate x and y values for plotting
        ploty = np.linspace(25, binary_warped.shape[0]-1, binary_warped.shape[0] )
        
        left_fitx = np.zeros_like(ploty)
        right_fitx = np.zeros_like(ploty)
        for i in range(tparams.degree + 1):
            left_fitx += left_fit[i]*(ploty**(tparams.degree-i))
            right_fitx += right_fit[i]*(ploty**(tparams.degree-i))
            
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        #print('The function failed to fit a line!')
        out_img, left_fitx, right_fitx, ploty, left_fit, right_fit = [], None, None, None, None, None
        return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit
    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    if save:
        fig,ax = plt.subplots()
        ax.imshow(out_img)
        ax.plot(left_fitx, ploty, color='yellow')
        ax.plot(right_fitx, ploty, color='yellow')
        plt.savefig(name)

    return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit


def fit_polynomial_prior_util(img_shape, leftx, lefty, rightx, righty, lline, rline, degree):
    '''
    util function to fit polynomial
    '''
    
    
    left_fit = np.polyfit(lefty, leftx, degree)
    right_fit = np.polyfit(righty, rightx, degree)
    
    lline.addCurrCoeffs(left_fit)
    rline.addCurrCoeffs(right_fit)
    
    
    left_fit = lline.getSmoothenedCoeffs()
    right_fit = rline.getSmoothenedCoeffs()
            
    # Generate x and y values for plotting
    ploty = np.linspace(25, img_shape[0]-1, img_shape[0])
    
    left_fitx = np.zeros_like(ploty, dtype=np.float64)
    right_fitx = np.zeros_like(ploty, dtype=np.float64)
    for i in range(degree + 1):
        left_fitx += left_fit[i]*(ploty**(degree-i))
        right_fitx += right_fit[i]*(ploty**(degree-i))
            
    lline.addCurrFit(left_fitx, ploty)
    rline.addCurrFit(right_fitx, ploty)
    return left_fitx, right_fitx, ploty, left_fit, right_fit


def fit_polynomial_prior(binary_warped, tparams, lline, rline):
    '''
    this function will searh for lane pixels given line coeffs from previous frame.
    if nothing is found then we return empty matrices
    '''
    global framecount
    
    framecount += 1
    #print(framecount)
    left_fit_old = lline.getSmoothenedCoeffs();
    right_fit_old = rline.getSmoothenedCoeffs();
    margin = tparams.margin

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_fitx_old = np.zeros_like(nonzeroy, dtype=np.float64)
    right_fitx_old = np.zeros_like(nonzeroy, dtype=np.float64)
    for i in range(tparams.degree + 1):
        left_fitx_old += left_fit_old[i]*(nonzeroy**(tparams.degree-i))
        right_fitx_old += right_fit_old[i]*(nonzeroy**(tparams.degree-i))
    left_lane_inds = ((nonzerox > left_fitx_old - margin) & (nonzerox < left_fitx_old + margin))
    right_lane_inds = ((nonzerox > right_fitx_old - margin) & (nonzerox < right_fitx_old + margin))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) > 0 and len(lefty) > 0 and len(rightx) > 0 and len(righty) > 0:
        
        try:
            # Fit new polynomial
            left_fitx, right_fitx, ploty, left_fit, right_fit = fit_polynomial_prior_util(binary_warped.shape, leftx, lefty, rightx, righty, lline, rline, tparams.degree)
            
            ## Visualization ## 
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                      ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                      ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            
            # Plot the polynomial lines onto the image
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            ## End visualization steps ##
        except:
            result, left_fitx, right_fitx, ploty, left_fit, right_fit = [], None, None, None, None, None                
    else:
        result, left_fitx, right_fitx, ploty, left_fit, right_fit = [], None, None, None, None, None
    
    return result, left_fitx, right_fitx, ploty, left_fit, right_fit


def unwarpProject(warped, undist, Minv, left_fitx, right_fitx, ploty):
    '''
    unwarp i.e. go from the birds-eye-view/ top-view back to front-view.
    Here we use the lane projected image in top-view to front-view.
    '''
    
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    return result


def getRealFit(left_fitx, right_fitx, ploty, ym_per_pix, xm_per_pix, degree):
    '''
    get a polynomial which first converts points to real world measurement,
    given the conversion factors from real word to image pixels
    '''
    
    ploty_cr = ploty*ym_per_pix
    left_fit_cr = np.polyfit(ploty_cr, left_fitx*xm_per_pix, degree)
    right_fit_cr = np.polyfit(ploty_cr, right_fitx*xm_per_pix, degree)
    
    return left_fit_cr, right_fit_cr, ploty_cr


def measureCurvatureReal(left_fit_cr, right_fit_cr, y_eval, degree):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    
    # Calculation of R_curve (radius of curvature)
    if degree == 2:
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    elif degree == 3:
        left_curverad = ((1 + (3*left_fit_cr[0]*(y_eval)**2 + 2*left_fit_cr[1]*y_eval + left_fit_cr[2])**2)**1.5) / np.absolute(3*left_fit_cr[0]*y_eval + 2*left_fit_cr[1])
        right_curverad = ((1 + (3*right_fit_cr[0]*(y_eval)**2 + 2*right_fit_cr[1]*y_eval + right_fit_cr[2])**2)**1.5) / np.absolute(3*right_fit_cr[0]*y_eval + 2*right_fit_cr[1])
    
    return left_curverad, right_curverad

 
def measureCamCarOffsetReal(left_fit_cr, right_fit_cr, ymax_r, xmax_cr, degree):
    '''
    measure the car offset in meters
    '''
    
    #left_x = (left_fit_cr[0]*(ymax_r**2)) + (left_fit_cr[1]*ymax_r) + (left_fit_cr[2])
    #right_x = (right_fit_cr[0]*(ymax_r**2)) + (right_fit_cr[1]*ymax_r) + (right_fit_cr[2])
    
    
    left_x = 0
    right_x = 0
    for i in range(degree + 1):
        left_x += left_fit_cr[i]*ymax_r**(degree-i)
        right_x += right_fit_cr[i]*ymax_r**(degree-i)
        
        
    lane_center = np.mean([left_x, right_x])
    img_center = xmax_cr/2
    return img_center-lane_center
    

def measureResult(img, left_fitx, right_fitx, ploty, ym_per_pix, xm_per_pix, degree):
    '''
    calculate the radius of curvature and car offset from lane center and annotate the frame with 
    this data
    '''
    
    left_fit_cr, right_fit_cr, ploty_cr = getRealFit(left_fitx, right_fitx, ploty, ym_per_pix, xm_per_pix, degree)
    y_eval = np.max(ploty_cr)
    lcurve, rcurve = measureCurvatureReal(left_fit_cr, right_fit_cr, y_eval, degree)
    offset = measureCamCarOffsetReal(left_fit_cr, right_fit_cr, y_eval, img.shape[1]*xm_per_pix, degree)
    avg_curve = (lcurve + rcurve)/2
    str1 = 'Radius of curvature: %.1f m' % avg_curve
    result = cv2.putText(img, str1, (500,30), 0, 1, (0,0,0), 2, cv2.LINE_AA)
    str2 = 'Vehicle offset from lane center: %.1f m' % offset
    result = cv2.putText(result, str2, (500,60), 0, 1, (0,0,0), 2, cv2.LINE_AA)
    if np.isnan(avg_curve) or np.isnan(offset):
        return False, result
    else:
        return True, result

 
def imagePipeline(img, tparams, n):
    '''
    pipeline to run over images in test folder.
    this does not use knowledge of detections from previous frames
    '''
    
    undist = cv2.undistort(img, tparams.cameraMatrix, tparams.distCoeffs, None, tparams.cameraMatrix) 
    binary = getBinary(undist, tparams)
    binary_warped = cv2.warpPerspective(binary, tparams.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image
    out_img, left_fitx, right_fitx, ploty, left_fit, right_fit = fit_polynomial(binary_warped, tparams, save=True, name='output_images/test_lanes_out'+str(n)+'.jpg')
    result = unwarpProject(binary_warped, undist, tparams.Minv, left_fitx, right_fitx, ploty)
    ret, measured_result = measureResult(result, left_fitx, right_fitx, ploty, tparams.ym_per_pix, tparams.xm_per_pix, tparams.degree)
    plt.imsave('output_images/test_undist_out'+str(n)+'.jpg', undist, format='jpeg')
    plt.imsave('output_images/test_bin_out'+str(n)+'.jpg', binary, format='jpeg')
    plt.imsave('output_images/test_warped_out'+str(n)+'.jpg', binary_warped, format='jpeg')
    plt.imsave('output_images/test_lanes_out'+str(n)+'.jpg', out_img, format='jpeg')
    plt.imsave('output_images/test_unwarped'+str(n)+'.jpg', result, format='jpeg')
    plt.imsave('output_images/test_measured'+str(n)+'.jpg', measured_result, format='jpeg')
    
    
def videoPipeline(img, tparams, lline, rline):
    '''
    this pipeline is for a video frame where we use the knowledge of the lane detection from the last frame to
    make lane predictions in current frame smoother and faster.
    '''
    
    global failcount1
    undist = cv2.undistort(img, tparams.cameraMatrix, tparams.distCoeffs, None, tparams.cameraMatrix) 
    binary = getBinary(undist, tparams)
    binary_warped = cv2.warpPerspective(binary, tparams.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image
    if lline.detected and rline.detected:
        out_img, left_fitx, right_fitx, ploty, left_fit, right_fit = fit_polynomial_prior(binary_warped, tparams, lline, rline)
        if out_img == []:
            if failcount1 < 25:
                left_fit = lline.getSmoothenedCoeffs()
                right_fit = rline.getSmoothenedCoeffs()
                left_fitx = lline.curr_x
                right_fitx = rline.curr_x
                ploty = lline.curr_y
                failcount1 += 1
                #print(failcount1)
                #print('Failed smoothing: Reusing last val')
                if failcount1 == 25:
                    failcount1 = 0
                    #print(failcount1)
                    lline.detected = False
                    rline.detected = False
            else:
                #print('Failed reusing')
        else:
            #print('Using smoothened coeffs')
    else:
        #print('none')
        out_img, left_fitx, right_fitx, ploty, left_fit, right_fit = fit_polynomial(binary_warped, tparams, save=False)
        if out_img == []:
            out_img = binary_warped
            left_fitx, right_fitx, ploty = 0,0,0
            #print('Failed starting from scratch')
        else:
            lline.addCurrCoeffs(left_fit)
            lline.addCurrFit(left_fitx, ploty)
            rline.addCurrCoeffs(right_fit)
            rline.addCurrFit(right_fitx, ploty)
            #print('Starting from scratch')
    if np.all(ploty) != 0:
        result = unwarpProject(binary_warped, undist, tparams.Minv, left_fitx, right_fitx, ploty)
        ret, measured_result = measureResult(result, left_fitx, right_fitx, ploty, tparams.ym_per_pix, tparams.xm_per_pix, tparams.degree)
        
    else:
#        mpimg.imsave('diagnostics/f_orig'+str(failcount1)+'.jpg', img)
#        mpimg.imsave('diagnostics/f_binarywarped'+str(failcount1)+'.jpg', binary_warped)
#        mpimg.imsave('diagnostics/f_line'+str(failcount1)+'.jpg', out_img)
#        outfile1 = TemporaryFile()
#        outfile2 = TemporaryFile()
#        np.save(outfile1, left_fitx)
#        np.save(outfile2, right_fitx)
        measured_result = img
        
    return measured_result
    
    
   
def processImages():
    '''
    process images in the test_images dir.
    '''
    global tparams
    
    timages = glob.glob('test_images/*.jpg')
    for ii,timage in enumerate(timages):
        #print(ii)
        img = mpimg.imread(timage)
        imagePipeline(img, tparams, ii)


def processVideo(inpVid, outVid):
    '''
    process video
    '''
    
    global tparams
    
    clip1 = VideoFileClip(inpVid)#.subclip(0,10)
    lLine = Line(tparams.smoothing, tparams.degree+1)
    rLine = Line(tparams.smoothing, tparams.degree+1)
    processedVid = clip1.fl_image(lambda img: videoPipeline(img, tparams, lLine, rLine))
    
    
    processedVid.write_videofile(outVid, audio=False)
    clip1.close()
    
failcount1 = 0
tparams = TunableParams()
processImages()
processVideo('project_video.mp4', 'project_out_video.mp4')
processVideo('challenge_video.mp4', 'challenge_out_video.mp4')
processVideo('harder_challenge_video.mp4', 'hchallenge_out_video.mp4')
