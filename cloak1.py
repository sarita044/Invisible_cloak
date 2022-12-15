import cv2
import numpy as np
import time
  
# replace the red pixels ( or undesired area ) with
# background pixels to generate the invisibility feature.
  

  
# in order to check the cv2 version
# print(cv2.__version__)   
  
# taking video.mp4 as input.
capture_video = cv2.VideoCapture(0)
     
# give the camera to warm up
time.sleep(3) 
count = 0 
background = 0 
  
# capturing the background in range of 60
# first  few seconds are dedicated to background frame so that it 
# could easily save the background image
for i in range(60):
    return_val, background = capture_video.read()
    if return_val == False :
        continue 
  
background = np.flip(background, axis = 1) # flipping of the frame 
  
# we are reading from video 
while (capture_video.isOpened()):
    return_val, img = capture_video.read()
    if not return_val :
        break 
    count = count + 1
    img = np.flip(img, axis = 1)
  
    # convert the image - BGR to HSV
    # as we focused on detection of red color 
  
    # converting BGR to HSV for better detection 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
     # ranges should be carefully chosen
    # setting the lower and upper range for mask1
    # lower_red = np.array([100, 40, 40])       
    # upper_red = np.array([100, 255, 255])
    # mask1 = cv2.inRange(hsv, lower_red, upper_red)
    # # setting the lower and upper range for mask2 
    # lower_red = np.array([155, 40, 40])
    # upper_red = np.array([180, 255, 255])
    # mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    # setting the lower and upper range for mask1
    
    lower_blue = np.array([40, 90, 40]) 
    upper_blue = np.array([255, 255, 160])
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    # setting the lower and upper range for mask2 
   
    lower_blue = np.array([85, 90, 40])
    upper_blue = np.array([200,255, 255])
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
  
   
    mask1 = mask1 + mask2
  
    # Refining the mask corresponding to the detected red color
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3),
                                         np.uint8), iterations = 2)
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations = 1)
    mask2 = cv2.bitwise_not(mask1)
  
    # Generating the final output
    res1 = cv2.bitwise_and(background, background, mask = mask1)
    res2 = cv2.bitwise_and(img, img, mask = mask2)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
  
    cv2.imshow("INVISIBLE MAN", final_output)

    k = cv2.waitKey(10)
    if k == 27:
        break