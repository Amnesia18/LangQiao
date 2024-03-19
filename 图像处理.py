import cv2
import numpy as np

# Load the image
image = cv2.imread('E:\\SideViewCamera\\43_7_SideViewCamera.jpg')

# Apply Gaussian blur
gaussian_blur = cv2.GaussianBlur(image, (3, 3), 0)

# Enhance contrast
# Convert to YUV and equalize the histogram of the Y channel
image_yuv = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2YUV)
image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
contrast_enhanced = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

# Increase brightness
brightness_increase = 40
brighter_image = cv2.add(contrast_enhanced, (brightness_increase, brightness_increase, brightness_increase, 0))

# Save or display the result

cv2.imshow('Result', brighter_image)
cv2.waitKey(0)
cv2.destroyAllWindows()