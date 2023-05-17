import cv2
import numpy as np

# Load the iris image
img = cv2.imread("CASIA2/001_1_2.jpg")
cv2.imshow('i',img)
cv2.waitKey(0)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the gray ranges for each region
pupil_range = [0, 50]
iris_range = [45, 130]
sclera_range = [80, 150]
eyelash_range = [150, 255]

# Apply adaptive binary threshold segmentation to extract each region
pupil = cv2.inRange(gray, pupil_range[0], pupil_range[1])
iris = cv2.inRange(gray, iris_range[0], iris_range[1])
sclera = cv2.inRange(gray, sclera_range[0], sclera_range[1])
eyelash = cv2.inRange(gray, eyelash_range[0], eyelash_range[1])

# Refine each region with erosion, dilation, and maximum connected area operations
kernel = np.ones((5, 5), np.uint8)
pupil = cv2.erode(pupil, kernel, iterations=1)
pupil = cv2.dilate(pupil, kernel, iterations=1)
pupil = cv2.morphologyEx(pupil, cv2.MORPH_CLOSE, kernel)

iris = cv2.erode(iris, kernel, iterations=1)
iris = cv2.dilate(iris, kernel, iterations=1)
iris = cv2.morphologyEx(iris, cv2.MORPH_CLOSE, kernel)

sclera = cv2.erode(sclera, kernel, iterations=1)
sclera = cv2.dilate(sclera, kernel, iterations=1)
sclera = cv2.morphologyEx(sclera, cv2.MORPH_CLOSE, kernel)

eyelash = cv2.erode(eyelash, kernel, iterations=1)
eyelash = cv2.dilate(eyelash, kernel, iterations=1)
eyelash = cv2.morphologyEx(eyelash, cv2.MORPH_CLOSE, kernel)

# Combine the iris, sclera, and eyelash regions to extract the pupil region
mask = iris + sclera + eyelash
pupil1 = cv2.bitwise_and(pupil, mask)

pupil=cv2.bitwise_xor(pupil,pupil1)
#pupil=cv2.bitwise_not(pupil)

#Find the contours of the pupil region
contours, hierarchy = cv2.findContours(pupil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Find the contour with the largest area (assuming it is the pupil)
max_area = 0
max_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

# Find the center and radius of the largest contour
(x,y), radius = cv2.minEnclosingCircle(max_contour)
center = (int(x), int(y))
radius = int(radius)

# Draw a circle around the pupil
cv2.circle(img, center, radius, (0, 255, 0), 2)

# Display the result

cv2.imshow("Pupil", pupil)
cv2.imshow("Iris", iris)
cv2.imshow("Sclera", sclera)
cv2.imshow("Eyelash", eyelash)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Pupil", img)

cv2.waitKey(0)