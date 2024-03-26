import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import os

# Read the image into grayscale
sample = cv2.imread('../data/breakfast2.png')

# Define output directory
output_dir = '../output/task2'
os.makedirs(output_dir, exist_ok=True)

# Display and save grayscale image
sample_small = cv2.resize(sample, (640, 480))
cv2.imshow('Grey scale image', sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(output_dir, 'grey_scale_image.png'), sample_small)

# Convert the original image to HSV and take H channel
sample_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
sample_h = sample_hsv[:, :, 0]

# Show the H channel of the image
sample_small = cv2.resize(sample_h, (640, 480))
cv2.imshow('H channel of the image', sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(output_dir, 'h_channel_image.png'), sample_small)

# Convert the original image to grayscale
sample_grey = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

# Show the grey scale image
sample_small = cv2.resize(sample_grey, (640, 480))
cv2.imshow('Grey scale image', sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(output_dir, 'grey_scale_image2.png'), sample_small)

# Binarize the image using Otsu's method
ret1, binary_image = cv2.threshold(sample_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

sample_small = cv2.resize(binary_image, (640, 480))
cv2.imshow('Image after Otsu''s thresholding', sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(output_dir, 'otsu_thresholded_image.png'), sample_small)

# Filling the holes
im_floodfill = binary_image.copy()
h, w = binary_image.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(im_floodfill, mask, (0, 0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
binary_image = binary_image | im_floodfill_inv  # Bitwise OR operation

# Define the kernel for morphological operations
kernel = np.ones((9, 9), np.uint8)

binary_image = cv2.erode(binary_image, kernel, iterations=1)
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

sample_small = cv2.resize(binary_image, (640, 480))
cv2.imshow('Image after morphological transformation', sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(output_dir, 'morphological_processed_image.png'), sample_small)

# Find connected pixels and compose them into objects
labels = measure.label(binary_image)

# Calculate features for each object
properties = measure.regionprops(labels)

# Calculate features for each object
features = np.zeros((len(properties), 2))

for i in range(0, len(properties)):
    min_row, min_col, max_row, max_col = properties[i].bbox
    object_mask = labels[min_row:max_row, min_col:max_col] == (i + 1)
    hue_channel_masked = np.where(object_mask, sample_h[min_row:max_row, min_col:max_col], 0)
    local_avg_intensity_h = np.mean(hue_channel_masked[hue_channel_masked != 0])
    features[i, 0] = properties[i].perimeter
    features[i, 1] = local_avg_intensity_h

# Choose the thresholds for features
thrF1 = 275
thrF2 = 45

# Show objects in the feature space
plt.plot(features[:, 0], features[:, 1], 'yo', label='Squares')
plt.plot(features[(features[:, 0] < thrF1) & (features[:, 1] > thrF2), 0],
         features[(features[:, 0] < thrF1) & (features[:, 1] > thrF2), 1],
         'bo', label='Blue Circles')
plt.plot(features[(features[:, 0] < thrF1) & (features[:, 1] < thrF2), 0],
         features[(features[:, 0] < thrF1) & (features[:, 1] < thrF2), 1],
         'go', label='Red Circles')

plt.xlabel('Feature 1: Perimeter')
plt.ylabel('Feature 2: LAI')
plt.legend()
plt.savefig(os.path.join(output_dir, 'feature_plot.png'))
plt.show()

# Classify, count, and display the objects
squares = 0
blue_circles = 0
red_circles = 0

fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))

for i in range(0, len(properties)):
    if (features[i, 0] > thrF1 and features[i, 1] < thrF2):
        squares += 1
        ax.plot(np.round(properties[i].centroid[1]), np.round(properties[i].centroid[0]), '.g', markersize=15)

    if (features[i, 0] < thrF1 and features[i, 1] > thrF2):
        blue_circles += 1
        ax.plot(np.round(properties[i].centroid[1]), np.round(properties[i].centroid[0]), '.b', markersize=15)

    if (features[i, 0] < thrF1 and features[i, 1] < thrF2):
        red_circles += 1
        ax.plot(np.round(properties[i].centroid[1]), np.round(properties[i].centroid[0]), '.r', markersize=15)

plt.savefig(os.path.join(output_dir, 'classified_image.png'))
plt.show()

# That's all! Let's display the result:
print("I found %d squares, %d blue donuts, and %d red donuts." % (squares, blue_circles, red_circles))
