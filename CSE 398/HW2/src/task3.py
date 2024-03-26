import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import os

# Read the image as grayscale
sample = cv2.imread('../data/pills.png', cv2.IMREAD_GRAYSCALE)

# Define output directory
output_dir = 'output/task3'
os.makedirs(output_dir, exist_ok=True)

# Display and save grayscale image
sample_small = cv2.resize(sample, (640, 480))
cv2.imshow('Grey scale image', sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(output_dir, 'grey_scale_image.png'), sample_small)

# Binarize the image using Otsu's method
ret1, binary_image = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

sample_small = cv2.resize(binary_image, (640, 480))
cv2.imshow('Image after Otsu''s thresholding', sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(output_dir, 'otsu_thresholded_image.png'), sample_small)

# Apply morphological operations
kernel = np.ones((14, 14), np.uint8)
binary_image_cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
binary_image_cleaned = cv2.erode(binary_image_cleaned, kernel, iterations=1)
binary_image_cleaned = cv2.dilate(binary_image_cleaned, kernel, iterations=1)

sample_small = cv2.resize(binary_image_cleaned, (640, 480))
cv2.imshow('Image after morphological operations', sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(output_dir, 'morphological_processed_image.png'), sample_small)

# Find connected pixels and group them into objects
labels_cleaned = measure.label(binary_image_cleaned, connectivity=2, background=0)

# Calculate features for each object after noise removal
features_cleaned = measure.regionprops(labels_cleaned)

print("I found %d objects in total." % (len(features_cleaned)))

# Calculate the ratio between major and minor axes for each object
his = []
for i in range(0, len(features_cleaned)):
    if features_cleaned[i].minor_axis_length > 0:
        his.append(features_cleaned[i].major_axis_length / features_cleaned[i].minor_axis_length)

# Plot histogram
plt.hist(his)
plt.xlabel("Ratio")
plt.ylabel("Count")
plt.savefig(os.path.join(output_dir, 'ratio_histogram.png'))
plt.show()

# Select a proper threshold
fThr = 1.5

# Classify, count and display the objects
pink_pill = 0
white_pill = 0

fig, ax = plt.subplots()
ax.imshow(sample, cmap=plt.cm.gray)

for i in range(0, len(his)):
    if his[i] <= fThr:
        pink_pill += 1
        y, x = features_cleaned[i].centroid
        ax.plot(x, y, '.g', markersize=10)
    else:
        white_pill += 1
        y, x = features_cleaned[i].centroid
        ax.plot(x, y, '.b', markersize=10)

plt.savefig(os.path.join(output_dir, 'classified_image.png'))
plt.show()

# Display the result
print("I found %d pink pills, and %d white pills." % (pink_pill, white_pill))
