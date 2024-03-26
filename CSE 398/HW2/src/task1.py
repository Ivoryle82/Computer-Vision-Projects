import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import os

# Read the image as grayscale
sample = cv2.imread('../data/breakfast1.png', cv2.IMREAD_GRAYSCALE)

# Define output directory
output_dir = '../output/task1'
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

# Define the kernel for morphological operations
kernel = np.ones((10, 10), np.uint8)

# Perform morphological closing
binary_image_cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
binary_image_cleaned = cv2.erode(binary_image_cleaned, kernel, iterations=2)

# Display and save image after morphological operations
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

# Calculate the ratio between the major and minor axes
his = [features_cleaned[i].major_axis_length / features_cleaned[i].minor_axis_length
       for i in range(len(features_cleaned)) if features_cleaned[i].minor_axis_length > 0]

# Plot and save histogram
plt.hist(his)
plt.xlabel("Ratio")
plt.ylabel("Count")
plt.savefig(os.path.join(output_dir, 'histogram.png'))
plt.show()

# Select a proper threshold
fThr = 1.6

# Count squares and cashews, display and save the classified image
squares = 0
cashews = 0

fig, ax = plt.subplots()
ax.imshow(sample, cmap=plt.cm.gray)

for i in range(len(his)):
    if his[i] <= fThr:
        squares += 1
        y, x = features_cleaned[i].centroid
        ax.plot(x, y, '.g', markersize=10)
    else:
        cashews += 1
        y, x = features_cleaned[i].centroid
        ax.plot(x, y, '.b', markersize=10)

plt.savefig(os.path.join(output_dir, 'classified_image.png'))
plt.show()

# Display the result
print("I found %d squares and %d cashew nuts." % (squares, cashews))
