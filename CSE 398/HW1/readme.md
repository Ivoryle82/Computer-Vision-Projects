# Spring 2024 CSE 398 HW1: Image Processing

**Student Name:** Ivory Le

---

## Task 1: Image Sharpening

To sharpen the image, we can apply a filter to emphasize the edges. One common matrix that can achieve this effect is:

[0 -1 0]

[-1 4 -1]

[0 -1 0]

This filter is a Laplacian filter enhances the contrast, emphasizing rapid intensity, making them appear sharper.

## Task 2: Filter for Noise Reduncing

The median filter is the preferred choice for handling outliers in images. In our case, the image contains outliers such as black or white pixels, which can affect the results of other filters.

When applying a mean filter, these outliers can lead to a grainy appearance in the filtered image. This is because the mean filter considers all pixel values, including the outliers, resulting in a loss of detail.

On the other hand, a Gaussian filter can blur out the image but fails to address the outlier issue effectively.

In contrast, the median filter selectively chooses the median value from the neighborhood of each pixel. By disregarding extreme outlier values, it produces a smoother image while preserving important details.

---


