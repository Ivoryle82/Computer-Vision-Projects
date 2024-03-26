import os
import cv2
import numpy as np

def convolve(image, kernel):
    (iH, iW, iD) = image.shape[:3]
    (kH, kW) = kernel.shape[:2]

    padW = (kW - 1) // 2
    padH = (kH - 1) // 2
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW, iD), dtype="float32")

    for y in np.arange(padH, iH + padH):
        for x in np.arange(padW, iW + padW):
            for z in range(3):
                roi = image[y - padH:y + padH + 1, x - padW:x + padW + 1, z]
                k = (roi * kernel).sum()
                output[y - padH, x - padW, z] = k

    return output

def rescale_intensity(img, in_range=(0, 255)):
    min_val, max_val = in_range
    img = np.clip(img, min_val, max_val)
    return img

def display_save(title, img, output_path):
    # Rescale intensity
    img_rescaled = rescale_intensity(img, in_range=(0, 255))
    
    # Convert to uint8
    img_uint8 = img_rescaled.astype("uint8")

    # Create the directory if it doesn't exist
    output_folder = os.path.dirname(output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save image
    cv2.imwrite(output_path, img_uint8)

# Function for median blur
def median_blur(image, kernel_size):
    # Copy the input image to avoid modifying the original image
    output_image = np.copy(image)
    padding = kernel_size // 2
    
    # Iterate over each pixel in the image
    for i in range(padding, image.shape[0] - padding):
        for j in range(padding, image.shape[1] - padding):
            for k in range (3):
                # Extract the neighborhood of the current pixel
                neighborhood = image[i - padding:i + padding + 1, j - padding:j + padding + 1,k]
                # Compute the median value of the neighborhood
                median_value = np.median(neighborhood)
                # Assign the median value to the corresponding pixel in the output image
                output_image[i, j, k] = median_value
            
    return output_image 

# Mean filter kernel
mean_filter = np.ones((3, 3), dtype="float") * (1.0 / 9)

# Gaussian filter kernel
gaussian_filter = np.array((
    [1,  2,  1,  2,  1],
    [2,  4,  8,  4,  2],
    [1,  8, 16,  8,  1],
    [2,  4,  8,  4,  2],
    [1,  2,  1,  2,  1]), dtype="float") / 64 

# Specify the path to the folder containing the images
input_folder = '/Users/ivoryle/Desktop/CSE 398/HW1/data/input2'
output_folder = '/Users/ivoryle/Desktop/CSE 398/HW1/output'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
# Specify the subfolder name
subfolder_name = 'noise'

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

# Loop over each image file
for image_file in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(input_folder, image_file)

    # Load the input image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Loop over each filter in the kernel bank
    for (filter_name, filter_kernel) in [("mean filter", mean_filter), ("gaussian filter", gaussian_filter)]:
        # Apply the filter to the image
        filtered_image = convolve(image, filter_kernel)

        # Write the filtered image to the output folder
        output_path = os.path.join(output_folder, subfolder_name, f"{os.path.splitext(image_file)[0]}_{filter_name}.jpg")
        display_save(filter_name, filtered_image, output_path)

    # Apply median filter
    median_filtered_image = median_blur(image, 3)

    # Write the median filtered image to the output folder
    output_path = os.path.join(output_folder, subfolder_name, f"{os.path.splitext(image_file)[0]}_median.jpg")
    display_save("Median filter", median_filtered_image, output_path)
